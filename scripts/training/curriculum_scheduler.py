#!/usr/bin/env python3
"""
Curriculum training scheduler for progressive difficulty training.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-17_phase4_curriculum.md

Key Components:
    - 4-stage curriculum: easy → easy+medium → medium+hard → hard
    - Composite reward integration
    - Progressive model checkpointing

Usage:
    python scripts/training/curriculum_scheduler.py \
        --model /data/qwen3_or_debug_merged \
        --benchmarks_dir data/benchmarks/per_type \
        --output /data/curriculum_output \
        --num_stages 4 \
        --epochs_per_stage 2
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

from src.training.composite_reward import composite_reward_func


# Curriculum stages
CURRICULUM_STAGES = [
    {
        "name": "stage_1_easy",
        "description": "Easy problems only - build foundation",
        "difficulties": ["easy"],
        "epochs": 2,
    },
    {
        "name": "stage_2_easy_medium",
        "description": "Easy + Medium problems - gradual increase",
        "difficulties": ["easy", "medium"],
        "epochs": 2,
    },
    {
        "name": "stage_3_medium_hard",
        "description": "Medium + Hard problems - challenge",
        "difficulties": ["medium", "hard"],
        "epochs": 2,
    },
    {
        "name": "stage_4_hard",
        "description": "Hard problems only - mastery",
        "difficulties": ["hard"],
        "epochs": 2,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Curriculum training scheduler")

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="/data/qwen3_or_debug_merged",
        help="Path to base model or previous stage checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/curriculum_output",
        help="Base output directory"
    )

    # Data args
    parser.add_argument(
        "--benchmarks_dir",
        type=str,
        default="data/benchmarks/per_type",
        help="Directory containing per-type benchmarks"
    )
    parser.add_argument(
        "--error_types",
        type=str,
        nargs="+",
        default=["D"],
        help="Error types to use (A, B, C, D)"
    )

    # Training args
    parser.add_argument("--num_stages", type=int, default=4, help="Number of curriculum stages")
    parser.add_argument("--epochs_per_stage", type=int, default=2, help="Epochs per stage")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt")

    # GRPO args
    parser.add_argument("--beta", type=float, default=0.0, help="KL penalty (0 for DAPO-style)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")

    # LoRA args
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Control
    parser.add_argument("--start_stage", type=int, default=1, help="Stage to start from (1-indexed)")
    parser.add_argument("--continue_from", type=str, default=None, help="Continue from checkpoint")

    return parser.parse_args()


def load_benchmark_dataset(
    benchmarks_dir: str,
    error_types: List[str],
    difficulties: List[str],
) -> Dataset:
    """
    Load and combine benchmark datasets for specified error types and difficulties.

    Args:
        benchmarks_dir: Base directory containing benchmarks
        error_types: List of error types (A, B, C, D)
        difficulties: List of difficulties (easy, medium, hard)

    Returns:
        Combined Dataset with 'prompt' column
    """
    all_prompts = []

    for error_type in error_types:
        for difficulty in difficulties:
            benchmark_name = f"type_{error_type}_{difficulty}"
            dataset_path = Path(benchmarks_dir) / benchmark_name / "dataset.json"

            if not dataset_path.exists():
                print(f"  [WARN] Dataset not found: {dataset_path}")
                continue

            with open(dataset_path, 'r') as f:
                data = json.load(f)

            problems = data.get('problems', data) if isinstance(data, dict) else data

            for i, problem in enumerate(problems):
                problem_desc = problem.get('problem_description', problem.get('description', 'Unknown'))
                model_file = problem.get('model_file', '')
                iis_info = problem.get('iis_constraints', [])

                system_msg = """You are an expert OR debugger. You must diagnose and repair infeasible optimization models.

Available actions:
- GET_IIS(): Get Irreducible Inconsistent Subsystem
- DROP_CONSTRAINT(name): Remove a constraint
- RELAX_CONSTRAINT(name, delta): Relax RHS by delta
- SUBMIT(): Submit final solution

Always wrap your reasoning in <think>...</think> tags. Identify the IIS constraints causing infeasibility."""

                user_msg = f"""## Problem Description
{problem_desc}

## Model File
{model_file}

## Solver Status
INFEASIBLE

## IIS Constraints
{json.dumps(iis_info)}

Diagnose the infeasibility and propose a repair action."""

                prompt = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ]

                all_prompts.append({
                    "prompt": prompt,
                    "problem_id": problem.get('problem_id', f'{benchmark_name}_{i}'),
                    "model_file": model_file,
                    "iis_constraints": json.dumps(iis_info) if isinstance(iis_info, list) else iis_info,
                    "difficulty": difficulty,
                    "error_type": error_type,
                })

    print(f"  Loaded {len(all_prompts)} problems")
    return Dataset.from_list(all_prompts)


def run_training_stage(
    stage_config: Dict,
    model_path: str,
    output_dir: Path,
    benchmarks_dir: str,
    error_types: List[str],
    args,
) -> str:
    """
    Run a single curriculum training stage.

    Args:
        stage_config: Stage configuration dict
        model_path: Path to model checkpoint
        output_dir: Output directory for this stage
        benchmarks_dir: Benchmarks directory
        error_types: Error types to use
        args: Command line arguments

    Returns:
        str: Path to final checkpoint for this stage
    """
    stage_name = stage_config["name"]
    difficulties = stage_config["difficulties"]
    epochs = stage_config.get("epochs", args.epochs_per_stage)

    print(f"\n{'='*60}")
    print(f"Curriculum Stage: {stage_name}")
    print(f"{'='*60}")
    print(f"Description: {stage_config['description']}")
    print(f"Difficulties: {difficulties}")
    print(f"Epochs: {epochs}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save stage config
    config_file = output_dir / "stage_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "stage": stage_config,
            "model_path": model_path,
            "args": vars(args),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    # Load dataset for this stage
    print(f"\nLoading datasets for difficulties: {difficulties}")
    dataset = load_benchmark_dataset(
        benchmarks_dir=benchmarks_dir,
        error_types=error_types,
        difficulties=difficulties,
    )

    if len(dataset) == 0:
        print(f"  [ERROR] No data loaded for stage {stage_name}")
        return model_path

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure GRPO
    print("\nConfiguring GRPOTrainer...")
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),

        # Batch settings
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        # GRPO settings
        num_generations=args.num_generations,
        max_completion_length=512,
        temperature=args.temperature,

        # Learning
        learning_rate=args.learning_rate,
        num_train_epochs=epochs,
        beta=args.beta,

        # Logging
        logging_steps=5,
        save_steps=100,
        report_to="tensorboard",
        logging_dir=str(output_dir / "tensorboard"),

        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,

        # Memory
        gradient_checkpointing=True,
        bf16=True,
    )

    # Configure LoRA
    print(f"\nConfiguring LoRA (r={args.lora_r}, alpha={args.lora_alpha})...")
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create trainer
    print("\nCreating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model_path,
        args=grpo_config,
        reward_funcs=composite_reward_func,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print(f"\nStarting training for {stage_name}...")
    print("=" * 60)
    trainer.train()

    # Save final checkpoint
    final_output = output_dir / "final"
    print(f"\nSaving checkpoint to: {final_output}")
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))

    # Save training summary
    summary_file = output_dir / "training_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "stage": stage_name,
            "dataset_size": len(dataset),
            "epochs": epochs,
            "final_checkpoint": str(final_output),
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Stage {stage_name} complete!")
    print(f"{'='*60}")

    return str(final_output)


def merge_lora_adapter(adapter_path: str, base_model: str, output_path: str):
    """
    Merge LoRA adapter with base model for next stage.

    Args:
        adapter_path: Path to LoRA adapter
        base_model: Path to base model
        output_path: Path to save merged model
    """
    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    print(f"\nMerging LoRA adapter...")
    print(f"  Adapter: {adapter_path}")
    print(f"  Base: {base_model}")
    print(f"  Output: {output_path}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Load and merge adapter
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Save merged model
    model.save_pretrained(output_path)

    # Copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    tokenizer.save_pretrained(output_path)

    print(f"  Merged model saved to: {output_path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("Curriculum Training Scheduler")
    print("=" * 60)
    print(f"Base model: {args.model}")
    print(f"Benchmarks: {args.benchmarks_dir}")
    print(f"Output: {args.output}")
    print(f"Error types: {args.error_types}")
    print(f"Stages: {args.num_stages}")
    print(f"Epochs per stage: {args.epochs_per_stage}")
    print("=" * 60)

    # Create output directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)

    # Save overall config
    config_file = output_base / "curriculum_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "args": vars(args),
            "stages": CURRICULUM_STAGES[:args.num_stages],
            "started_at": datetime.now().isoformat(),
        }, f, indent=2)

    # Track progress
    current_model = args.continue_from if args.continue_from else args.model
    stage_results = []

    # Run curriculum stages
    for i, stage_config in enumerate(CURRICULUM_STAGES[:args.num_stages], 1):
        if i < args.start_stage:
            print(f"\nSkipping stage {i} (starting from stage {args.start_stage})")
            continue

        stage_output = output_base / stage_config["name"]

        # Run training stage
        checkpoint_path = run_training_stage(
            stage_config=stage_config,
            model_path=current_model,
            output_dir=stage_output,
            benchmarks_dir=args.benchmarks_dir,
            error_types=args.error_types,
            args=args,
        )

        stage_results.append({
            "stage": stage_config["name"],
            "checkpoint": checkpoint_path,
        })

        # Merge adapter for next stage (if not last stage)
        if i < args.num_stages:
            merged_path = str(output_base / f"{stage_config['name']}_merged")
            merge_lora_adapter(
                adapter_path=checkpoint_path,
                base_model=current_model,
                output_path=merged_path,
            )
            current_model = merged_path

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final results
    results_file = output_base / "curriculum_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "stages": stage_results,
            "final_model": current_model,
            "completed_at": datetime.now().isoformat(),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("Curriculum Training Complete!")
    print("=" * 60)
    print(f"Final model: {current_model}")
    print(f"Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()

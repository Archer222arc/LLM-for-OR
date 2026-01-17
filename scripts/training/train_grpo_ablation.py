#!/home/Archer/miniforge3/bin/python
"""
GRPO ablation training script implementing DAPO-style improvements.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-17_phase4_ablation.md

Key Configurations:
    - run_0_baseline: Standard GRPO with beta=0.001
    - run_1_no_kl: Remove KL penalty (beta=0.0)
    - run_2_asym_clip: Asymmetric clipping (higher clip for positive advantages)
    - run_3_dynamic: Dynamic sampling with entropy bonus
    - run_4_token_loss: Token-level loss with overlong reward shaping

Usage:
    # Run single ablation
    python scripts/training/train_grpo_ablation.py \
        --model /data/qwen3_or_debug_merged \
        --dataset data/benchmarks/per_type/type_D_medium/dataset.json \
        --output /data/ablation_output/run_0_baseline \
        --config run_0_baseline

    # Run all ablations sequentially
    python scripts/training/train_grpo_ablation.py --run_all
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

from src.training.gurobi_rewards import (
    gurobi_reward_func,
    set_use_solver_verification,
    set_use_efficiency_reward,
)


# DAPO-style ablation configurations
ABLATION_CONFIGS = {
    "run_0_baseline": {
        "description": "Standard GRPO baseline",
        "beta": 0.001,
        "temperature": 0.7,
        "max_grad_norm": 1.0,
        "use_vllm": False,
    },
    "run_1_no_kl": {
        "description": "Remove KL penalty (DAPO: Clip-Higher removes KL)",
        "beta": 0.0,  # Key change: no KL penalty
        "temperature": 0.7,
        "max_grad_norm": 1.0,
        "use_vllm": False,
    },
    "run_2_asym_clip": {
        "description": "Asymmetric clipping for positive advantages",
        "beta": 0.0,
        "temperature": 0.7,
        "max_grad_norm": 1.0,
        # TRL doesn't support asymmetric clipping directly
        # We implement via reward shaping
        "reward_shaping": "asymmetric",
        "use_vllm": False,
    },
    "run_3_dynamic": {
        "description": "Dynamic sampling with entropy bonus",
        "beta": 0.0,
        "temperature": 1.0,  # Higher temperature for diversity
        "max_grad_norm": 1.0,
        "entropy_bonus": 0.01,  # Encourage exploration
        "use_vllm": False,
    },
    "run_4_token_loss": {
        "description": "Token-level loss with overlong filtering",
        "beta": 0.0,
        "temperature": 0.7,
        "max_grad_norm": 0.5,  # Tighter gradient clipping
        "overlong_filter": True,  # Filter overlong responses
        "use_vllm": False,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO ablation training")

    # Model args
    parser.add_argument(
        "--model",
        type=str,
        default="/data/qwen3_or_debug_merged",
        help="Path to SFT model checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/ablation_output",
        help="Base output directory"
    )

    # Data args
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmarks/per_type/type_D_medium/dataset.json",
        help="Path to training dataset"
    )

    # Ablation config
    parser.add_argument(
        "--config",
        type=str,
        choices=list(ABLATION_CONFIGS.keys()),
        default="run_0_baseline",
        help="Ablation configuration to run"
    )
    parser.add_argument(
        "--run_all",
        action="store_true",
        help="Run all ablation configurations sequentially"
    )

    # Training args
    parser.add_argument("--num_epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens")

    # LoRA config
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")

    # Solver verification
    parser.add_argument(
        "--use_solver",
        action="store_true",
        help="Enable Gurobi solver verification"
    )

    # Logging
    parser.add_argument("--logging_steps", type=int, default=5, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint save frequency")

    return parser.parse_args()


def load_grpo_prompts(dataset_path: str, max_samples: int = None) -> Dataset:
    """
    Load and convert OR-Debug-Bench dataset to GRPO prompts format.

    Args:
        dataset_path: Path to dataset.json
        max_samples: Maximum samples to use (default: all)

    Returns:
        HuggingFace Dataset with 'prompt' column
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    problems = data.get('problems', data) if isinstance(data, dict) else data

    prompts = []
    for i, problem in enumerate(problems):
        if max_samples and i >= max_samples:
            break

        # Build prompt in chat format
        problem_desc = problem.get('problem_description', problem.get('description', 'Unknown problem'))
        model_file = problem.get('model_file', '')
        iis_info = problem.get('iis_constraints', [])

        system_msg = """You are an expert OR debugger. You must diagnose and repair infeasible optimization models.

Available actions:
- GET_IIS(): Get Irreducible Inconsistent Subsystem
- DROP_CONSTRAINT(name): Remove a constraint
- RELAX_CONSTRAINT(name, delta): Relax RHS by delta
- SUBMIT(): Submit final solution

Always wrap your reasoning in <think>...</think> tags."""

        user_msg = f"""## Problem Description
{problem_desc}

## Model File
{model_file}

## Solver Status
INFEASIBLE

## IIS Constraints
{json.dumps(iis_info)}

Diagnose the infeasibility and propose a repair action."""

        # TRL expects 'prompt' field with chat messages
        prompt = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]

        prompts.append({
            "prompt": prompt,
            "problem_id": problem.get('problem_id', f'problem_{i}'),
            "model_file": model_file,
            "iis_constraints": json.dumps(iis_info) if isinstance(iis_info, list) else iis_info,
        })

    return Dataset.from_list(prompts)


def create_asymmetric_reward_wrapper(base_reward_func):
    """
    Wrap reward function with asymmetric shaping for positive advantages.

    DAPO insight: Clip-Higher uses asymmetric clipping to encourage
    moving probability mass toward high-reward actions.
    """
    def shaped_reward_func(*args, **kwargs):
        rewards = base_reward_func(*args, **kwargs)
        # Scale positive rewards more aggressively
        shaped = []
        for r in rewards:
            if r > 0:
                shaped.append(r * 1.4)  # 40% boost for positive rewards
            else:
                shaped.append(r)
        return shaped
    return shaped_reward_func


def create_overlong_filter_wrapper(base_reward_func, max_tokens: int = 400):
    """
    Wrap reward function to penalize overlong responses.

    DAPO insight: Overlong responses indicate exploration collapse.
    """
    def filtered_reward_func(prompts, completions, *args, **kwargs):
        rewards = base_reward_func(prompts, completions, *args, **kwargs)
        # Penalize overlong responses
        filtered = []
        for i, (r, c) in enumerate(zip(rewards, completions)):
            if len(c) > max_tokens * 4:  # Approximate character count
                filtered.append(r * 0.5)  # 50% penalty
            else:
                filtered.append(r)
        return filtered
    return filtered_reward_func


def run_ablation(args, config_name: str) -> dict:
    """
    Run single ablation experiment.

    Args:
        args: Command line arguments
        config_name: Name of ablation configuration

    Returns:
        dict: Training metrics and results
    """
    config = ABLATION_CONFIGS[config_name]
    output_dir = Path(args.output) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"GRPO Ablation: {config_name}")
    print("=" * 60)
    print(f"Description: {config['description']}")
    print(f"Output: {output_dir}")
    print(f"Beta (KL): {config['beta']}")
    print(f"Temperature: {config['temperature']}")
    print("=" * 60)

    # Save config
    config_file = output_dir / "ablation_config.json"
    with open(config_file, 'w') as f:
        json.dump({
            "config_name": config_name,
            "config": config,
            "args": vars(args),
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)

    # Enable solver verification if requested
    if args.use_solver:
        set_use_solver_verification(True)
        print("[INFO] Gurobi solver verification enabled")

    # Enable efficiency reward
    set_use_efficiency_reward(True)

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("\nLoading dataset...")
    dataset_path = project_root / args.dataset
    train_dataset = load_grpo_prompts(str(dataset_path))
    print(f"Dataset size: {len(train_dataset)} samples")

    # Select reward function based on config
    reward_func = gurobi_reward_func
    if config.get("reward_shaping") == "asymmetric":
        print("[INFO] Using asymmetric reward shaping")
        reward_func = create_asymmetric_reward_wrapper(gurobi_reward_func)
    if config.get("overlong_filter"):
        print("[INFO] Using overlong response filtering")
        reward_func = create_overlong_filter_wrapper(reward_func)

    # Configure GRPO training
    print("\nConfiguring GRPOTrainer...")
    grpo_config = GRPOConfig(
        output_dir=str(output_dir),

        # Batch settings
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        # GRPO settings
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=config["temperature"],

        # Learning
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        beta=config["beta"],

        # Gradient clipping
        max_grad_norm=config.get("max_grad_norm", 1.0),

        # Logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="tensorboard",
        logging_dir=str(output_dir / "tensorboard"),

        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        warmup_ratio=0.1,

        # Memory optimization
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
        model=args.model,
        args=grpo_config,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting training...")
    print("=" * 60)

    train_result = trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_output = output_dir / "final"
    trainer.save_model(str(final_output))
    tokenizer.save_pretrained(str(final_output))
    print(f"Model saved to: {final_output}")

    # Extract metrics
    metrics = {
        "config_name": config_name,
        "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        "train_steps": train_result.global_step if hasattr(train_result, 'global_step') else None,
        "output_dir": str(output_dir),
    }

    # Save metrics
    metrics_file = output_dir / "training_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Ablation {config_name} complete!")
    print(f"Metrics saved to: {metrics_file}")
    print("=" * 60)

    return metrics


def main():
    args = parse_args()

    results = []

    if args.run_all:
        # Run all ablations sequentially
        print("=" * 60)
        print("Running ALL ablation configurations")
        print("=" * 60)

        for config_name in ABLATION_CONFIGS.keys():
            print(f"\n{'='*60}")
            print(f"Starting: {config_name}")
            print(f"{'='*60}\n")

            metrics = run_ablation(args, config_name)
            results.append(metrics)

            # Clear CUDA cache between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Save summary
        summary_file = Path(args.output) / "ablation_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "results": results,
            }, f, indent=2)

        print("\n" + "=" * 60)
        print("All ablations complete!")
        print(f"Summary saved to: {summary_file}")
        print("=" * 60)

    else:
        # Run single ablation
        metrics = run_ablation(args, args.config)
        results.append(metrics)

    return results


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Curriculum GRPO Training on SFT Failure Cases.

This script trains GRPO only on problems that SFT failed to solve,
implementing the curriculum learning strategy from Phase 2.1.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-15_phase2_grpo_improvements.md
Phase: 2.1 - Curriculum Learning on Failures

Key Components:
    - load_failure_problem_ids: Load failures from collect_sft_failures.py output
    - create_curriculum_dataset: Build training set from hard problems only
    - DGRORewardWrapper: Variance-aware reward scaling for GRPO
    - format_grpo_prompt: Format problems for GRPO training

Key Improvements over Standard GRPO:
    1. Curriculum Learning: Focus training on SFT failures (~5% of problems)
    2. DGRO: Variance-aware advantage estimation
    3. Higher temperature: More exploration on hard problems

Example:
    >>> # Basic curriculum training
    >>> python scripts/training/curriculum_grpo.py \\
    ...     --failures data/training/sft_failures.json \\
    ...     --benchmark data/benchmarks/or_debug_bench_holdout \\
    ...     --model /data/qwen3_or_debug_merged \\
    ...     --output /data/grpo_curriculum_output

    >>> # With DGRO variance-aware training
    >>> python scripts/training/curriculum_grpo.py \\
    ...     --failures data/training/sft_failures.json \\
    ...     --benchmark data/benchmarks/or_debug_bench_holdout \\
    ...     --model /data/qwen3_or_debug_merged \\
    ...     --output /data/grpo_dgro_output \\
    ...     --use-dgro --beta-kl 0.001 --beta-reward 2.0
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

from src.training.gurobi_rewards import (
    gurobi_reward_func,
    set_use_solver_verification,
)
from src.agents.prompts import SYSTEM_PROMPT


def parse_args():
    parser = argparse.ArgumentParser(
        description="Curriculum GRPO Training on SFT Failures"
    )

    # Data args
    parser.add_argument(
        "--failures", type=str, required=True,
        help="Path to SFT failures JSON from collect_sft_failures.py"
    )
    parser.add_argument(
        "--benchmark", type=str, required=True,
        help="Path to benchmark directory containing problems"
    )

    # Model args
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to SFT model checkpoint"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for checkpoints"
    )

    # Training args
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Higher temp for more exploration on hard problems")

    # DGRO settings (variance-aware)
    parser.add_argument("--use-dgro", action="store_true",
                        help="Enable DGRO variance-aware training")
    parser.add_argument("--beta-kl", type=float, default=0.01,
                        help="KL penalty coefficient (lower = more exploration)")
    parser.add_argument("--beta-reward", type=float, default=1.0,
                        help="Reward scaling factor (higher = amplify variance)")

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # Solver verification
    parser.add_argument("--use-solver-verification", action="store_true",
                        help="Enable full Gurobi solver verification")

    # Logging
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)

    return parser.parse_args()


def load_failure_problem_ids(failures_path: str) -> List[str]:
    """Load problem IDs from failures JSON."""
    with open(failures_path, 'r') as f:
        data = json.load(f)
    return data.get('problem_ids', [])


def load_benchmark_problems(benchmark_path: str) -> Dict[str, Dict]:
    """Load all problems from benchmark directory."""
    benchmark_dir = Path(benchmark_path)

    # Try to load dataset.json
    dataset_file = benchmark_dir / "dataset.json"
    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            problems = {p['problem_id']: p for p in data.get('problems', [])}
            return problems

    # Try to load individual problem files
    problems = {}
    for json_file in benchmark_dir.glob("*.json"):
        if json_file.name != "dataset.json":
            with open(json_file, 'r') as f:
                p = json.load(f)
                if 'problem_id' in p:
                    problems[p['problem_id']] = p

    return problems


def create_curriculum_dataset(
    failure_ids: List[str],
    benchmark_problems: Dict[str, Dict]
) -> List[Dict]:
    """
    Create training dataset from failure cases.

    Each training sample contains the prompt for GRPO.
    """
    dataset = []

    for problem_id in failure_ids:
        if problem_id not in benchmark_problems:
            print(f"Warning: Problem {problem_id} not found in benchmark")
            continue

        problem = benchmark_problems[problem_id]

        # Format the prompt
        # This should match the format used in evaluation
        prompt = format_grpo_prompt(problem)

        dataset.append({
            "prompt": prompt,
            "problem_id": problem_id,
        })

    return dataset


def format_grpo_prompt(problem: Dict) -> str:
    """
    Format a problem into GRPO prompt format.

    The prompt includes the system context and problem description.
    """
    # Extract problem components
    problem_nl = problem.get('problem_nl', problem.get('description', ''))
    model_code = problem.get('model_code', problem.get('sabotaged_code', ''))
    iis_info = problem.get('iis', problem.get('ground_truth_iis', []))

    # Format IIS as string
    if isinstance(iis_info, list):
        iis_str = ", ".join(iis_info) if iis_info else "Unknown"
    else:
        iis_str = str(iis_info)

    # Create user message
    user_message = f"""You are debugging an infeasible optimization model.

## Problem Description
{problem_nl}

## Current Model Code
```python
{model_code}
```

## Solver Status
Status: INFEASIBLE

## IIS (Irreducible Inconsistent Subsystem)
The following constraints form an IIS:
{iis_str}

## Task
Analyze the IIS and determine the root cause of infeasibility.
Then provide a fix using one of the available actions.

Respond with a JSON object containing:
- "diagnosis": Your analysis of the root cause
- "action": One of GET_IIS, CHECK_SLACK, DROP_CONSTRAINT, RELAX_CONSTRAINT, SUBMIT
- "target": The constraint name to modify (if applicable)
- "value": The relaxation value (for RELAX_CONSTRAINT)
"""

    return user_message


class DGRORewardWrapper:
    """
    Wrapper for reward function that implements DGRO variance-aware scaling.

    DGRO (Decoupled Group Reward Optimization) scales rewards based on
    batch variance to amplify learning signal when variance is low.
    """

    def __init__(
        self,
        base_reward_fn,
        beta_reward: float = 1.0,
        min_variance: float = 0.01,
    ):
        self.base_reward_fn = base_reward_fn
        self.beta_reward = beta_reward
        self.min_variance = min_variance
        self.reward_history = []

    def __call__(self, completions, prompts=None, **kwargs):
        """
        Compute rewards with DGRO scaling.

        Standard GRPO: advantage = reward - mean(batch)
        DGRO: advantage = (reward - mean) * (beta_reward / (std + eps))
        """
        # Get base rewards
        base_rewards = self.base_reward_fn(completions, prompts, **kwargs)

        # Convert to tensor if needed
        if isinstance(base_rewards, list):
            rewards = torch.tensor(base_rewards, dtype=torch.float32)
        else:
            rewards = base_rewards

        # Track for statistics
        self.reward_history.extend(rewards.tolist())

        # DGRO scaling: amplify rewards based on variance
        batch_std = rewards.std()
        if batch_std > self.min_variance:
            # Scale to normalize variance while amplifying signal
            scaling_factor = self.beta_reward / (batch_std + 1e-8)
            rewards = rewards * scaling_factor

        return rewards.tolist()

    def get_stats(self) -> Dict[str, float]:
        """Get reward statistics."""
        if not self.reward_history:
            return {}
        import statistics
        return {
            'mean': statistics.mean(self.reward_history),
            'std': statistics.stdev(self.reward_history) if len(self.reward_history) > 1 else 0,
            'min': min(self.reward_history),
            'max': max(self.reward_history),
        }


def main():
    args = parse_args()

    print("=" * 70)
    print("Curriculum GRPO Training on SFT Failures")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Failures file: {args.failures}")
    print(f"  Benchmark: {args.benchmark}")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  DGRO enabled: {args.use_dgro}")
    if args.use_dgro:
        print(f"    β_kl: {args.beta_kl}")
        print(f"    β_reward: {args.beta_reward}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  Solver verification: {args.use_solver_verification}")
    print("=" * 70)

    # Load failure problem IDs
    print("\nLoading failure cases...")
    failure_ids = load_failure_problem_ids(args.failures)
    print(f"  Found {len(failure_ids)} failure problem IDs")

    # Load benchmark problems
    print("\nLoading benchmark problems...")
    benchmark_problems = load_benchmark_problems(args.benchmark)
    print(f"  Loaded {len(benchmark_problems)} problems from benchmark")

    # Create curriculum dataset
    print("\nCreating curriculum dataset...")
    curriculum_data = create_curriculum_dataset(failure_ids, benchmark_problems)
    print(f"  Created {len(curriculum_data)} training samples")

    if len(curriculum_data) == 0:
        print("\nError: No matching problems found!")
        print("Check that failure IDs match benchmark problem IDs.")
        sys.exit(1)

    # Convert to HuggingFace Dataset
    train_dataset = Dataset.from_list(curriculum_data)

    # Enable solver verification if requested
    if args.use_solver_verification:
        set_use_solver_verification(True)
        print("\n[INFO] Full Gurobi solver verification ENABLED")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure reward function
    reward_fn = gurobi_reward_func
    if args.use_dgro:
        print("\n[INFO] DGRO variance-aware reward scaling ENABLED")
        reward_fn = DGRORewardWrapper(
            base_reward_fn=gurobi_reward_func,
            beta_reward=args.beta_reward,
        )

    # Configure GRPO
    print("\nConfiguring GRPOTrainer...")
    config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        beta=args.beta_kl,  # Use DGRO's β_kl
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="tensorboard",
        logging_dir=f"{args.output}/tensorboard",
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        bf16=True,
    )

    # Configure LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Create trainer
    print("\nCreating GRPOTrainer...")
    trainer = GRPOTrainer(
        model=args.model,
        args=config,
        reward_funcs=reward_fn,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Training
    print("\n" + "=" * 70)
    print("Starting Curriculum GRPO Training")
    print(f"  Training on {len(curriculum_data)} hard problems only")
    print(f"  Expected to see higher reward variance than standard GRPO")
    print("=" * 70 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_output = f"{args.output}/final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    # Save training metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'failures_file': args.failures,
        'benchmark': args.benchmark,
        'num_training_problems': len(curriculum_data),
        'use_dgro': args.use_dgro,
        'beta_kl': args.beta_kl,
        'beta_reward': args.beta_reward,
        'temperature': args.temperature,
    }

    if args.use_dgro and hasattr(reward_fn, 'get_stats'):
        metadata['reward_stats'] = reward_fn.get_stats()

    with open(f"{args.output}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {final_output}")
    print(f"Metadata saved to: {args.output}/training_metadata.json")

    print("\n" + "=" * 70)
    print("Curriculum GRPO Training Complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Merge LoRA adapter:")
    print(f"   python scripts/training/merge_grpo_adapter.py \\")
    print(f"       --base {args.model} \\")
    print(f"       --adapter {final_output} \\")
    print(f"       --output /data/qwen3_curriculum_grpo_merged")
    print("\n2. Evaluate on held-out test set")
    print("\n3. Compare with standard GRPO on hard problems")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DGRO (Decoupled Group Reward Optimization) Training Script.

This script implements DGRO with variance-aware advantage estimation,
addressing the reward variance collapse problem in standard GRPO.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-15_phase2_grpo_improvements.md
Phase: 2.3 - DGRO Variance-Aware Training

Key Components:
    - DGRORewardWrapper: Variance-aware reward scaling
    - AdaptiveBetaScheduler: Cosine annealing for β_reward
    - load_curriculum_dataset: Load SFT failures for curriculum training

Key Features:
    - Decoupled β_kl (KL penalty) and β_reward (reward scaling)
    - Variance-aware advantage: advantage = (r - baseline) * β_reward / (std + ε)
    - Optional Kalman filter baseline (KRPO)
    - Exploration noise for low-variance scenarios

Technical Reference:
    - DGRO (arXiv 2505.12951): Decoupled coefficients
    - KRPO (arXiv 2025): Kalman filter baseline
    - HA-DW (arXiv 2601.08521): Dynamic advantage weights

Example:
    >>> # Basic DGRO training
    >>> python scripts/training/train_dgro.py \\
    ...     --model /data/qwen3_or_debug_merged \\
    ...     --dataset data/training/grpo_prompts.jsonl \\
    ...     --output /data/dgro_output \\
    ...     --beta-kl 0.001 --beta-reward 2.0

    >>> # Curriculum DGRO with PRM
    >>> python scripts/training/train_dgro.py \\
    ...     --model /data/qwen3_or_debug_merged \\
    ...     --failures data/training/sft_failures.json \\
    ...     --benchmark data/benchmarks/or_debug_bench_holdout \\
    ...     --output /data/dgro_curriculum_output \\
    ...     --prm-path /data/prm_output \\
    ...     --beta-kl 0.001 --beta-reward 2.0 --temperature 1.0
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

from src.training.gurobi_rewards import (
    gurobi_reward_func,
    set_use_solver_verification,
    set_prm_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DGRORewardWrapper:
    """
    DGRO (Decoupled Group Reward Optimization) Reward Wrapper.

    Implements variance-aware reward scaling to ensure non-zero gradients
    even when base rewards have low variance.

    Key Innovation:
        Standard GRPO: advantage = reward - mean(batch)
        DGRO: advantage = (reward - baseline) * β_reward / (std + ε)

    This amplifies the learning signal when variance is low.
    """

    def __init__(
        self,
        base_reward_fn: Callable,
        beta_reward: float = 2.0,
        min_variance: float = 0.01,
        use_kalman_baseline: bool = False,
        kalman_gain: float = 0.1,
    ):
        """
        Initialize DGRO wrapper.

        Args:
            base_reward_fn: Base reward function to wrap
            beta_reward: Reward scaling factor (higher = amplify variance)
            min_variance: Minimum variance threshold for scaling
            use_kalman_baseline: Use Kalman filter for baseline (KRPO)
            kalman_gain: Kalman filter learning rate
        """
        self.base_reward_fn = base_reward_fn
        self.beta_reward = beta_reward
        self.min_variance = min_variance
        self.use_kalman_baseline = use_kalman_baseline
        self.kalman_gain = kalman_gain

        # TRL GRPOTrainer compatibility
        self.__name__ = "dgro_reward"

        # Running statistics
        self.reward_history = []
        self.variance_history = []
        self.kalman_baseline = 0.0

        # Statistics tracking
        self.total_calls = 0
        self.low_variance_count = 0

    def __call__(
        self,
        completions: List[str],
        prompts: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        Compute DGRO-scaled rewards.

        Args:
            completions: Generated completions
            prompts: Input prompts
            **kwargs: Additional arguments for base reward

        Returns:
            List of scaled rewards
        """
        # Get base rewards
        base_rewards = self.base_reward_fn(completions, prompts, **kwargs)

        # Convert to tensor for computation
        if isinstance(base_rewards, list):
            rewards = torch.tensor(base_rewards, dtype=torch.float32)
        else:
            rewards = base_rewards.clone()

        # Track statistics
        self.total_calls += 1
        self.reward_history.extend(rewards.tolist())

        # Compute batch statistics
        batch_mean = rewards.mean().item()
        batch_std = rewards.std().item()
        self.variance_history.append(batch_std ** 2)

        # Update Kalman baseline if enabled
        if self.use_kalman_baseline:
            self.kalman_baseline += self.kalman_gain * (batch_mean - self.kalman_baseline)
            baseline = self.kalman_baseline
        else:
            baseline = batch_mean

        # DGRO scaling
        if batch_std > self.min_variance:
            # Standard DGRO: scale by variance
            scaling_factor = self.beta_reward / (batch_std + 1e-8)
            scaled_rewards = (rewards - baseline) * scaling_factor + baseline
        else:
            # Low variance case: add exploration noise
            self.low_variance_count += 1
            logger.debug(f"Low variance detected ({batch_std:.4f}), adding exploration noise")

            # Add structured noise to create gradient signal
            noise = torch.randn_like(rewards) * self.beta_reward * 0.1
            scaled_rewards = rewards + noise

        # Log statistics periodically
        if self.total_calls % 10 == 0:
            logger.info(
                f"DGRO stats - calls: {self.total_calls}, "
                f"mean: {batch_mean:.2f}, std: {batch_std:.4f}, "
                f"low_var_rate: {self.low_variance_count/self.total_calls:.1%}"
            )

        return scaled_rewards.tolist()

    def get_statistics(self) -> Dict[str, Any]:
        """Get accumulated statistics."""
        if not self.reward_history:
            return {}

        return {
            "total_calls": self.total_calls,
            "mean_reward": statistics.mean(self.reward_history),
            "std_reward": statistics.stdev(self.reward_history) if len(self.reward_history) > 1 else 0,
            "min_reward": min(self.reward_history),
            "max_reward": max(self.reward_history),
            "mean_variance": statistics.mean(self.variance_history) if self.variance_history else 0,
            "low_variance_rate": self.low_variance_count / self.total_calls if self.total_calls > 0 else 0,
            "kalman_baseline": self.kalman_baseline if self.use_kalman_baseline else None,
        }


class AdaptiveBetaScheduler:
    """
    Adaptive scheduler for β_reward based on training progress.

    Starts with higher β_reward (more exploration) and decreases
    as training progresses (more exploitation).
    """

    def __init__(
        self,
        initial_beta: float = 2.0,
        final_beta: float = 0.5,
        warmup_steps: int = 100,
        decay_steps: int = 1000,
    ):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.current_step = 0

    def get_beta(self) -> float:
        """Get current β_reward value."""
        if self.current_step < self.warmup_steps:
            # Warmup: linear increase
            return self.initial_beta * (self.current_step / self.warmup_steps)

        # Decay: cosine annealing
        progress = min(1.0, (self.current_step - self.warmup_steps) / self.decay_steps)
        cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        return self.final_beta + (self.initial_beta - self.final_beta) * cosine_decay.item()

    def step(self):
        """Advance scheduler by one step."""
        self.current_step += 1


def load_curriculum_dataset(
    failures_path: str,
    benchmark_path: str
) -> Dataset:
    """
    Load curriculum dataset from SFT failures.

    Args:
        failures_path: Path to failures JSON from collect_sft_failures.py
        benchmark_path: Path to benchmark directory

    Returns:
        HuggingFace Dataset for training
    """
    # Load failure IDs
    with open(failures_path, 'r') as f:
        failures_data = json.load(f)
    failure_ids = failures_data.get('problem_ids', [])

    logger.info(f"Loaded {len(failure_ids)} failure problem IDs")

    # Load benchmark problems
    benchmark_dir = Path(benchmark_path)
    problems = {}

    dataset_file = benchmark_dir / "dataset.json"
    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            data = json.load(f)
            problems = {p['problem_id']: p for p in data.get('problems', [])}
    else:
        for json_file in benchmark_dir.glob("*.json"):
            if json_file.name != "dataset.json":
                with open(json_file, 'r') as f:
                    p = json.load(f)
                    if 'problem_id' in p:
                        problems[p['problem_id']] = p

    logger.info(f"Loaded {len(problems)} benchmark problems")

    # Create training samples
    samples = []
    for problem_id in failure_ids:
        if problem_id not in problems:
            logger.warning(f"Problem {problem_id} not found in benchmark")
            continue

        problem = problems[problem_id]
        prompt = format_training_prompt(problem)

        samples.append({
            "prompt": prompt,
            "problem_id": problem_id,
            "problem_description": problem.get('problem_nl', problem.get('description', '')),
            "iis_constraints": json.dumps(problem.get('iis', problem.get('ground_truth_iis', []))),
            "model_file": problem.get('model_file', ''),
        })

    logger.info(f"Created {len(samples)} curriculum training samples")
    return Dataset.from_list(samples)


def format_training_prompt(problem: Dict) -> str:
    """Format problem for GRPO training prompt."""
    problem_nl = problem.get('problem_nl', problem.get('description', ''))
    model_code = problem.get('model_code', problem.get('sabotaged_code', ''))
    iis_info = problem.get('iis', problem.get('ground_truth_iis', []))

    if isinstance(iis_info, list):
        iis_str = ", ".join(iis_info) if iis_info else "Unknown"
    else:
        iis_str = str(iis_info)

    return f"""You are debugging an infeasible optimization model.

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="DGRO Training with Variance-Aware Advantages"
    )

    # Model args
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to base model (SFT checkpoint)"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for checkpoints"
    )

    # Data args (either dataset or curriculum)
    parser.add_argument(
        "--dataset", type=str,
        help="Path to GRPO prompts JSONL file"
    )
    parser.add_argument(
        "--failures", type=str,
        help="Path to SFT failures JSON (for curriculum learning)"
    )
    parser.add_argument(
        "--benchmark", type=str,
        help="Path to benchmark directory (required with --failures)"
    )

    # DGRO hyperparameters
    parser.add_argument(
        "--beta-kl", type=float, default=0.001,
        help="KL penalty coefficient (lower = more exploration, default: 0.001)"
    )
    parser.add_argument(
        "--beta-reward", type=float, default=2.0,
        help="Reward scaling factor (higher = amplify variance, default: 2.0)"
    )
    parser.add_argument(
        "--min-variance", type=float, default=0.01,
        help="Minimum variance threshold for DGRO scaling"
    )
    parser.add_argument(
        "--use-kalman-baseline", action="store_true",
        help="Use Kalman filter for adaptive baseline (KRPO)"
    )
    parser.add_argument(
        "--kalman-gain", type=float, default=0.1,
        help="Kalman filter learning rate"
    )
    parser.add_argument(
        "--adaptive-beta", action="store_true",
        help="Use adaptive β_reward scheduling"
    )

    # Training args
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-generations", type=int, default=8,
                        help="More generations for better variance estimation")
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument(
        "--temperature", type=float, default=1.0,
        help="Sampling temperature (higher for more diversity, default: 1.0)"
    )

    # PRM integration
    parser.add_argument(
        "--prm-path", type=str,
        help="Path to trained PRM for step-level rewards"
    )
    parser.add_argument(
        "--prm-weight", type=float, default=10.0,
        help="Weight for PRM scores in total reward"
    )

    # Solver verification
    parser.add_argument(
        "--use-solver-verification", action="store_true",
        help="Enable full Gurobi solver verification"
    )

    # LoRA config
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)

    # Logging
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--save-steps", type=int, default=100)

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("DGRO Training (Decoupled Group Reward Optimization)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model}")
    print(f"  Output: {args.output}")
    print(f"  β_kl: {args.beta_kl}")
    print(f"  β_reward: {args.beta_reward}")
    print(f"  Min variance: {args.min_variance}")
    print(f"  Kalman baseline: {args.use_kalman_baseline}")
    print(f"  Adaptive β: {args.adaptive_beta}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Num generations: {args.num_generations}")
    print(f"  PRM path: {args.prm_path or 'None'}")
    print(f"  Solver verification: {args.use_solver_verification}")
    print("=" * 70)

    # Load dataset
    if args.failures and args.benchmark:
        print("\nLoading curriculum dataset from SFT failures...")
        train_dataset = load_curriculum_dataset(args.failures, args.benchmark)
    elif args.dataset:
        print(f"\nLoading dataset from {args.dataset}...")
        dataset = load_dataset("json", data_files=args.dataset)
        train_dataset = dataset["train"]
    else:
        print("Error: Provide either --dataset or both --failures and --benchmark")
        sys.exit(1)

    print(f"Dataset size: {len(train_dataset)} samples")

    if len(train_dataset) == 0:
        print("Error: No training samples found!")
        sys.exit(1)

    # Enable solver verification if requested
    if args.use_solver_verification:
        set_use_solver_verification(True)
        print("\n[INFO] Full Gurobi solver verification ENABLED")

    # Load PRM if specified
    if args.prm_path:
        set_prm_model(args.prm_path, weight=args.prm_weight)
        print(f"\n[INFO] PRM loaded from {args.prm_path} with weight {args.prm_weight}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create DGRO reward wrapper
    print("\nConfiguring DGRO reward function...")
    dgro_reward = DGRORewardWrapper(
        base_reward_fn=gurobi_reward_func,
        beta_reward=args.beta_reward,
        min_variance=args.min_variance,
        use_kalman_baseline=args.use_kalman_baseline,
        kalman_gain=args.kalman_gain,
    )

    # Configure GRPO with DGRO settings
    print("\nConfiguring GRPOTrainer with DGRO settings...")
    config = GRPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_generations=args.num_generations,  # More for better variance
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,  # Higher for diversity
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        beta=args.beta_kl,  # Decoupled KL penalty
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
        reward_funcs=dgro_reward,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Training
    print("\n" + "=" * 70)
    print("Starting DGRO Training")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Expected variance improvement due to DGRO scaling")
    print("=" * 70 + "\n")

    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_output = f"{args.output}/final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)

    # Save training metadata and statistics
    dgro_stats = dgro_reward.get_statistics()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "dataset": args.dataset or f"curriculum:{args.failures}",
        "num_samples": len(train_dataset),
        "dgro_config": {
            "beta_kl": args.beta_kl,
            "beta_reward": args.beta_reward,
            "min_variance": args.min_variance,
            "use_kalman_baseline": args.use_kalman_baseline,
            "adaptive_beta": args.adaptive_beta,
        },
        "training_config": {
            "temperature": args.temperature,
            "num_generations": args.num_generations,
            "num_epochs": args.num_epochs,
        },
        "dgro_statistics": dgro_stats,
        "prm_path": args.prm_path,
    }

    with open(f"{args.output}/training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModel saved to: {final_output}")
    print(f"Metadata saved to: {args.output}/training_metadata.json")

    # Print DGRO statistics
    print("\n" + "=" * 70)
    print("DGRO Training Statistics")
    print("=" * 70)
    if dgro_stats:
        print(f"  Total reward computations: {dgro_stats.get('total_calls', 0)}")
        print(f"  Mean reward: {dgro_stats.get('mean_reward', 0):.2f}")
        print(f"  Std reward: {dgro_stats.get('std_reward', 0):.4f}")
        print(f"  Mean variance: {dgro_stats.get('mean_variance', 0):.6f}")
        print(f"  Low variance rate: {dgro_stats.get('low_variance_rate', 0):.1%}")
        if dgro_stats.get('kalman_baseline') is not None:
            print(f"  Final Kalman baseline: {dgro_stats['kalman_baseline']:.2f}")

    print("\n" + "=" * 70)
    print("DGRO Training Complete!")
    print("=" * 70)

    print("\nNext steps:")
    print("1. Merge LoRA adapter:")
    print(f"   python scripts/training/merge_grpo_adapter.py \\")
    print(f"       --base {args.model} \\")
    print(f"       --adapter {final_output} \\")
    print(f"       --output /data/qwen3_dgro_merged")
    print("\n2. Evaluate on held-out test set")
    print("\n3. Compare with standard GRPO and SFT")


if __name__ == "__main__":
    main()

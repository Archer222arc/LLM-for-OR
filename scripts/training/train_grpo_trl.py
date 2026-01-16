#!/home/Archer/miniforge3/bin/python
"""
GRPO training script using HuggingFace TRL with Gurobi rewards.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-15_phase2_grpo_improvements.md

Key Components:
    - GRPOTrainer: TRL trainer for Group Relative Policy Optimization
    - gurobi_reward_func: Gurobi solver-based reward computation
    - LoraConfig: Parameter-efficient fine-tuning configuration

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python scripts/training/train_grpo_trl.py

    # With custom config
    CUDA_VISIBLE_DEVICES=0 python scripts/training/train_grpo_trl.py \
        --model_path /data/qwen3_or_debug_merged \
        --output_dir /data/grpo_output \
        --num_epochs 3
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig

from src.training.gurobi_rewards import (
    gurobi_reward_func,
    set_use_solver_verification,
    get_use_solver_verification,
)


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training with TRL and Gurobi rewards")

    # Model args
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/qwen3_or_debug_merged",
        help="Path to SFT model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/grpo_output",
        help="Output directory for checkpoints"
    )

    # Data args
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/training/grpo_prompts.jsonl",
        help="Path to GRPO prompts dataset"
    )

    # Training args
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--num_generations", type=int, default=4, help="Completions per prompt")
    parser.add_argument("--max_completion_length", type=int, default=512, help="Max completion tokens")

    # GRPO specific
    parser.add_argument("--beta", type=float, default=0.1, help="KL penalty coefficient (default: 0.1)")
    parser.add_argument("--scale_rewards", type=str, default="batch", help="Reward scaling method")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for generation diversity (default: 0.7)"
    )

    # Solver verification
    parser.add_argument(
        "--use_solver_verification",
        action="store_true",
        help="Enable full Gurobi solver verification for accurate outcome reward"
    )

    # LoRA config (for independent training from base model, use larger rank)
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank (use 64 for base model)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (use 128 for base model)")

    # Logging
    parser.add_argument("--logging_steps", type=int, default=10, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=500, help="Checkpoint save frequency")

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("GRPO Training with TRL and Gurobi Rewards")
    print("=" * 60)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation}")
    print(f"Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"Num generations: {args.num_generations}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Temperature: {args.temperature}")
    print(f"Beta (KL): {args.beta}")
    print(f"LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    print(f"Solver verification: {args.use_solver_verification}")
    print("=" * 60)

    # Enable solver verification if requested
    if args.use_solver_verification:
        set_use_solver_verification(True)
        print("\n[INFO] Full Gurobi solver verification ENABLED")
        print("  - Outcome reward: +100 (OPTIMAL), -50 (INFEASIBLE)")
        print("  - This provides accurate success/failure signals")

    # Check CUDA
    print(f"\nCUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    print("\nLoading dataset...")
    dataset_path = project_root / args.dataset_path
    dataset = load_dataset("json", data_files=str(dataset_path))
    train_dataset = dataset["train"]
    print(f"Dataset size: {len(train_dataset)} samples")

    # Configure GRPO training
    print("\nConfiguring GRPOTrainer...")
    config = GRPOConfig(
        # Output
        output_dir=args.output_dir,

        # Batch settings
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,

        # GRPO settings
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,  # Sampling temperature for diversity

        # Learning
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        beta=args.beta,

        # Logging
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="tensorboard",
        logging_dir=f"{args.output_dir}/tensorboard",

        # Optimization
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
        warmup_ratio=0.1,

        # Memory optimization
        gradient_checkpointing=True,
        bf16=True,
    )

    # Configure LoRA for memory efficiency
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
        model=args.model_path,
        args=config,
        reward_funcs=gurobi_reward_func,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train()

    # Save final model
    print("\nSaving final model...")
    final_output = f"{args.output_dir}/final"
    trainer.save_model(final_output)
    tokenizer.save_pretrained(final_output)
    print(f"Model saved to: {final_output}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

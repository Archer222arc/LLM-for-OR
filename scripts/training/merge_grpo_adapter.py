#!/home/Archer/miniforge3/bin/python
"""
Merge GRPO LoRA adapter with base SFT model.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/05_TRAINING.md

Usage:
    python scripts/training/merge_grpo_adapter.py \
        --base_model /data/qwen3_or_debug_merged \
        --adapter /data/grpo_output/final \
        --output /data/qwen3_or_debug_grpo_merged
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge GRPO LoRA adapter with base model")
    parser.add_argument(
        "--base_model",
        type=str,
        default="/data/qwen3_or_debug_merged",
        help="Path to base SFT model"
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default="/data/grpo_output/final",
        help="Path to GRPO LoRA adapter"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data/qwen3_or_debug_grpo_merged",
        help="Output path for merged model"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Merge GRPO LoRA Adapter")
    print("=" * 60)
    print(f"Base model: {args.base_model}")
    print(f"Adapter: {args.adapter}")
    print(f"Output: {args.output}")
    print("=" * 60)

    # Check paths exist
    if not Path(args.base_model).exists():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")
    if not Path(args.adapter).exists():
        raise FileNotFoundError(f"Adapter not found: {args.adapter}")

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load base model
    print("\nLoading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    print(f"Base model loaded: {base_model.config.model_type}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    print(f"Tokenizer vocabulary size: {tokenizer.vocab_size}")

    # Load LoRA adapter
    print("\nLoading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)
    print("Adapter loaded successfully")

    # Merge adapter into base model
    print("\nMerging adapter into base model...")
    merged_model = model.merge_and_unload()
    print("Merge complete")

    # Save merged model
    print(f"\nSaving merged model to {args.output}...")
    merged_model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    print("Save complete")

    # Verify saved model
    print("\nVerifying saved model...")
    saved_files = list(output_path.glob("*.safetensors"))
    total_size = sum(f.stat().st_size for f in saved_files)
    print(f"Saved {len(saved_files)} safetensor files, total size: {total_size / 1e9:.2f} GB")

    print("\n" + "=" * 60)
    print("Merge complete!")
    print(f"Merged model saved to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()

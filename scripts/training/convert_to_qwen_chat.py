#!/usr/bin/env python3
"""
Convert Alpaca-style SFT data to Qwen Chat format for LLaMA-Factory.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/research/sft_research.md

Key Points:
    - Converts instruction/input/output to messages format
    - Preserves <think> tags in assistant responses
    - Uses system prompt for OR debugging context
    - Outputs JSONL format for LLaMA-Factory

Input Format (Alpaca):
    {
        "instruction": "Debug the infeasible optimization model...",
        "input": "## Problem\\nID: ...\\n## IIS\\n...",
        "output": "<think>\\n...\\n</think>\\n\\nAction: ...",
        "metadata": {...}
    }

Output Format (Qwen Chat / ShareGPT):
    {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction + input},
            {"role": "assistant", "content": output}
        ]
    }

Usage:
    python scripts/training/convert_to_qwen_chat.py \\
        --input data/training/sft_gpt52chat.json \\
        --output data/training/sft_qwen_chat.jsonl

    # Include heuristic data
    python scripts/training/convert_to_qwen_chat.py \\
        --input data/training/sft_gpt52chat.json data/training/sft_heuristic.json \\
        --output data/training/sft_combined.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


# System prompt for OR debugging (from research)
SYSTEM_PROMPT = """You are an Operations Research debugging assistant.
Analyze problems step-by-step in <think> tags, then output the action.

Format:
<think>
Your detailed reasoning about the infeasibility...
</think>

Action: ACTION_TYPE(arguments)

Available actions:
- GET_IIS(): Compute Irreducible Infeasible Subsystem
- DROP_CONSTRAINT(name): Remove a constraint
- RELAX_CONSTRAINT(name, amount): Relax a constraint's RHS
- SUBMIT(): Submit the repaired model"""


def convert_alpaca_to_qwen_chat(
    alpaca_item: Dict[str, Any],
    system_prompt: str = SYSTEM_PROMPT,
) -> Dict[str, Any]:
    """Convert a single Alpaca-style item to Qwen Chat format.

    Args:
        alpaca_item: Dict with instruction, input, output keys
        system_prompt: System prompt for the assistant

    Returns:
        Dict with messages list in ShareGPT format
    """
    # Combine instruction and input for user message
    user_content = alpaca_item.get("instruction", "")
    if alpaca_item.get("input"):
        user_content += f"\n\n{alpaca_item['input']}"

    # Get assistant response (preserve <think> tags as-is)
    assistant_content = alpaca_item.get("output", "")

    # Build messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]

    return {"messages": messages}


def convert_file(
    input_path: Path,
    system_prompt: str = SYSTEM_PROMPT,
) -> List[Dict[str, Any]]:
    """Convert an entire Alpaca-style JSON file to Qwen Chat format.

    Args:
        input_path: Path to input JSON file
        system_prompt: System prompt for the assistant

    Returns:
        List of converted items
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Handle both formats: {"data": [...]} or [...]
    if isinstance(data, dict) and 'data' in data:
        items = data['data']
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError(f"Unexpected data format in {input_path}")

    converted = []
    for item in items:
        converted.append(convert_alpaca_to_qwen_chat(item, system_prompt))

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert Alpaca-style SFT data to Qwen Chat format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        nargs='+',
        required=True,
        help='Input JSON file(s) in Alpaca format'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output JSONL file in Qwen Chat format'
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default=None,
        help='Custom system prompt (uses default OR debugging prompt if not specified)'
    )
    parser.add_argument(
        '--split-ratio',
        type=float,
        default=None,
        help='If specified, create train/val split with this ratio (e.g., 0.9 for 90%% train)'
    )

    args = parser.parse_args()

    system_prompt = args.system_prompt or SYSTEM_PROMPT
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert all input files
    all_converted = []
    for input_file in args.input:
        input_path = Path(input_file)
        if not input_path.exists():
            print(f"Warning: {input_path} not found, skipping")
            continue

        converted = convert_file(input_path, system_prompt)
        print(f"Converted {len(converted)} samples from {input_path.name}")
        all_converted.extend(converted)

    print(f"\nTotal samples: {len(all_converted)}")

    # Handle train/val split
    if args.split_ratio:
        import random
        random.seed(42)
        random.shuffle(all_converted)

        split_idx = int(len(all_converted) * args.split_ratio)
        train_data = all_converted[:split_idx]
        val_data = all_converted[split_idx:]

        # Save train
        train_path = output_path.with_suffix('.train.jsonl')
        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Train set: {len(train_data)} samples -> {train_path}")

        # Save val
        val_path = output_path.with_suffix('.val.jsonl')
        with open(val_path, 'w', encoding='utf-8') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Val set: {len(val_data)} samples -> {val_path}")
    else:
        # Save all to single file
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in all_converted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"Saved to: {output_path}")

    # Print sample for verification
    if all_converted:
        print("\n" + "="*60)
        print("Sample output (first item):")
        print("="*60)
        sample = all_converted[0]
        print(json.dumps(sample, indent=2, ensure_ascii=False)[:1500] + "...")


if __name__ == '__main__':
    main()

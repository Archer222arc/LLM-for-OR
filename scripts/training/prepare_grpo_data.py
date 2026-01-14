#!/usr/bin/env python3
"""
Prepare GRPO training dataset from SFT data.

Converts SFT chat format to GRPO prompt format with metadata.

Usage:
    python scripts/training/prepare_grpo_data.py \
        --input data/training/sft_qwen_chat.train.jsonl \
        --output data/training/grpo_prompts.jsonl
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional


def extract_problem_metadata(user_content: str) -> Dict[str, Any]:
    """
    Extract problem metadata from user message content.

    Args:
        user_content: The user message content containing problem details

    Returns:
        Dict with problem_id, problem_type, iis_constraints, constraint_names
    """
    metadata = {
        "problem_id": None,
        "problem_type": None,
        "iis_constraints": [],
        "iis_bounds": [],
        "constraint_names": [],
        "model_file": None,
    }

    # Extract Problem ID
    id_match = re.search(r"ID:\s*(\S+)", user_content)
    if id_match:
        metadata["problem_id"] = id_match.group(1)

    # Extract Type
    type_match = re.search(r"Type:\s*(\w+)", user_content)
    if type_match:
        metadata["problem_type"] = type_match.group(1)

    # Extract IIS Constraints
    iis_match = re.search(r"Conflicting Constraints:\s*\[([^\]]*)\]", user_content)
    if iis_match:
        constraints_str = iis_match.group(1)
        constraints = re.findall(r"'([^']+)'", constraints_str)
        metadata["iis_constraints"] = constraints

    # Extract IIS Bounds
    bounds_match = re.search(r"Conflicting Bounds:\s*\[([^\]]*)\]", user_content)
    if bounds_match:
        bounds_str = bounds_match.group(1)
        bounds = re.findall(r"'([^']+)'", bounds_str)
        metadata["iis_bounds"] = bounds

    # Extract Constraint Names
    names_match = re.search(r"Constraint Names:\s*\[([^\]]*)\]", user_content)
    if names_match:
        names_str = names_match.group(1)
        names = re.findall(r"'([^']+)'", names_str)
        metadata["constraint_names"] = names

    # Construct model file path (if problem_id is available)
    if metadata["problem_id"]:
        metadata["model_file"] = f"data/benchmarks/models/{metadata['problem_id']}.mps"

    return metadata


def convert_sft_to_grpo(sft_record: Dict) -> Optional[Dict]:
    """
    Convert a single SFT record to GRPO format.

    Args:
        sft_record: SFT record with 'messages' field

    Returns:
        GRPO record with 'prompt' (as chat messages) and metadata fields
    """
    messages = sft_record.get("messages", [])

    if len(messages) < 2:
        return None

    # Find system and user messages
    system_content = ""
    user_content = ""

    for msg in messages:
        if msg["role"] == "system":
            system_content = msg["content"]
        elif msg["role"] == "user":
            user_content = msg["content"]

    if not user_content:
        return None

    # Build prompt as list of messages (chat format for TRL GRPOTrainer)
    # TRL will apply the chat template automatically when prompt is a list
    prompt_messages = []
    if system_content:
        prompt_messages.append({"role": "system", "content": system_content})
    prompt_messages.append({"role": "user", "content": user_content})

    # Extract metadata
    metadata = extract_problem_metadata(user_content)

    # Create GRPO record with prompt as list of messages
    grpo_record = {
        "prompt": prompt_messages,
        "problem_id": metadata["problem_id"],
        "problem_type": metadata["problem_type"],
        "iis_constraints": json.dumps(metadata["iis_constraints"]),
        "iis_bounds": json.dumps(metadata["iis_bounds"]),
        "constraint_names": json.dumps(metadata["constraint_names"]),
        "model_file": metadata["model_file"],
    }

    return grpo_record


def main():
    parser = argparse.ArgumentParser(description="Prepare GRPO dataset from SFT data")
    parser.add_argument(
        "--input",
        type=str,
        default="data/training/sft_qwen_chat.train.jsonl",
        help="Input SFT data file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/training/grpo_prompts.jsonl",
        help="Output GRPO prompts file"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process records
    converted = 0
    skipped = 0
    seen_problems = set()  # Deduplicate by problem_id

    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            try:
                sft_record = json.loads(line)
                grpo_record = convert_sft_to_grpo(sft_record)

                if grpo_record and grpo_record["problem_id"]:
                    # Deduplicate
                    if grpo_record["problem_id"] not in seen_problems:
                        seen_problems.add(grpo_record["problem_id"])
                        f_out.write(json.dumps(grpo_record) + "\n")
                        converted += 1
                    else:
                        skipped += 1
                else:
                    skipped += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing JSON: {e}")
                skipped += 1

    print(f"Conversion complete!")
    print(f"  Converted: {converted} unique problems")
    print(f"  Skipped: {skipped} (duplicates or invalid)")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()

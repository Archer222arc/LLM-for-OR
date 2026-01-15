#!/usr/bin/env python3
"""
Process Reward Model (PRM) for OR Debugging.

This module implements a ThinkPRM-style process reward model that evaluates
the quality of each reasoning step in OR debugging trajectories.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-15_phase2_grpo_improvements.md
Phase: 2.2 - Process Reward Model Integration

Key Components:
    - StepLabel: Dataclass for labeled training examples
    - StepLabelGenerator: Auto-generate labels from solver feedback
    - ProcessRewardModel: Train and score reasoning steps
    - compute_step_label: Core labeling logic using solver state

Label Generation Rules (Auto-labeled, No Manual Annotation):
    - |IIS_t+1| < |IIS_t| → label = 1.0 (IIS reduced)
    - status == OPTIMAL → label = 1.0 (problem solved)
    - diagnosis ∩ actual_IIS ≠ ∅ → label = 0.5 (correct diagnosis)
    - diagnostic action → label = 0.2 (information gathering)
    - otherwise → label = 0.0 (no progress)

Example:
    >>> # Generate labels from evaluation database
    >>> python -m src.training.process_reward_model generate \\
    ...     --db outputs/experiments/2026-01-15/sft_eval/results.db \\
    ...     --output data/training/prm_labels.json

    >>> # Train PRM classifier
    >>> python -m src.training.process_reward_model train \\
    ...     --labels data/training/prm_labels.json \\
    ...     --model Qwen/Qwen3-1.5B \\
    ...     --output /data/prm_output

    >>> # Score steps during inference
    >>> from src.training.process_reward_model import ProcessRewardModel
    >>> prm = ProcessRewardModel.load("/data/prm_output")
    >>> score = prm.score_step(state_text, action_text)
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


@dataclass
class StepLabel:
    """Labeled step for PRM training."""
    step_id: str                     # Unique identifier
    problem_id: str                  # Problem this step belongs to
    step_index: int                  # Position in trajectory (0-indexed)

    # State information
    state_text: str                  # Formatted state representation
    iis_before: List[str]           # IIS constraints before action
    status_before: str              # Solver status before action

    # Action information
    action_text: str                # Model's action output
    action_type: str                # Parsed action type
    action_target: Optional[str]    # Target constraint/variable
    diagnosis: List[str]            # Constraints diagnosed by model

    # Post-action state (for label computation)
    iis_after: List[str]            # IIS constraints after action
    status_after: str               # Solver status after action

    # Label (computed automatically)
    label: float = 0.0              # 0.0, 0.5, or 1.0
    label_reason: str = ""          # Why this label was assigned


@dataclass
class PRMConfig:
    """Configuration for PRM training."""
    model_name: str = "Qwen/Qwen3-1.5B"
    max_length: int = 2048
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05


def compute_step_label(
    iis_before: List[str],
    iis_after: List[str],
    status_before: str,
    status_after: str,
    diagnosis: List[str],
    ground_truth_iis: List[str]
) -> Tuple[float, str]:
    """
    Compute step-level label based on solver feedback.

    This is the core auto-labeling logic that eliminates manual annotation.

    Args:
        iis_before: IIS constraints before this step
        iis_after: IIS constraints after this step
        status_before: Solver status before (INFEASIBLE, OPTIMAL, etc.)
        status_after: Solver status after
        diagnosis: Constraints the model identified as problematic
        ground_truth_iis: Actual IIS from the problem

    Returns:
        Tuple of (label, reason)
    """
    # Rule 1: Problem solved → highest reward
    if status_after == "OPTIMAL":
        return 1.0, "solved"

    # Rule 2: IIS reduction → excellent progress
    if len(iis_after) < len(iis_before):
        reduction = len(iis_before) - len(iis_after)
        return 1.0, f"iis_reduced_by_{reduction}"

    # Rule 3: Correct diagnosis → partial credit
    if diagnosis and ground_truth_iis:
        diagnosis_set = set(d.lower() for d in diagnosis)
        iis_set = set(c.lower() for c in ground_truth_iis)
        overlap = diagnosis_set & iis_set
        if overlap:
            overlap_ratio = len(overlap) / len(iis_set)
            if overlap_ratio >= 0.5:
                return 0.5, f"correct_diagnosis_{len(overlap)}/{len(iis_set)}"
            else:
                return 0.3, f"partial_diagnosis_{len(overlap)}/{len(iis_set)}"

    # Rule 4: Diagnostic action without result (neutral)
    # Give small positive for gathering information
    # This will be determined by action_type in the caller

    # Default: no progress
    return 0.0, "no_progress"


def format_step_input(
    problem_description: str,
    iis_constraints: List[str],
    solver_status: str,
    action_history: List[str]
) -> str:
    """
    Format step state for PRM input.

    Creates a standardized text representation that the PRM can evaluate.
    """
    history_text = "\n".join(f"- Step {i+1}: {a}" for i, a in enumerate(action_history[-5:]))
    iis_text = ", ".join(iis_constraints) if iis_constraints else "None"

    return f"""## Problem
{problem_description[:500]}...

## Current State
Solver Status: {solver_status}
IIS Constraints: {iis_text}

## Recent History
{history_text if history_text else "No previous actions"}

## Next Action to Evaluate:
"""


class StepLabelGenerator:
    """
    Generate step-level labels from evaluation trajectories.

    Uses solver feedback to automatically label each step without
    manual annotation.
    """

    def __init__(self, benchmark_path: Optional[str] = None):
        """
        Initialize the label generator.

        Args:
            benchmark_path: Path to benchmark for ground truth lookup
        """
        self.benchmark_problems = {}
        if benchmark_path:
            self._load_benchmark(benchmark_path)

    def _load_benchmark(self, benchmark_path: str):
        """Load benchmark problems for ground truth IIS lookup."""
        benchmark_dir = Path(benchmark_path)

        # Try dataset.json first
        dataset_file = benchmark_dir / "dataset.json"
        if dataset_file.exists():
            with open(dataset_file, 'r') as f:
                data = json.load(f)
                for p in data.get('problems', []):
                    self.benchmark_problems[p['problem_id']] = p
        else:
            # Load individual files
            for json_file in benchmark_dir.glob("*.json"):
                if json_file.name != "dataset.json":
                    with open(json_file, 'r') as f:
                        p = json.load(f)
                        if 'problem_id' in p:
                            self.benchmark_problems[p['problem_id']] = p

    def generate_labels_from_trajectory(
        self,
        trajectory: Dict[str, Any]
    ) -> List[StepLabel]:
        """
        Generate step labels from a single trajectory.

        Args:
            trajectory: Dictionary containing:
                - problem_id: Problem identifier
                - steps: List of step dictionaries
                - Each step: {state, action, iis_before, iis_after, status}

        Returns:
            List of StepLabel objects
        """
        labels = []
        problem_id = trajectory.get('problem_id', 'unknown')

        # Get ground truth IIS from benchmark
        ground_truth_iis = []
        if problem_id in self.benchmark_problems:
            problem = self.benchmark_problems[problem_id]
            ground_truth_iis = problem.get('iis', problem.get('ground_truth_iis', []))

        steps = trajectory.get('steps', [])
        for i, step in enumerate(steps):
            # Extract step information
            iis_before = step.get('iis_before', step.get('iis', []))
            iis_after = step.get('iis_after', [])
            status_before = step.get('status_before', 'INFEASIBLE')
            status_after = step.get('status_after', step.get('final_status', 'INFEASIBLE'))

            # Parse action
            action_text = step.get('action', step.get('response', ''))
            action_type = step.get('action_type', 'UNKNOWN')
            action_target = step.get('target', step.get('action_target', None))
            diagnosis = step.get('diagnosis', step.get('diagnosed_constraints', []))

            # Compute label
            label, reason = compute_step_label(
                iis_before=iis_before,
                iis_after=iis_after,
                status_before=status_before,
                status_after=status_after,
                diagnosis=diagnosis,
                ground_truth_iis=ground_truth_iis
            )

            # Adjust label for diagnostic actions
            if action_type in ['GET_IIS', 'CHECK_SLACK'] and label == 0.0:
                label = 0.2  # Small positive for information gathering
                reason = "diagnostic_action"

            # Create state text
            state_text = format_step_input(
                problem_description=trajectory.get('problem_nl', trajectory.get('description', '')),
                iis_constraints=iis_before,
                solver_status=status_before,
                action_history=[s.get('action', '')[:100] for s in steps[:i]]
            )

            step_label = StepLabel(
                step_id=f"{problem_id}_step{i}",
                problem_id=problem_id,
                step_index=i,
                state_text=state_text,
                iis_before=iis_before,
                status_before=status_before,
                action_text=action_text,
                action_type=action_type,
                action_target=action_target,
                diagnosis=diagnosis,
                iis_after=iis_after,
                status_after=status_after,
                label=label,
                label_reason=reason
            )

            labels.append(step_label)

        return labels

    def generate_labels_from_db(
        self,
        db_path: str,
        model_name: str = "sft"
    ) -> List[StepLabel]:
        """
        Generate labels from SQLite evaluation database.

        This reads trajectories from the evaluation results database
        and generates step-level labels.
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query trajectories
        cursor.execute("""
            SELECT problem_id, trajectory_json, ground_truth_iis
            FROM evaluation_results
            WHERE model_name = ? AND trajectory_json IS NOT NULL
        """, (model_name,))

        all_labels = []
        for row in cursor.fetchall():
            problem_id, traj_json, iis_json = row

            try:
                trajectory = json.loads(traj_json) if traj_json else {}
                ground_truth_iis = json.loads(iis_json) if iis_json else []

                # Add metadata
                trajectory['problem_id'] = problem_id
                if ground_truth_iis:
                    # Update benchmark lookup
                    self.benchmark_problems[problem_id] = {
                        'problem_id': problem_id,
                        'ground_truth_iis': ground_truth_iis
                    }

                labels = self.generate_labels_from_trajectory(trajectory)
                all_labels.extend(labels)

            except Exception as e:
                logger.warning(f"Error processing {problem_id}: {e}")
                continue

        conn.close()
        return all_labels


class ProcessRewardModel:
    """
    Process Reward Model for step-level evaluation.

    Uses a fine-tuned classifier to score reasoning steps.
    """

    def __init__(self, model_path: Optional[str] = None, config: Optional[PRMConfig] = None):
        """
        Initialize PRM.

        Args:
            model_path: Path to trained PRM checkpoint
            config: Training configuration
        """
        self.model_path = model_path
        self.config = config or PRMConfig()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    @classmethod
    def load(cls, model_path: str) -> "ProcessRewardModel":
        """Load a trained PRM from checkpoint."""
        prm = cls(model_path=model_path)
        prm._load_model()
        return prm

    def _load_model(self):
        """Load model and tokenizer."""
        if self._loaded:
            return

        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from peft import PeftModel

            # Try to load as PEFT model first
            config_path = Path(self.model_path) / "adapter_config.json"
            if config_path.exists():
                # Load base model and adapter
                base_model_name = self.config.model_name
                self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_name,
                    num_labels=1,
                    torch_dtype="auto"
                )
                self.model = PeftModel.from_pretrained(base_model, self.model_path)
            else:
                # Load full model
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path,
                    num_labels=1,
                    torch_dtype="auto"
                )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            self.model.eval()
            self._loaded = True
            logger.info(f"Loaded PRM from {self.model_path}")

        except Exception as e:
            logger.error(f"Failed to load PRM: {e}")
            raise

    def score_step(
        self,
        state_text: str,
        action_text: str,
        **kwargs
    ) -> float:
        """
        Score a single reasoning step.

        Args:
            state_text: Current state representation
            action_text: Proposed action

        Returns:
            Score in [0, 1] indicating step quality
        """
        if not self._loaded:
            self._load_model()

        import torch

        # Format input
        input_text = f"{state_text}\n{action_text}"

        # Tokenize
        inputs = self.tokenizer(
            input_text,
            max_length=self.config.max_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Apply sigmoid to get probability
            score = torch.sigmoid(outputs.logits).item()

        return score

    def score_trajectory(
        self,
        trajectory: List[Dict[str, str]]
    ) -> List[float]:
        """
        Score all steps in a trajectory.

        Args:
            trajectory: List of {state_text, action_text} dicts

        Returns:
            List of scores for each step
        """
        scores = []
        for step in trajectory:
            score = self.score_step(
                state_text=step.get('state_text', ''),
                action_text=step.get('action_text', '')
            )
            scores.append(score)
        return scores

    def train(
        self,
        labels: List[StepLabel],
        output_dir: str
    ):
        """
        Train PRM on labeled steps.

        Args:
            labels: List of StepLabel objects
            output_dir: Where to save the trained model
        """
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            TrainingArguments,
            Trainer
        )
        from peft import LoraConfig, get_peft_model, TaskType
        from datasets import Dataset

        logger.info(f"Training PRM on {len(labels)} labeled steps")

        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            num_labels=1,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )

        # Configure LoRA
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )

        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

        # Prepare dataset
        def format_example(label: StepLabel) -> Dict[str, Any]:
            text = f"{label.state_text}\n{label.action_text}"
            return {
                "text": text,
                "label": label.label
            }

        dataset_dict = [format_example(l) for l in labels]
        dataset = Dataset.from_list(dataset_dict)

        # Tokenize
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                max_length=self.config.max_length,
                truncation=True,
                padding="max_length"
            )

        dataset = dataset.map(tokenize_fn, batched=True)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            bf16=torch.cuda.is_available(),
            report_to="tensorboard",
            logging_dir=f"{output_dir}/tensorboard"
        )

        # Custom compute_metrics for regression
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = predictions.squeeze()
            mse = ((predictions - labels) ** 2).mean()
            return {"mse": mse}

        # Train
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Save
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        self.model_path = output_dir
        self._loaded = True

        logger.info(f"PRM saved to {output_dir}")


def analyze_label_distribution(labels: List[StepLabel]) -> Dict[str, Any]:
    """Analyze the distribution of generated labels."""
    from collections import Counter

    label_counts = Counter(l.label for l in labels)
    reason_counts = Counter(l.label_reason for l in labels)
    action_type_counts = Counter(l.action_type for l in labels)

    return {
        "total_steps": len(labels),
        "label_distribution": dict(label_counts),
        "reason_distribution": dict(reason_counts),
        "action_type_distribution": dict(action_type_counts),
        "avg_label": sum(l.label for l in labels) / len(labels) if labels else 0
    }


def main():
    """CLI entry point for PRM operations."""
    parser = argparse.ArgumentParser(
        description="Process Reward Model for OR Debugging"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate labels command
    gen_parser = subparsers.add_parser("generate", help="Generate step labels")
    gen_parser.add_argument(
        "--trajectories", type=str,
        help="Path to trajectories JSON file"
    )
    gen_parser.add_argument(
        "--db", type=str,
        help="Path to SQLite database with trajectories"
    )
    gen_parser.add_argument(
        "--model-name", type=str, default="sft",
        help="Model name in database (default: sft)"
    )
    gen_parser.add_argument(
        "--benchmark", type=str,
        help="Path to benchmark for ground truth lookup"
    )
    gen_parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for generated labels"
    )

    # Train command
    train_parser = subparsers.add_parser("train", help="Train PRM")
    train_parser.add_argument(
        "--labels", type=str, required=True,
        help="Path to generated labels JSON"
    )
    train_parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-1.5B",
        help="Base model for PRM"
    )
    train_parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for trained PRM"
    )
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--lr", type=float, default=2e-5)

    # Score command
    score_parser = subparsers.add_parser("score", help="Score steps with trained PRM")
    score_parser.add_argument(
        "--prm", type=str, required=True,
        help="Path to trained PRM"
    )
    score_parser.add_argument(
        "--input", type=str, required=True,
        help="Input file with steps to score"
    )
    score_parser.add_argument(
        "--output", type=str, required=True,
        help="Output file for scored steps"
    )

    args = parser.parse_args()

    if args.command == "generate":
        print("=" * 60)
        print("Generating Step Labels for PRM Training")
        print("=" * 60)

        generator = StepLabelGenerator(benchmark_path=args.benchmark)

        if args.db:
            print(f"\nLoading from database: {args.db}")
            labels = generator.generate_labels_from_db(args.db, args.model_name)
        elif args.trajectories:
            print(f"\nLoading from file: {args.trajectories}")
            with open(args.trajectories, 'r') as f:
                trajectories = json.load(f)
            labels = []
            for traj in trajectories:
                labels.extend(generator.generate_labels_from_trajectory(traj))
        else:
            print("Error: Provide --db or --trajectories")
            sys.exit(1)

        print(f"\nGenerated {len(labels)} step labels")

        # Analyze distribution
        analysis = analyze_label_distribution(labels)
        print(f"\nLabel Distribution:")
        for label_val, count in sorted(analysis['label_distribution'].items()):
            print(f"  {label_val}: {count} ({count/len(labels)*100:.1f}%)")

        print(f"\nReason Distribution:")
        for reason, count in sorted(analysis['reason_distribution'].items(),
                                    key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {reason}: {count}")

        # Save labels
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "source": args.db or args.trajectories,
                "analysis": analysis
            },
            "labels": [asdict(l) for l in labels]
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"\nSaved to: {output_path}")

    elif args.command == "train":
        print("=" * 60)
        print("Training Process Reward Model")
        print("=" * 60)

        # Load labels
        print(f"\nLoading labels from: {args.labels}")
        with open(args.labels, 'r') as f:
            data = json.load(f)

        labels = [StepLabel(**l) for l in data['labels']]
        print(f"Loaded {len(labels)} labels")

        # Configure and train
        config = PRMConfig(
            model_name=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )

        prm = ProcessRewardModel(config=config)
        prm.train(labels, args.output)

        print(f"\nPRM saved to: {args.output}")

    elif args.command == "score":
        print("=" * 60)
        print("Scoring Steps with PRM")
        print("=" * 60)

        # Load PRM
        print(f"\nLoading PRM from: {args.prm}")
        prm = ProcessRewardModel.load(args.prm)

        # Load input
        with open(args.input, 'r') as f:
            steps = json.load(f)

        # Score steps
        results = []
        for step in steps:
            score = prm.score_step(
                state_text=step.get('state_text', ''),
                action_text=step.get('action_text', '')
            )
            results.append({**step, "prm_score": score})

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nScored {len(results)} steps")
        print(f"Average score: {sum(r['prm_score'] for r in results)/len(results):.3f}")
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()

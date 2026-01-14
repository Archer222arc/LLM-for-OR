#!/home/Archer/miniforge3/bin/python
"""
Evaluate local transformers models on OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/04_EVAL.md

This script evaluates locally hosted Qwen/LLaMA models using the same
benchmark infrastructure as the API-based evaluation. Results are stored
in SQLite database with incremental saving and resume support.

Usage:
    # Basic evaluation (creates outputs/experiments/{date}/{model}_{limit}samples/)
    python scripts/evaluation/evaluate_local_model.py \
        --model /data/qwen3_or_debug_merged \
        --limit 200

    # Custom experiment name
    python scripts/evaluation/evaluate_local_model.py \
        --model /data/qwen3_or_debug_merged \
        --limit 200 \
        --exp-name sft_eval

    # SGLang backend with concurrency
    python scripts/evaluation/evaluate_local_model.py \
        --backend sglang \
        --model /data/qwen3_or_debug_merged \
        --limit 200 \
        --concurrency 8 \
        --exp-name sft_sglang_200

Output structure:
    outputs/experiments/{YYYY-MM-DD}/{exp_name}/
    ├── config.yaml     # Experiment configuration
    ├── git_hash.txt    # Git commit hash
    ├── results.db      # SQLite database (primary storage)
    └── results.json    # Exported JSON (backward compatibility)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

from src.agents.base_agent import BaseAgent
from src.agents.prompts import SYSTEM_PROMPT, format_state
from src.environments import DebugState, Action, ActionType, SolverDebugEnv
from src.solvers import GurobiSolver
from src.evaluation.metrics import EpisodeResult, MetricsCalculator
from src.evaluation.result_db import ResultDB


# =============================================================================
# SGLangModelAgent - SGLang server-based agent (faster inference)
# =============================================================================

class SGLangModelAgent(BaseAgent):
    """
    SGLang server-based agent for OR debugging.

    Uses OpenAI-compatible API to call SGLang server for faster inference.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:30000/v1",
        model_name: str = "default",
        temperature: float = 0.0,
        max_tokens: int = 512,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or "sglang")

        self.base_url = base_url
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client for SGLang
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key="EMPTY")
        print(f"SGLang agent initialized, connecting to {base_url}")

        # Token tracking
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def act(self, state: DebugState) -> Action:
        """Select an action using SGLang server."""
        user_message = format_state(state)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else 0.0,
        )

        # Track tokens
        if response.usage:
            self._total_input_tokens += response.usage.prompt_tokens
            self._total_output_tokens += response.usage.completion_tokens
            self._total_tokens += response.usage.total_tokens

        content = response.choices[0].message.content
        return self._parse_response(content)

    def _parse_response(self, response: str) -> Action:
        """Parse LLM response to Action."""
        try:
            response = response.strip()
            start = response.find('{')
            end = response.rfind('}') + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                return self._parse_plain_text(response)

            action_name = data.get("action", "").upper()
            action_name = self._normalize_action_name(action_name)
            target = data.get("target")
            value = data.get("value")
            value2 = data.get("value2")
            action_type = getattr(ActionType, action_name, ActionType.GET_IIS)

            return Action(
                action_type=action_type,
                target=target,
                value=value,
                value2=value2
            )
        except Exception:
            return Action(ActionType.GET_IIS)

    def _normalize_action_name(self, name: str) -> str:
        """Normalize action name variations."""
        name = name.upper().replace(" ", "_").replace("-", "_")
        mappings = {
            "GETIIS": "GET_IIS", "IIS": "GET_IIS",
            "CHECKSLACK": "CHECK_SLACK",
            "DROPCONSTRAINT": "DROP_CONSTRAINT", "DROP": "DROP_CONSTRAINT",
            "RELAXCONSTRAINT": "RELAX_CONSTRAINT", "RELAX": "RELAX_CONSTRAINT",
            "UPDATERHS": "UPDATE_RHS", "UPDATEBOUNDS": "UPDATE_BOUNDS",
        }
        return mappings.get(name, name)

    def _parse_plain_text(self, response: str) -> Action:
        """Parse plain text response."""
        response_upper = response.upper()
        if "DROP" in response_upper and "CONSTRAINT" in response_upper:
            return Action(ActionType.DROP_CONSTRAINT)
        elif "RELAX" in response_upper:
            return Action(ActionType.RELAX_CONSTRAINT)
        elif "SUBMIT" in response_upper:
            return Action(ActionType.SUBMIT)
        elif "RESET" in response_upper:
            return Action(ActionType.RESET)
        else:
            return Action(ActionType.GET_IIS)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def get_token_stats(self) -> Dict[str, Any]:
        """Get token statistics."""
        return {
            'total_tokens': self._total_tokens,
            'total_input_tokens': self._total_input_tokens,
            'total_output_tokens': self._total_output_tokens,
            'total_reasoning_tokens': 0,
            'api_call_count': 0,
            'tokens_per_step': [],
        }


# =============================================================================
# LocalModelAgent - Transformers-based agent
# =============================================================================

class LocalModelAgent(BaseAgent):
    """
    Local transformers model agent for OR debugging.

    Uses a locally loaded HuggingFace model for inference.
    """

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or Path(model_path).name)

        self.model_path = model_path
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        # Load model and tokenizer
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        self.model.eval()
        print(f"Model loaded: {self.model.config.model_type}")

        # Token tracking
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def act(self, state: DebugState) -> Action:
        """Select an action using the local model."""
        user_message = format_state(state)

        # Build chat messages
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ]

        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            if self.temperature > 0:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        # Decode response (only new tokens)
        output_ids = outputs[0][input_length:]
        response = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Track tokens
        output_length = len(output_ids)
        self._total_input_tokens += input_length
        self._total_output_tokens += output_length
        self._total_tokens += input_length + output_length

        # Parse response to action
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Action:
        """Parse LLM response to Action."""
        try:
            # Try to extract JSON from response
            response = response.strip()

            # Find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1

            if start != -1 and end > start:
                json_str = response[start:end]
                data = json.loads(json_str)
            else:
                # No JSON found, try to parse as plain text
                return self._parse_plain_text(response)

            action_name = data.get("action", "").upper()
            action_name = self._normalize_action_name(action_name)

            target = data.get("target")
            value = data.get("value")
            value2 = data.get("value2")

            # Map to ActionType
            action_type = getattr(ActionType, action_name, ActionType.GET_IIS)

            return Action(
                action_type=action_type,
                target=target,
                value=value,
                value2=value2
            )
        except Exception as e:
            # Default to GET_IIS on parse failure
            return Action(ActionType.GET_IIS)

    def _normalize_action_name(self, name: str) -> str:
        """Normalize action name variations."""
        name = name.upper().replace(" ", "_").replace("-", "_")

        # Handle common variations
        mappings = {
            "GETIIS": "GET_IIS",
            "IIS": "GET_IIS",
            "CHECKSLACK": "CHECK_SLACK",
            "DROPCONSTRAINT": "DROP_CONSTRAINT",
            "DROP": "DROP_CONSTRAINT",
            "RELAXCONSTRAINT": "RELAX_CONSTRAINT",
            "RELAX": "RELAX_CONSTRAINT",
            "UPDATERHS": "UPDATE_RHS",
            "UPDATEBOUNDS": "UPDATE_BOUNDS",
        }

        return mappings.get(name, name)

    def _parse_plain_text(self, response: str) -> Action:
        """Parse plain text response."""
        response_upper = response.upper()

        if "DROP" in response_upper and "CONSTRAINT" in response_upper:
            return Action(ActionType.DROP_CONSTRAINT)
        elif "RELAX" in response_upper:
            return Action(ActionType.RELAX_CONSTRAINT)
        elif "SUBMIT" in response_upper:
            return Action(ActionType.SUBMIT)
        elif "RESET" in response_upper:
            return Action(ActionType.RESET)
        else:
            return Action(ActionType.GET_IIS)

    def reset(self) -> None:
        """Reset agent state."""
        self.clear_history()
        self._total_tokens = 0
        self._total_input_tokens = 0
        self._total_output_tokens = 0

    def get_token_stats(self) -> Dict[str, Any]:
        """Get token statistics."""
        return {
            'total_tokens': self._total_tokens,
            'total_input_tokens': self._total_input_tokens,
            'total_output_tokens': self._total_output_tokens,
            'total_reasoning_tokens': 0,
            'api_call_count': 0,
            'tokens_per_step': [],
        }


# =============================================================================
# Benchmark Loading
# =============================================================================

@dataclass
class BenchmarkProblem:
    """A benchmark problem instance."""
    problem_id: str
    env: SolverDebugEnv
    ground_truth_fix: Optional[str] = None
    ground_truth_iis: Optional[List[str]] = None
    original_objective: Optional[float] = None
    original_constraints: Optional[List[str]] = None


def load_benchmark(
    dataset_path: str,
    limit: Optional[int] = None
) -> List[BenchmarkProblem]:
    """Load benchmark dataset."""
    dataset_path = Path(dataset_path)

    if dataset_path.is_dir():
        json_path = dataset_path / "dataset.json"
    else:
        json_path = dataset_path

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    problems = []
    dataset_dir = json_path.parent

    # Apply limit early to avoid loading all problems
    problem_list = data.get('problems', data)
    if limit:
        problem_list = problem_list[:limit]

    for p in problem_list:
        model_path = Path(p['model_file'])
        if not model_path.is_absolute():
            model_path = dataset_dir / model_path

        if not model_path.exists():
            continue

        try:
            solver = GurobiSolver.from_file(str(model_path))
            env = SolverDebugEnv(
                solver=solver,
                problem_nl=p.get('problem_nl', ''),
                max_steps=20
            )

            problems.append(BenchmarkProblem(
                problem_id=p['problem_id'],
                env=env,
                ground_truth_fix=p.get('ground_truth_fix'),
                ground_truth_iis=p.get('iis_constraints', []),
                original_objective=p.get('original_objective'),
                original_constraints=p.get('constraint_names', [])
            ))
        except Exception as e:
            print(f"Error loading {p['problem_id']}: {e}")
            continue

    print(f"Loaded {len(problems)} problems")
    return problems


# =============================================================================
# Episode Execution
# =============================================================================

def run_episode(
    env: SolverDebugEnv,
    agent: LocalModelAgent,
    max_steps: int = 20
) -> EpisodeResult:
    """Run a single evaluation episode."""
    state, _ = env.reset()  # Gymnasium interface returns (state, info)
    agent.reset()

    trajectory = []
    diagnosed_constraints = []
    success_at_step = None
    total_reward = 0.0

    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Track diagnosed constraints
        if action.action_type == ActionType.DROP_CONSTRAINT and action.target:
            diagnosed_constraints.append(action.target)
        elif action.action_type == ActionType.RELAX_CONSTRAINT and action.target:
            diagnosed_constraints.append(action.target)

        trajectory.append({
            "step": step,
            "action": action.to_dict(),
            "reward": reward,
            "status": next_state.solver_status
        })

        total_reward += reward

        # Check for success
        if next_state.solver_status == "OPTIMAL" and success_at_step is None:
            success_at_step = step + 1

        if done:
            break

        state = next_state

    return EpisodeResult(
        success=next_state.solver_status == "OPTIMAL",
        final_status=next_state.solver_status,
        steps=len(trajectory),
        total_reward=total_reward,
        trajectory=trajectory,
        iis_actions=[],
        diagnosed_constraints=diagnosed_constraints,
        ground_truth_iis=[],
        ground_truth_fix=None,
        agent_name=agent.name,
        problem_id="",
        success_at_step=success_at_step,
        total_tokens=agent._total_tokens,
        total_input_tokens=agent._total_input_tokens,
        total_output_tokens=agent._total_output_tokens,
        total_reasoning_tokens=0,
        api_call_count=0,
        tokens_per_step=[],
        wall_clock_seconds=0.0
    )


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_metrics_from_results(results: List[Dict]) -> Dict[str, Any]:
    """Compute evaluation metrics from result dictionaries."""
    n = len(results)
    if n == 0:
        return {}

    successes = sum(1 for r in results if r['success'])
    recovery_rate = successes / n

    # RR@k calculations
    rr_at = {}
    for k in [5, 10, 15, 20]:
        count = sum(1 for r in results if r.get('success_at_step') and r['success_at_step'] <= k)
        rr_at[f'rr_at_{k}'] = count / n

    # Average steps for successful episodes
    successful_steps = [r['steps'] for r in results if r['success']]
    avg_steps = sum(successful_steps) / len(successful_steps) if successful_steps else 0

    # Diagnosis accuracy (basic)
    da_count = 0
    da_total = 0
    for r in results:
        gt_iis = r.get('ground_truth_iis', [])
        diagnosed = r.get('diagnosed_constraints', [])
        if gt_iis:
            for d in diagnosed:
                if d in gt_iis:
                    da_count += 1
            da_total += len(gt_iis)
    diagnosis_accuracy = da_count / da_total if da_total > 0 else 0

    # Token statistics
    total_tokens = sum(r.get('total_tokens', 0) for r in results)
    avg_tokens = total_tokens / n if n > 0 else 0

    return {
        'n_episodes': n,
        'recovery_rate': recovery_rate,
        'avg_steps': avg_steps,
        **rr_at,
        'diagnosis_accuracy': diagnosis_accuracy,
        'avg_tokens': avg_tokens,
        'total_tokens': total_tokens,
    }


# =============================================================================
# Experiment Directory Setup
# =============================================================================

def setup_experiment_dir(
    output_dir: str,
    exp_name: str,
    config: Dict[str, Any]
) -> Tuple[Path, ResultDB]:
    """
    Create experiment directory structure and initialize database.

    Args:
        output_dir: Base output directory
        exp_name: Experiment name
        config: Experiment configuration

    Returns:
        Tuple of (experiment directory path, ResultDB instance)
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    exp_dir = Path(output_dir) / date_str / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config.yaml
    with open(exp_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Save git_hash.txt
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=project_root
        ).decode().strip()
        (exp_dir / "git_hash.txt").write_text(git_hash)
    except subprocess.CalledProcessError:
        pass  # Not a git repo

    # Initialize database
    db = ResultDB(str(exp_dir / "results.db"))

    return exp_dir, db


# =============================================================================
# Main Evaluation Loop
# =============================================================================

def evaluate_single_problem(
    problem: BenchmarkProblem,
    sglang_url: str,
    temperature: float,
    model_name: str,
) -> EpisodeResult:
    """
    Evaluate a single problem using a fresh SGLang agent.

    Used for concurrent evaluation where each thread gets its own agent.

    Returns:
        EpisodeResult object (not converted to dict)
    """
    # Create a fresh agent for this thread
    agent = SGLangModelAgent(
        base_url=sglang_url,
        model_name="default",
        temperature=temperature,
        name=model_name
    )

    episode_start = time.time()
    result = run_episode(problem.env, agent, max_steps=20)
    result.problem_id = problem.problem_id
    result.ground_truth_iis = problem.ground_truth_iis or []
    result.ground_truth_fix = problem.ground_truth_fix
    result.wall_clock_seconds = time.time() - episode_start

    if problem.original_objective:
        result.original_objective = problem.original_objective
    if problem.original_constraints:
        result.original_constraints = set(problem.original_constraints)

    return result


def evaluate_model(
    model_path: str,
    dataset_path: str,
    output_dir: str = "outputs/experiments",
    exp_name: Optional[str] = None,
    limit: Optional[int] = None,
    temperature: float = 0.0,
    backend: str = "transformers",
    sglang_url: str = "http://localhost:30000/v1",
    concurrency: int = 1,
) -> Dict[str, Any]:
    """
    Evaluate a local model on OR-Debug-Bench.

    Args:
        model_path: Path to the local model
        dataset_path: Path to benchmark dataset
        output_dir: Base output directory (default: outputs/experiments)
        exp_name: Experiment name (default: {model}_{limit}samples)
        limit: Maximum number of problems to evaluate
        temperature: Sampling temperature
        backend: Inference backend ("transformers" or "sglang")
        sglang_url: SGLang server URL (only used with backend="sglang")
        concurrency: Number of concurrent evaluations (only for sglang backend)

    Returns:
        Evaluation summary
    """
    model_name = Path(model_path).name

    # Generate experiment name if not provided
    if exp_name is None:
        exp_name = f"{model_name}_{limit or 'full'}samples"

    # Prepare config
    config = {
        "model": model_path,
        "model_name": model_name,
        "dataset": dataset_path,
        "limit": limit,
        "temperature": temperature,
        "backend": backend,
        "sglang_url": sglang_url if backend == "sglang" else None,
        "concurrency": concurrency,
        "timestamp": datetime.now().isoformat(),
    }

    # Setup experiment directory and database
    exp_dir, db = setup_experiment_dir(output_dir, exp_name, config)

    print("=" * 60)
    print("Local Model Evaluation on OR-Debug-Bench")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Backend: {backend}")
    if backend == "sglang":
        print(f"SGLang URL: {sglang_url}")
        print(f"Concurrency: {concurrency}")
    print(f"Dataset: {dataset_path}")
    print(f"Limit: {limit}")
    print(f"Temperature: {temperature}")
    print(f"Output: {exp_dir}")
    print("=" * 60)

    # Load benchmark first (before creating agents)
    problems = load_benchmark(dataset_path, limit=limit)

    # Check for resume using database
    completed_ids = db.get_completed_problems(model_name)
    results: List[EpisodeResult] = []

    # Load existing results from database for metrics calculation
    if completed_ids:
        results = db.get_model_results(model_name)
        print(f"Resuming: {len(completed_ids)} already done")

    # Filter out completed problems
    problems = [p for p in problems if p.problem_id not in completed_ids]
    print(f"Remaining problems to evaluate: {len(problems)}")

    if len(problems) == 0:
        print("All problems already evaluated!")
        elapsed = 0.0
    else:
        # Run evaluation
        start_time = time.time()

        # Use concurrent evaluation for sglang backend with concurrency > 1
        if backend == "sglang" and concurrency > 1:
            print(f"\nRunning concurrent evaluation with {concurrency} workers...")

            def process_problem(problem):
                result = evaluate_single_problem(
                    problem=problem,
                    sglang_url=sglang_url,
                    temperature=temperature,
                    model_name=model_name
                )
                # Save to database immediately (thread-safe with WAL mode)
                db.save_episode_result(model_name, result)
                return result

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(process_problem, p): p for p in problems}

                for future in tqdm(as_completed(futures), total=len(problems), desc="Evaluating"):
                    result = future.result()
                    results.append(result)
        else:
            # Sequential evaluation
            if backend == "sglang":
                agent = SGLangModelAgent(
                    base_url=sglang_url,
                    model_name="default",
                    temperature=temperature,
                    name=model_name
                )
            else:
                agent = LocalModelAgent(
                    model_path=model_path,
                    temperature=temperature,
                    name=model_name
                )

            for problem in tqdm(problems, desc="Evaluating"):
                episode_start = time.time()

                result = run_episode(problem.env, agent, max_steps=20)
                result.problem_id = problem.problem_id
                result.ground_truth_iis = problem.ground_truth_iis or []
                result.ground_truth_fix = problem.ground_truth_fix
                result.wall_clock_seconds = time.time() - episode_start

                if problem.original_objective:
                    result.original_objective = problem.original_objective
                if problem.original_constraints:
                    result.original_constraints = set(problem.original_constraints)

                # Save to database immediately
                db.save_episode_result(model_name, result)
                results.append(result)

        elapsed = time.time() - start_time

    # Compute metrics using MetricsCalculator
    calc = MetricsCalculator()
    summary = calc.compute_summary(results)

    # Save summary to database
    db.update_summary(model_name, summary, agent_config=config, elapsed_seconds=elapsed)

    # Export JSON for backward compatibility
    json_path = exp_dir / "results.json"
    db.export_json(str(json_path))

    # Close database
    db.close()

    print(f"\nResults saved to: {exp_dir}")
    print_summary(summary, model_name, elapsed)

    # Return summary with paths
    return {
        "exp_dir": str(exp_dir),
        "db_path": str(exp_dir / "results.db"),
        "json_path": str(json_path),
        "summary": summary,
        "elapsed_seconds": elapsed,
    }


def print_summary(summary: Dict[str, Any], model_name: str, elapsed: float):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print(f"Evaluation Summary: {model_name}")
    print("=" * 60)
    print(f"Recovery Rate: {summary.get('recovery_rate', 0)*100:.1f}%")
    print(f"RR@5:  {summary.get('rr_at_5', 0)*100:.1f}%")
    print(f"RR@10: {summary.get('rr_at_10', 0)*100:.1f}%")
    print(f"RR@15: {summary.get('rr_at_15', 0)*100:.1f}%")
    print(f"RR@20: {summary.get('rr_at_20', 0)*100:.1f}%")
    print(f"Avg Steps: {summary.get('avg_steps', 0):.1f}")
    print(f"Diagnosis Accuracy: {summary.get('diagnosis_accuracy', 0)*100:.1f}%")
    print(f"Elapsed Time: {elapsed:.1f}s")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local model on OR-Debug-Bench"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to local model (or model name for sglang)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmarks/or_debug_bench_full",
        help="Path to benchmark dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/experiments",
        help="Base output directory (default: outputs/experiments)"
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (default: {model}_{limit}samples)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of problems to evaluate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 = greedy)"
    )
    # SGLang backend options
    parser.add_argument(
        "--backend",
        type=str,
        choices=["transformers", "sglang"],
        default="transformers",
        help="Inference backend: transformers (local) or sglang (server)"
    )
    parser.add_argument(
        "--sglang-url",
        type=str,
        default="http://localhost:30000/v1",
        help="SGLang server URL (only used with --backend sglang)"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of concurrent evaluations (only for sglang backend, enables continuous batching)"
    )
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        exp_name=args.exp_name,
        limit=args.limit,
        temperature=args.temperature,
        backend=args.backend,
        sglang_url=args.sglang_url,
        concurrency=args.concurrency,
    )


if __name__ == "__main__":
    main()

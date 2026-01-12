"""
Core metrics for OR-Debug-Bench evaluation.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/A4_Evaluation_Metrics.md

Key Components:
    - TokenUsage: Container for API call token usage (Test-Time Compute)
    - EpisodeResult: Container for episode outcome data
    - MetricsCalculator: Compute evaluation metrics from results

Example:
    >>> from src.evaluation import MetricsCalculator, EpisodeResult, TokenUsage
    >>> results = [EpisodeResult(success=True, steps=3, total_tokens=1500, ...)]
    >>> calc = MetricsCalculator()
    >>> summary = calc.compute_summary(results)
    >>> print(f"Token Efficiency: {summary['token_efficiency']:.2f}")
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import statistics


@dataclass
class TokenUsage:
    """
    Token usage for a single API call.

    Captures the test-time compute metrics from a single LLM API call,
    including input tokens, output tokens (including reasoning tokens for
    models like o1), and timing information.

    References:
        - [ICLR 2025] Scaling LLM Test-Time Compute Optimally
        - [arXiv 2506.12928] Scaling Test-time Compute for LLM Agents

    Attributes:
        input_tokens: Number of tokens in the prompt (context + user message)
        output_tokens: Number of tokens generated (including reasoning for o1)
        total_tokens: Sum of input + output tokens
        reasoning_tokens: Reasoning tokens for chain-of-thought models (o1, etc.)
        model: Model name used for this call
        provider: Provider name (openai, azure_openai, anthropic, azure_foundry)
        timestamp: ISO format timestamp of the call
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0  # For o1/thinking models
    model: str = ""
    provider: str = ""
    timestamp: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "model": self.model,
            "provider": self.provider,
            "timestamp": self.timestamp,
        }


@dataclass
class EpisodeResult:
    """
    Container for episode outcome data.

    Stores all relevant information from a single debugging episode,
    including success status, steps taken, rewards, and trajectory.

    Attributes:
        success: Whether the episode successfully recovered to OPTIMAL
        final_status: Final solver status (OPTIMAL, INFEASIBLE, etc.)
        steps: Total number of steps taken
        total_reward: Cumulative reward over the episode
        trajectory: List of (state, action, reward) dictionaries
        iis_actions: Actions taken on IIS constraints
        diagnosed_constraints: Constraints the agent identified as problematic
        ground_truth_iis: The actual IIS constraints from solver
        ground_truth_fix: The known correct fix (if available)
        agent_name: Name of the agent that ran this episode
        problem_id: Identifier for the problem instance
        original_objective: Original objective value (for OP calculation)
        recovered_objective: Recovered objective value (for OP calculation)
        original_constraint_count: Original number of constraints (for FP calculation)
        remaining_constraint_count: Remaining constraints after fix (for FP calculation)
        original_constraints: Set of original constraint names (for FP calculation)
        remaining_constraints: Set of remaining constraint names (for FP calculation)
        success_at_step: Step number when success was achieved (for RR@k)
    """

    success: bool
    final_status: str
    steps: int
    total_reward: float
    trajectory: List[Dict[str, Any]] = field(default_factory=list)
    iis_actions: List[str] = field(default_factory=list)
    diagnosed_constraints: List[str] = field(default_factory=list)
    ground_truth_iis: List[str] = field(default_factory=list)
    ground_truth_fix: Optional[str] = None
    agent_name: str = ""
    problem_id: str = ""
    # New fields for OP/FP/RR@k metrics
    original_objective: Optional[float] = None
    recovered_objective: Optional[float] = None
    original_constraint_count: int = 0
    remaining_constraint_count: int = 0
    original_constraints: List[str] = field(default_factory=list)
    remaining_constraints: List[str] = field(default_factory=list)
    success_at_step: Optional[int] = None  # Step when OPTIMAL was first achieved

    # Test-Time Compute tracking (new fields for token-based metrics)
    total_tokens: int = 0  # Total tokens consumed across all API calls
    total_input_tokens: int = 0  # Total input/prompt tokens
    total_output_tokens: int = 0  # Total output/completion tokens
    total_reasoning_tokens: int = 0  # Reasoning tokens (for o1/thinking models)
    api_call_count: int = 0  # Number of LLM API calls made
    tokens_per_step: List[int] = field(default_factory=list)  # Token usage per step
    api_calls: List[Dict[str, Any]] = field(default_factory=list)  # Detailed call logs
    wall_clock_seconds: float = 0.0  # Total elapsed time

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "final_status": self.final_status,
            "steps": self.steps,
            "total_reward": self.total_reward,
            "trajectory_length": len(self.trajectory),
            "iis_actions_count": len(self.iis_actions),
            "diagnosed_constraints": self.diagnosed_constraints,
            "ground_truth_iis": self.ground_truth_iis,
            "ground_truth_fix": self.ground_truth_fix,
            "agent_name": self.agent_name,
            "problem_id": self.problem_id,
            # OP/FP/RR@k fields
            "original_objective": self.original_objective,
            "recovered_objective": self.recovered_objective,
            "original_constraint_count": self.original_constraint_count,
            "remaining_constraint_count": self.remaining_constraint_count,
            "success_at_step": self.success_at_step,
            # Test-Time Compute fields
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "api_call_count": self.api_call_count,
            "tokens_per_step": self.tokens_per_step,
            "wall_clock_seconds": self.wall_clock_seconds,
        }


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmark runs.

    Attributes:
        max_steps: Maximum steps per episode
        n_episodes: Number of episodes to run per problem
        seed: Random seed for reproducibility
        verbose: Whether to print progress
    """

    max_steps: int = 50
    n_episodes: int = 1
    seed: Optional[int] = None
    verbose: bool = False


class MetricsCalculator:
    """
    Compute evaluation metrics from episode results.

    Provides methods to calculate:
    - Recovery Rate: Proportion of successful episodes
    - Average Steps: Mean steps to completion
    - Average Reward: Mean cumulative reward
    - Step Efficiency: Reward per step
    - Diagnosis Accuracy: IIS identification accuracy
    """

    def compute_recovery_rate(self, results: List[EpisodeResult]) -> float:
        """
        Compute recovery rate (success rate).

        Args:
            results: List of episode results

        Returns:
            Proportion of successful episodes (0.0 to 1.0)
        """
        if not results:
            return 0.0
        success_count = sum(1 for r in results if r.success)
        return success_count / len(results)

    def compute_avg_steps(self, results: List[EpisodeResult]) -> float:
        """
        Compute average steps per episode.

        Args:
            results: List of episode results

        Returns:
            Mean number of steps
        """
        if not results:
            return 0.0
        return statistics.mean(r.steps for r in results)

    def compute_median_steps(self, results: List[EpisodeResult]) -> float:
        """
        Compute median steps per episode.

        Args:
            results: List of episode results

        Returns:
            Median number of steps
        """
        if not results:
            return 0.0
        return statistics.median(r.steps for r in results)

    def compute_avg_reward(self, results: List[EpisodeResult]) -> float:
        """
        Compute average total reward.

        Args:
            results: List of episode results

        Returns:
            Mean cumulative reward
        """
        if not results:
            return 0.0
        return statistics.mean(r.total_reward for r in results)

    def compute_step_efficiency(self, results: List[EpisodeResult]) -> float:
        """
        Compute step efficiency (reward per step).

        Args:
            results: List of episode results

        Returns:
            Average reward divided by average steps
        """
        avg_steps = self.compute_avg_steps(results)
        if avg_steps == 0:
            return 0.0
        return self.compute_avg_reward(results) / avg_steps

    def compute_success_steps(
        self, results: List[EpisodeResult]
    ) -> Optional[float]:
        """
        Compute average steps for successful episodes only.

        Args:
            results: List of episode results

        Returns:
            Mean steps for successful episodes, or None if no successes
        """
        successful = [r for r in results if r.success]
        if not successful:
            return None
        return statistics.mean(r.steps for r in successful)

    def compute_diagnosis_accuracy(
        self,
        results: List[EpisodeResult],
        ground_truth_map: Optional[Dict[str, List[str]]] = None,
    ) -> Optional[float]:
        """
        Compute diagnosis accuracy for IIS identification.

        Measures whether the agent correctly identifies problematic constraints.
        Uses diagnosed_constraints from EpisodeResult if available,
        otherwise falls back to iis_actions.

        DA = |diagnosed ∩ actual_IIS| / |actual_IIS|

        Args:
            results: List of episode results
            ground_truth_map: Optional mapping from problem_id to IIS constraints
                              (used as fallback if result.ground_truth_iis is empty)

        Returns:
            Accuracy score (0.0 to 1.0), or None if no ground truth available
        """
        correct = 0
        total = 0

        for result in results:
            # Get ground truth IIS
            gt_iis = set(result.ground_truth_iis) if result.ground_truth_iis else set()
            if not gt_iis and ground_truth_map and result.problem_id in ground_truth_map:
                gt_iis = set(ground_truth_map[result.problem_id])

            if not gt_iis:
                continue

            # Get agent's diagnosed constraints
            diagnosed = set(result.diagnosed_constraints) if result.diagnosed_constraints else set()
            if not diagnosed:
                # Fallback to iis_actions (constraints dropped that were in IIS)
                diagnosed = set(result.iis_actions)

            # Compute recall: what fraction of actual IIS did agent identify?
            correct += len(gt_iis & diagnosed)
            total += len(gt_iis)

        if total == 0:
            return None
        return correct / total

    def compute_diagnosis_precision(
        self,
        results: List[EpisodeResult],
    ) -> Optional[float]:
        """
        Compute diagnosis precision for IIS identification.

        Precision = |diagnosed ∩ actual_IIS| / |diagnosed|
        Measures how accurate the agent's diagnoses are (avoiding false positives).

        Args:
            results: List of episode results

        Returns:
            Precision score (0.0 to 1.0), or None if no diagnoses made
        """
        correct = 0
        total = 0

        for result in results:
            gt_iis = set(result.ground_truth_iis) if result.ground_truth_iis else set()
            diagnosed = set(result.diagnosed_constraints) if result.diagnosed_constraints else set()

            if not diagnosed:
                diagnosed = set(result.iis_actions)

            if not diagnosed or not gt_iis:
                continue

            # Compute precision
            correct += len(gt_iis & diagnosed)
            total += len(diagnosed)

        if total == 0:
            return None
        return correct / total

    def compute_op(self, results: List[EpisodeResult]) -> Optional[float]:
        """
        Compute Optimality Preservation (OP) metric.

        OP measures how well the repair preserves the original objective value.
        Penalizes over-relaxation that leads to different optimal solutions.

        OP = max(0, 1 - |obj_recovered - obj_original| / |obj_original|)

        For minimization: a higher recovered objective is worse
        For maximization: a lower recovered objective is worse

        Args:
            results: List of episode results with objective values

        Returns:
            Average OP score (0.0 to 1.0), or None if no objective data
        """
        op_scores = []

        for result in results:
            if not result.success:
                continue
            if result.original_objective is None or result.recovered_objective is None:
                continue

            orig = result.original_objective
            recovered = result.recovered_objective

            if orig == 0:
                # Special case: if original objective is 0
                op = 1.0 if recovered == 0 else 0.0
            else:
                op = max(0.0, 1.0 - abs(recovered - orig) / abs(orig))

            op_scores.append(op)

        if not op_scores:
            return None
        return statistics.mean(op_scores)

    def compute_fp(self, results: List[EpisodeResult]) -> Optional[float]:
        """
        Compute Feasibility Preservation (FP) metric.

        FP measures how many original constraints are preserved after repair.
        Penalizes repairs that drop too many constraints.

        FP = |remaining_constraints ∩ original_constraints| / |original_constraints|

        Args:
            results: List of episode results with constraint data

        Returns:
            Average FP score (0.0 to 1.0), or None if no constraint data
        """
        fp_scores = []

        for result in results:
            if not result.success:
                continue

            # Use constraint sets if available
            if result.original_constraints and result.remaining_constraints:
                orig_set = set(result.original_constraints)
                remaining_set = set(result.remaining_constraints)
                if orig_set:
                    fp = len(remaining_set & orig_set) / len(orig_set)
                    fp_scores.append(fp)
            # Fallback to counts
            elif result.original_constraint_count > 0:
                fp = result.remaining_constraint_count / result.original_constraint_count
                fp_scores.append(min(1.0, fp))  # Cap at 1.0

        if not fp_scores:
            return None
        return statistics.mean(fp_scores)

    def compute_rr_at_k(self, results: List[EpisodeResult], k: int) -> float:
        """
        Compute Recovery Rate at k steps (RR@k).

        Measures the proportion of episodes that achieved success within k steps.
        Useful for test-time compute analysis.

        Args:
            results: List of episode results
            k: Maximum number of steps to consider

        Returns:
            Recovery rate at k steps (0.0 to 1.0)
        """
        if not results:
            return 0.0

        success_at_k = 0
        for result in results:
            # Check if success was achieved within k steps
            if result.success_at_step is not None:
                if result.success_at_step <= k:
                    success_at_k += 1
            elif result.success and result.steps <= k:
                # Fallback: if success_at_step not set, use final steps
                success_at_k += 1

        return success_at_k / len(results)

    def compute_faithfulness(
        self,
        results: List[EpisodeResult],
    ) -> Optional[float]:
        """
        Compute Faithfulness score.

        Faithfulness measures the consistency between agent's diagnosis
        and the actual IIS (Irreducible Infeasible Subsystem).

        Faithfulness = |agent_diagnosis ∩ actual_IIS| / |actual_IIS|

        This is equivalent to diagnosis_accuracy (recall), provided for clarity.

        Args:
            results: List of episode results

        Returns:
            Faithfulness score (0.0 to 1.0), or None if no ground truth
        """
        return self.compute_diagnosis_accuracy(results)

    # ==========================================================================
    # Test-Time Compute Metrics
    # ==========================================================================

    def _compute_avg_field(
        self, results: List[EpisodeResult], field: str
    ) -> float:
        """
        Compute average of a numeric field across results.

        Args:
            results: List of episode results
            field: Field name to average

        Returns:
            Average value, or 0 if no valid data
        """
        values = [getattr(r, field, 0) for r in results if getattr(r, field, 0) > 0]
        return sum(values) / len(values) if values else 0.0

    def compute_avg_tokens(self, results: List[EpisodeResult]) -> float:
        """
        Compute average total tokens consumed per episode.

        Args:
            results: List of episode results

        Returns:
            Average total tokens
        """
        tokens = [r.total_tokens for r in results if r.total_tokens > 0]
        return sum(tokens) / len(tokens) if tokens else 0.0

    def compute_tokens_per_step(self, results: List[EpisodeResult]) -> float:
        """
        Compute average tokens consumed per step.

        Args:
            results: List of episode results

        Returns:
            Tokens per step ratio
        """
        total_tokens = sum(r.total_tokens for r in results)
        total_steps = sum(r.steps for r in results)
        return total_tokens / total_steps if total_steps > 0 else 0.0

    def compute_token_efficiency(self, results: List[EpisodeResult]) -> float:
        """
        Compute token efficiency: success rate per thousand tokens.

        Higher values indicate better efficiency (more successes with fewer tokens).

        Token Efficiency = (Recovery Rate × 1000) / Avg Tokens

        Args:
            results: List of episode results

        Returns:
            Token efficiency score
        """
        rr = self.compute_recovery_rate(results)
        avg_tokens = self.compute_avg_tokens(results)
        return (rr * 1000 / avg_tokens) if avg_tokens > 0 else 0.0

    def compute_rr_at_token_budget(
        self, results: List[EpisodeResult], budget: int
    ) -> float:
        """
        Compute recovery rate within a given token budget.

        Measures what fraction of problems can be successfully solved
        using at most `budget` tokens.

        Args:
            results: List of episode results
            budget: Maximum token budget

        Returns:
            Recovery rate at token budget (0.0 to 1.0)
        """
        if not results:
            return 0.0

        # Count successes within budget
        successes_in_budget = sum(
            1 for r in results
            if r.success and r.total_tokens > 0 and r.total_tokens <= budget
        )

        return successes_in_budget / len(results)

    def compute_summary(
        self,
        results: List[EpisodeResult],
        ground_truth_map: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute comprehensive summary of all metrics.

        Args:
            results: List of episode results
            ground_truth_map: Optional ground truth for diagnosis accuracy

        Returns:
            Dictionary with all computed metrics
        """
        summary = {
            "n_episodes": len(results),
            "recovery_rate": self.compute_recovery_rate(results),
            "avg_steps": self.compute_avg_steps(results),
            "median_steps": self.compute_median_steps(results),
            "avg_reward": self.compute_avg_reward(results),
            "step_efficiency": self.compute_step_efficiency(results),
            "success_avg_steps": self.compute_success_steps(results),
        }

        # Add RR@k metrics for test-time compute analysis
        for k in [5, 10, 15, 20]:
            summary[f"rr_at_{k}"] = self.compute_rr_at_k(results, k)

        # Add OP (Optimality Preservation) if objective data available
        op = self.compute_op(results)
        if op is not None:
            summary["optimality_preservation"] = op

        # Add FP (Feasibility Preservation) if constraint data available
        fp = self.compute_fp(results)
        if fp is not None:
            summary["feasibility_preservation"] = fp

        # Add diagnosis metrics if ground truth available in results
        has_gt = any(r.ground_truth_iis for r in results)
        if has_gt or ground_truth_map:
            da = self.compute_diagnosis_accuracy(results, ground_truth_map)
            if da is not None:
                summary["diagnosis_accuracy"] = da
                summary["faithfulness"] = da  # Alias for clarity

            dp = self.compute_diagnosis_precision(results)
            if dp is not None:
                summary["diagnosis_precision"] = dp

        # ==========================================================================
        # Test-Time Compute Metrics (token-based)
        # ==========================================================================
        # Check if any results have token data
        has_token_data = any(r.total_tokens > 0 for r in results)

        if has_token_data:
            summary["avg_tokens"] = self.compute_avg_tokens(results)
            summary["avg_input_tokens"] = self._compute_avg_field(results, 'total_input_tokens')
            summary["avg_output_tokens"] = self._compute_avg_field(results, 'total_output_tokens')
            summary["avg_reasoning_tokens"] = self._compute_avg_field(results, 'total_reasoning_tokens')
            summary["tokens_per_step"] = self.compute_tokens_per_step(results)
            summary["token_efficiency"] = self.compute_token_efficiency(results)
            summary["avg_api_calls"] = self._compute_avg_field(results, 'api_call_count')
            summary["avg_wall_clock"] = self._compute_avg_field(results, 'wall_clock_seconds')

            # RR@TokenBudget for common budgets
            for budget in [5000, 10000, 20000, 50000]:
                summary[f"rr_at_{budget//1000}k_tokens"] = self.compute_rr_at_token_budget(results, budget)

        # Compute per-agent breakdown if multiple agents
        agent_names = set(r.agent_name for r in results if r.agent_name)
        if len(agent_names) > 1:
            summary["per_agent"] = {}
            for agent in agent_names:
                agent_results = [r for r in results if r.agent_name == agent]
                agent_summary = {
                    "n_episodes": len(agent_results),
                    "recovery_rate": self.compute_recovery_rate(agent_results),
                    "avg_steps": self.compute_avg_steps(agent_results),
                    "avg_reward": self.compute_avg_reward(agent_results),
                    "rr_at_5": self.compute_rr_at_k(agent_results, 5),
                    "rr_at_10": self.compute_rr_at_k(agent_results, 10),
                    "rr_at_20": self.compute_rr_at_k(agent_results, 20),
                }
                # Add token metrics for agent if available
                if any(r.total_tokens > 0 for r in agent_results):
                    agent_summary["avg_tokens"] = self.compute_avg_tokens(agent_results)
                    agent_summary["token_efficiency"] = self.compute_token_efficiency(agent_results)

                summary["per_agent"][agent] = agent_summary

        return summary

    def format_summary(self, summary: Dict[str, Any]) -> str:
        """
        Format summary dictionary as readable string.

        Args:
            summary: Summary dictionary from compute_summary

        Returns:
            Formatted string representation
        """
        lines = [
            "=" * 60,
            "OR-Debug-Bench Evaluation Summary",
            "=" * 60,
            f"Episodes:          {summary['n_episodes']}",
            f"Recovery Rate:     {summary['recovery_rate']:.2%}",
            f"Avg Steps:         {summary['avg_steps']:.2f}",
            f"Median Steps:      {summary['median_steps']:.2f}",
            f"Avg Reward:        {summary['avg_reward']:.2f}",
            f"Step Efficiency:   {summary['step_efficiency']:.2f}",
        ]

        if summary.get("success_avg_steps") is not None:
            lines.append(f"Success Steps:     {summary['success_avg_steps']:.2f}")

        # RR@k metrics
        lines.append("-" * 60)
        lines.append("Recovery Rate @ k Steps:")
        for k in [5, 10, 15, 20]:
            key = f"rr_at_{k}"
            if key in summary:
                lines.append(f"  RR@{k:<2}:           {summary[key]:.2%}")

        # Quality metrics
        if summary.get("optimality_preservation") is not None or summary.get("feasibility_preservation") is not None:
            lines.append("-" * 60)
            lines.append("Quality Metrics:")
            if summary.get("optimality_preservation") is not None:
                lines.append(f"  Optimality (OP):   {summary['optimality_preservation']:.2%}")
            if summary.get("feasibility_preservation") is not None:
                lines.append(f"  Feasibility (FP):  {summary['feasibility_preservation']:.2%}")

        # Diagnosis metrics
        if summary.get("diagnosis_accuracy") is not None or summary.get("diagnosis_precision") is not None:
            lines.append("-" * 60)
            lines.append("Diagnosis Metrics:")
            if summary.get("diagnosis_accuracy") is not None:
                lines.append(f"  Accuracy (DA):     {summary['diagnosis_accuracy']:.2%}")
            if summary.get("diagnosis_precision") is not None:
                lines.append(f"  Precision (DP):    {summary['diagnosis_precision']:.2%}")
            if summary.get("faithfulness") is not None:
                lines.append(f"  Faithfulness:      {summary['faithfulness']:.2%}")

        # Test-Time Compute metrics (token-based)
        if summary.get("avg_tokens") is not None:
            lines.append("-" * 60)
            lines.append("Test-Time Compute Metrics:")
            lines.append(f"  Avg Tokens:        {summary['avg_tokens']:,.0f}")
            if summary.get("avg_input_tokens"):
                lines.append(f"  Avg Input Tokens:  {summary['avg_input_tokens']:,.0f}")
            if summary.get("avg_output_tokens"):
                lines.append(f"  Avg Output Tokens: {summary['avg_output_tokens']:,.0f}")
            if summary.get("avg_reasoning_tokens") and summary.get("avg_reasoning_tokens") > 0:
                lines.append(f"  Avg Reasoning:     {summary['avg_reasoning_tokens']:,.0f}")
            if summary.get("tokens_per_step"):
                lines.append(f"  Tokens/Step:       {summary['tokens_per_step']:,.1f}")
            if summary.get("token_efficiency"):
                lines.append(f"  Token Efficiency:  {summary['token_efficiency']:.4f}")
            if summary.get("avg_api_calls"):
                lines.append(f"  Avg API Calls:     {summary['avg_api_calls']:.1f}")
            if summary.get("avg_wall_clock"):
                lines.append(f"  Avg Wall Clock:    {summary['avg_wall_clock']:.2f}s")

            # RR@TokenBudget
            has_token_budget = any(f"rr_at_{k}k_tokens" in summary for k in [5, 10, 20, 50])
            if has_token_budget:
                lines.append("-" * 60)
                lines.append("Recovery Rate @ Token Budget:")
                for budget_k in [5, 10, 20, 50]:
                    key = f"rr_at_{budget_k}k_tokens"
                    if key in summary:
                        lines.append(f"  RR@{budget_k}k:          {summary[key]:.2%}")

        if "per_agent" in summary:
            lines.append("-" * 60)
            lines.append("Per-Agent Breakdown:")
            for agent, metrics in summary["per_agent"].items():
                lines.append(
                    f"  {agent}: "
                    f"RR={metrics['recovery_rate']:.2%}, "
                    f"RR@10={metrics.get('rr_at_10', 0):.2%}, "
                    f"Steps={metrics['avg_steps']:.1f}"
                )

        lines.append("=" * 60)
        return "\n".join(lines)

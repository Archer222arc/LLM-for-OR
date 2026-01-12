"""
SQLite-based Result Storage for LLM Evaluation.

Provides concurrent-safe storage for evaluation results using SQLite.
Supports multi-process writes, resume functionality, and JSON export.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/plan/modules/04_EVAL.md
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Optional
from dataclasses import asdict

from .metrics import EpisodeResult, MetricsCalculator


class ResultDB:
    """
    SQLite database for storing evaluation results.

    Supports concurrent writes from multiple processes and provides
    resume functionality based on completed problem IDs.

    Example:
        >>> db = ResultDB("outputs/results.db")
        >>> db.save_episode_result("o4-mini", result)
        >>> completed = db.get_completed_problems("o4-mini")
        >>> db.export_json("outputs/results.json")
    """

    def __init__(self, db_path: str = "outputs/results.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Use WAL mode for better concurrent write performance
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None  # Autocommit mode
        )
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")

        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                problem_id TEXT NOT NULL,
                success INTEGER,
                final_status TEXT,
                steps INTEGER,
                total_reward REAL,
                diagnosed_constraints TEXT,
                ground_truth_iis TEXT,
                original_objective REAL,
                recovered_objective REAL,
                agent_name TEXT,
                timestamp TEXT,
                -- Test-Time Compute columns
                total_tokens INTEGER DEFAULT 0,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                reasoning_tokens INTEGER DEFAULT 0,
                api_call_count INTEGER DEFAULT 0,
                wall_clock_seconds REAL DEFAULT 0,
                UNIQUE(model_name, problem_id)
            );

            CREATE TABLE IF NOT EXISTS model_summaries (
                model_name TEXT PRIMARY KEY,
                n_episodes INTEGER,
                recovery_rate REAL,
                avg_steps REAL,
                median_steps REAL,
                avg_reward REAL,
                step_efficiency REAL,
                success_avg_steps REAL,
                rr_at_5 REAL,
                rr_at_10 REAL,
                rr_at_15 REAL,
                rr_at_20 REAL,
                optimality_preservation REAL,
                diagnosis_accuracy REAL,
                faithfulness REAL,
                diagnosis_precision REAL,
                agent_config TEXT,
                elapsed_seconds REAL,
                timestamp TEXT,
                -- Test-Time Compute columns
                avg_tokens REAL,
                avg_input_tokens REAL,
                avg_output_tokens REAL,
                avg_reasoning_tokens REAL,
                tokens_per_step REAL,
                token_efficiency REAL,
                avg_api_calls REAL,
                avg_wall_clock REAL
            );

            CREATE INDEX IF NOT EXISTS idx_model_problem
                ON evaluation_results(model_name, problem_id);
        """)

        # Migration: add new columns if they don't exist (for existing databases)
        self._migrate_schema()

    def _migrate_schema(self):
        """Add new columns to existing tables for backwards compatibility."""
        # Get existing columns in evaluation_results
        cursor = self.conn.execute("PRAGMA table_info(evaluation_results)")
        existing_cols = {row[1] for row in cursor.fetchall()}

        # Columns to add for evaluation_results
        token_cols = [
            ("total_tokens", "INTEGER DEFAULT 0"),
            ("input_tokens", "INTEGER DEFAULT 0"),
            ("output_tokens", "INTEGER DEFAULT 0"),
            ("reasoning_tokens", "INTEGER DEFAULT 0"),
            ("api_call_count", "INTEGER DEFAULT 0"),
            ("wall_clock_seconds", "REAL DEFAULT 0"),
        ]

        for col_name, col_type in token_cols:
            if col_name not in existing_cols:
                try:
                    self.conn.execute(
                        f"ALTER TABLE evaluation_results ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass  # Column might already exist

        # Get existing columns in model_summaries
        cursor = self.conn.execute("PRAGMA table_info(model_summaries)")
        existing_summary_cols = {row[1] for row in cursor.fetchall()}

        # Columns to add for model_summaries
        summary_token_cols = [
            ("avg_tokens", "REAL"),
            ("avg_input_tokens", "REAL"),
            ("avg_output_tokens", "REAL"),
            ("avg_reasoning_tokens", "REAL"),
            ("tokens_per_step", "REAL"),
            ("token_efficiency", "REAL"),
            ("avg_api_calls", "REAL"),
            ("avg_wall_clock", "REAL"),
        ]

        for col_name, col_type in summary_token_cols:
            if col_name not in existing_summary_cols:
                try:
                    self.conn.execute(
                        f"ALTER TABLE model_summaries ADD COLUMN {col_name} {col_type}"
                    )
                except sqlite3.OperationalError:
                    pass  # Column might already exist

    def save_episode_result(self, model_name: str, result: EpisodeResult) -> bool:
        """
        Save a single episode result to database.

        Uses INSERT OR REPLACE to handle duplicates (for resume).

        Args:
            model_name: Name of the model
            result: EpisodeResult object

        Returns:
            True if saved successfully
        """
        try:
            self.conn.execute("""
                INSERT OR REPLACE INTO evaluation_results (
                    model_name, problem_id, success, final_status, steps,
                    total_reward, diagnosed_constraints, ground_truth_iis,
                    original_objective, recovered_objective, agent_name, timestamp,
                    total_tokens, input_tokens, output_tokens, reasoning_tokens,
                    api_call_count, wall_clock_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_name,
                result.problem_id,
                1 if result.success else 0,
                result.final_status,
                result.steps,
                result.total_reward,
                json.dumps(getattr(result, 'diagnosed_constraints', [])),
                json.dumps(getattr(result, 'ground_truth_iis', [])),
                getattr(result, 'original_objective', None),
                getattr(result, 'recovered_objective', None),
                getattr(result, 'agent_name', model_name),
                datetime.now().isoformat(),
                # Test-Time Compute fields
                getattr(result, 'total_tokens', 0),
                getattr(result, 'total_input_tokens', 0),
                getattr(result, 'total_output_tokens', 0),
                getattr(result, 'total_reasoning_tokens', 0),
                getattr(result, 'api_call_count', 0),
                getattr(result, 'wall_clock_seconds', 0.0),
            ))
            return True
        except Exception as e:
            print(f"Error saving result: {e}")
            return False

    def get_completed_problems(self, model_name: str) -> Set[str]:
        """
        Get set of completed problem IDs for a model.

        Used for resume functionality.

        Args:
            model_name: Name of the model

        Returns:
            Set of problem IDs that have been evaluated
        """
        cursor = self.conn.execute(
            "SELECT problem_id FROM evaluation_results WHERE model_name = ?",
            (model_name,)
        )
        return {row[0] for row in cursor.fetchall()}

    def get_model_results(self, model_name: str) -> List[EpisodeResult]:
        """
        Get all results for a model.

        Args:
            model_name: Name of the model

        Returns:
            List of EpisodeResult objects including token data
        """
        cursor = self.conn.execute("""
            SELECT problem_id, success, final_status, steps, total_reward,
                   diagnosed_constraints, ground_truth_iis,
                   original_objective, recovered_objective, agent_name,
                   total_tokens, input_tokens, output_tokens, reasoning_tokens,
                   api_call_count, wall_clock_seconds
            FROM evaluation_results
            WHERE model_name = ?
        """, (model_name,))

        results = []
        for row in cursor.fetchall():
            results.append(EpisodeResult(
                problem_id=row[0],
                success=bool(row[1]),
                final_status=row[2],
                steps=row[3],
                total_reward=row[4],
                diagnosed_constraints=json.loads(row[5]) if row[5] else [],
                ground_truth_iis=json.loads(row[6]) if row[6] else [],
                original_objective=row[7],
                recovered_objective=row[8],
                agent_name=row[9],
                # Test-Time Compute fields
                total_tokens=row[10] or 0,
                total_input_tokens=row[11] or 0,
                total_output_tokens=row[12] or 0,
                total_reasoning_tokens=row[13] or 0,
                api_call_count=row[14] or 0,
                wall_clock_seconds=row[15] or 0.0,
            ))
        return results

    def update_summary(self, model_name: str, summary: Dict[str, Any],
                       agent_config: Optional[Dict] = None,
                       elapsed_seconds: Optional[float] = None):
        """
        Update or insert model summary.

        Args:
            model_name: Name of the model
            summary: Summary dictionary from MetricsCalculator
            agent_config: Optional agent configuration
            elapsed_seconds: Optional elapsed time
        """
        self.conn.execute("""
            INSERT OR REPLACE INTO model_summaries (
                model_name, n_episodes, recovery_rate, avg_steps, median_steps,
                avg_reward, step_efficiency, success_avg_steps,
                rr_at_5, rr_at_10, rr_at_15, rr_at_20,
                optimality_preservation, diagnosis_accuracy, faithfulness,
                diagnosis_precision, agent_config, elapsed_seconds, timestamp,
                avg_tokens, avg_input_tokens, avg_output_tokens, avg_reasoning_tokens,
                tokens_per_step, token_efficiency, avg_api_calls, avg_wall_clock
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_name,
            summary.get('n_episodes'),
            summary.get('recovery_rate'),
            summary.get('avg_steps'),
            summary.get('median_steps'),
            summary.get('avg_reward'),
            summary.get('step_efficiency'),
            summary.get('success_avg_steps'),
            summary.get('rr_at_5'),
            summary.get('rr_at_10'),
            summary.get('rr_at_15'),
            summary.get('rr_at_20'),
            summary.get('optimality_preservation'),
            summary.get('diagnosis_accuracy'),
            summary.get('faithfulness'),
            summary.get('diagnosis_precision'),
            json.dumps(agent_config) if agent_config else None,
            elapsed_seconds,
            datetime.now().isoformat(),
            # Test-Time Compute metrics
            summary.get('avg_tokens'),
            summary.get('avg_input_tokens'),
            summary.get('avg_output_tokens'),
            summary.get('avg_reasoning_tokens'),
            summary.get('tokens_per_step'),
            summary.get('token_efficiency'),
            summary.get('avg_api_calls'),
            summary.get('avg_wall_clock'),
        ))

    def get_all_models(self) -> List[str]:
        """Get list of all model names in database."""
        cursor = self.conn.execute(
            "SELECT DISTINCT model_name FROM evaluation_results"
        )
        return [row[0] for row in cursor.fetchall()]

    def get_summary(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get summary for a model."""
        cursor = self.conn.execute(
            "SELECT * FROM model_summaries WHERE model_name = ?",
            (model_name,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        columns = [desc[0] for desc in cursor.description]
        return dict(zip(columns, row))

    def export_json(self, output_path: str):
        """
        Export all results to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        models = self.get_all_models()

        output = {
            "timestamp": datetime.now().isoformat(),
            "database": str(self.db_path),
            "models": {}
        }

        for model_name in models:
            results = self.get_model_results(model_name)
            summary = self.get_summary(model_name)

            per_problem_results = []
            for r in results:
                result_dict = {
                    'problem_id': r.problem_id,
                    'success': r.success,
                    'final_status': r.final_status,
                    'steps': r.steps,
                    'total_reward': r.total_reward,
                    'diagnosed_constraints': getattr(r, 'diagnosed_constraints', []),
                    'ground_truth_iis': getattr(r, 'ground_truth_iis', []),
                    'original_objective': getattr(r, 'original_objective', None),
                    'recovered_objective': getattr(r, 'recovered_objective', None),
                    # Test-Time Compute fields
                    'total_tokens': getattr(r, 'total_tokens', 0),
                    'total_input_tokens': getattr(r, 'total_input_tokens', 0),
                    'total_output_tokens': getattr(r, 'total_output_tokens', 0),
                    'total_reasoning_tokens': getattr(r, 'total_reasoning_tokens', 0),
                    'api_call_count': getattr(r, 'api_call_count', 0),
                    'wall_clock_seconds': getattr(r, 'wall_clock_seconds', 0.0),
                }
                per_problem_results.append(result_dict)

            output["models"][model_name] = {
                "summary": summary,
                "per_problem_results": per_problem_results,
            }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, default=str)

        print(f"Exported {len(models)} models to {output_path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        cursor = self.conn.execute("""
            SELECT model_name, COUNT(*) as count,
                   SUM(success) as successes
            FROM evaluation_results
            GROUP BY model_name
        """)

        stats = {}
        for row in cursor.fetchall():
            stats[row[0]] = {
                "count": row[1],
                "successes": row[2],
                "recovery_rate": row[2] / row[1] if row[1] > 0 else 0
            }
        return stats

    def close(self):
        """Close database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

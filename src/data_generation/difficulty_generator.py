"""
Difficulty-stratified problem generation for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-16_dgro_training_results.md

Key Components:
    - DifficultyLevel: Enum for difficulty tiers (easy/medium/difficult)
    - DifficultyConfig: Configuration for each difficulty level
    - get_difficulty_config: Returns config for specified difficulty

Difficulty Levels:
    - Easy: IIS 1-2, Type A-C, expected SFT RR@5 >= 95%
    - Medium: IIS 3-5, Type D/E/F, expected SFT RR@5 75-85%
    - Difficult: IIS 5-10, Type E/G/H/I, expected SFT RR@5 50-70%

Example:
    >>> from src.data_generation.difficulty_generator import DifficultyConfig
    >>> config = DifficultyConfig.get_config("difficult")
    >>> config.error_types
    ['E', 'G', 'H', 'I']
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple


class DifficultyLevel(Enum):
    """Difficulty levels for OR-Debug-Bench problems."""

    EASY = "easy"
    MEDIUM = "medium"
    DIFFICULT = "difficult"


@dataclass
class DifficultyConfig:
    """Configuration for a difficulty level.

    Attributes:
        level: The difficulty level
        iis_range: (min, max) IIS size range
        error_types: List of error type codes to include
        expected_rr5: (min, max) expected RR@5 range for SFT
        description: Human-readable description
    """

    level: DifficultyLevel
    iis_range: Tuple[int, int]
    error_types: List[str]
    expected_rr5: Tuple[float, float]
    description: str

    # Predefined configurations
    EASY = None  # Set below after class definition
    MEDIUM = None
    DIFFICULT = None

    @classmethod
    def get_config(cls, difficulty: str) -> "DifficultyConfig":
        """Get configuration for specified difficulty level.

        Args:
            difficulty: One of "easy", "medium", "difficult"

        Returns:
            DifficultyConfig for the specified level

        Raises:
            ValueError: If difficulty is not recognized
        """
        configs = {
            "easy": cls.EASY,
            "medium": cls.MEDIUM,
            "difficult": cls.DIFFICULT,
        }

        if difficulty.lower() not in configs:
            raise ValueError(
                f"Unknown difficulty: {difficulty}. "
                f"Choose from: {list(configs.keys())}"
            )

        return configs[difficulty.lower()]

    def matches_problem(self, iis_size: int, error_type: str) -> bool:
        """Check if a problem matches this difficulty configuration.

        Args:
            iis_size: Number of constraints in IIS
            error_type: Error type code (A-I)

        Returns:
            True if problem matches this difficulty level
        """
        iis_min, iis_max = self.iis_range
        return (
            iis_min <= iis_size <= iis_max
            and error_type in self.error_types
        )


# Define the difficulty configurations
DifficultyConfig.EASY = DifficultyConfig(
    level=DifficultyLevel.EASY,
    iis_range=(1, 2),
    error_types=["A", "B", "C"],
    expected_rr5=(0.90, 1.0),
    description=(
        "Easy problems: Single-constraint IIS (1-2), simple error types (A-C). "
        "Expected SFT RR@5 >= 90%. Baseline for model sanity check."
    )
)

DifficultyConfig.MEDIUM = DifficultyConfig(
    level=DifficultyLevel.MEDIUM,
    iis_range=(3, 5),
    error_types=["D", "E", "F"],
    expected_rr5=(0.75, 0.85),
    description=(
        "Medium problems: Multi-constraint IIS (3-5), conflict/hidden errors (D-F). "
        "Expected SFT RR@5 75-85%. Tests basic debugging ability."
    )
)

DifficultyConfig.DIFFICULT = DifficultyConfig(
    level=DifficultyLevel.DIFFICULT,
    iis_range=(5, 10),
    error_types=["E", "G", "H", "I"],
    expected_rr5=(0.50, 0.70),
    description=(
        "Difficult problems: Large IIS (5-10), MDP-advantage types (G-I). "
        "Expected SFT RR@5 50-70%. Provides room for RL improvement."
    )
)


def get_error_types_for_difficulty(difficulty: str) -> List[str]:
    """Get error types for a specific difficulty level.

    Args:
        difficulty: One of "easy", "medium", "difficult"

    Returns:
        List of error type codes
    """
    config = DifficultyConfig.get_config(difficulty)
    return config.error_types


def get_iis_range_for_difficulty(difficulty: str) -> Tuple[int, int]:
    """Get IIS size range for a specific difficulty level.

    Args:
        difficulty: One of "easy", "medium", "difficult"

    Returns:
        (min_iis, max_iis) tuple
    """
    config = DifficultyConfig.get_config(difficulty)
    return config.iis_range


def classify_problem_difficulty(iis_size: int, error_type: str) -> str:
    """Classify a problem's difficulty based on IIS size and error type.

    Args:
        iis_size: Number of constraints in IIS
        error_type: Error type code (A-I)

    Returns:
        Difficulty level as string ("easy", "medium", or "difficult")
    """
    # Check in order: difficult -> medium -> easy
    # This ensures hard problems are classified correctly
    for level in ["difficult", "medium", "easy"]:
        config = DifficultyConfig.get_config(level)
        if config.matches_problem(iis_size, error_type):
            return level

    # Default to medium if no match
    return "medium"

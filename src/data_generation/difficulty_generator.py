"""
Difficulty-stratified problem generation for OR-Debug-Bench.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/progress/2026-01-17_phase4_rl_improvement_plan.md

Key Components:
    - DifficultyLevel: Enum for difficulty tiers (easy/medium/hard)
    - DifficultyConfig: Configuration for each difficulty level
    - PerTypeDifficultyConfig: Per-type difficulty configuration (Phase 4)
    - get_difficulty_config: Returns config for specified difficulty

Per-Type Difficulty (Phase 4):
    Each error type (A, B, C, D) has its own easy/medium/hard variants:
    - Type A (Semantic): 1/3/5 constraints, obfuscation control
    - Type B (Bounds): 1/3/5 variables, chaining control
    - Type C (Logic): 1/2/3 nested levels
    - Type D (RHS): 1/3/5 conflicts, cascading control

Legacy Difficulty Levels (cross-type):
    - Easy: IIS 1-2, Type A-C, expected SFT RR@5 >= 95%
    - Medium: IIS 3-5, Type D/E/F, expected SFT RR@5 75-85%
    - Difficult: IIS 5-10, Type E/G/H/I, expected SFT RR@5 50-70%

Example:
    >>> from src.data_generation.difficulty_generator import PerTypeDifficultyConfig
    >>> config = PerTypeDifficultyConfig.get_config("A", "hard")
    >>> config.num_constraints
    5
    >>> config.obfuscation
    True
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any


class DifficultyLevel(Enum):
    """Difficulty levels for OR-Debug-Bench problems."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"  # Renamed from DIFFICULT for clarity
    DIFFICULT = "difficult"  # Alias for backward compatibility


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


# =============================================================================
# Phase 4: Per-Type Difficulty Configuration
# =============================================================================

# Global registry for per-type configs (outside dataclass to avoid mutable default issue)
_PER_TYPE_REGISTRY: Dict[str, "PerTypeDifficultyConfig"] = {}


@dataclass
class PerTypeDifficultyConfig:
    """Configuration for per-type difficulty stratification.

    Each error type (A, B, C, D) has specific parameters that control
    difficulty within that type.

    Attributes:
        error_type: The error type code (A, B, C, D)
        difficulty: The difficulty level (easy, medium, hard)
        num_constraints: Number of constraints to inject
        num_variables: Number of variables involved
        obfuscation: Whether to use misleading names
        chaining: Whether to create chained dependencies
        cascading: Whether to create cascading conflicts
        expected_sft_rr5: Expected SFT RR@5 performance
        params: Additional type-specific parameters
    """

    error_type: str
    difficulty: str
    num_constraints: int = 1
    num_variables: int = 1
    obfuscation: bool = False
    chaining: bool = False
    cascading: bool = False
    expected_sft_rr5: Tuple[float, float] = (0.9, 1.0)
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def register(cls, config: "PerTypeDifficultyConfig") -> None:
        """Register a configuration in the registry."""
        key = f"{config.error_type}_{config.difficulty}"
        _PER_TYPE_REGISTRY[key] = config

    @classmethod
    def get_config(cls, error_type: str, difficulty: str) -> "PerTypeDifficultyConfig":
        """Get configuration for specified error type and difficulty.

        Args:
            error_type: Error type code (A, B, C, D)
            difficulty: Difficulty level (easy, medium, hard)

        Returns:
            PerTypeDifficultyConfig for the specified combination

        Raises:
            ValueError: If combination is not found
        """
        key = f"{error_type.upper()}_{difficulty.lower()}"
        if key not in _PER_TYPE_REGISTRY:
            raise ValueError(
                f"No configuration for {error_type} {difficulty}. "
                f"Available: {list(_PER_TYPE_REGISTRY.keys())}"
            )
        return _PER_TYPE_REGISTRY[key]

    @classmethod
    def get_all_configs(cls) -> List["PerTypeDifficultyConfig"]:
        """Get all registered configurations."""
        return list(_PER_TYPE_REGISTRY.values())

    @classmethod
    def get_configs_for_type(cls, error_type: str) -> List["PerTypeDifficultyConfig"]:
        """Get all difficulty configs for a specific error type."""
        return [
            config for config in _PER_TYPE_REGISTRY.values()
            if config.error_type == error_type.upper()
        ]

    @classmethod
    def get_configs_for_difficulty(cls, difficulty: str) -> List["PerTypeDifficultyConfig"]:
        """Get all type configs for a specific difficulty level."""
        return [
            config for config in _PER_TYPE_REGISTRY.values()
            if config.difficulty == difficulty.lower()
        ]


# =============================================================================
# Type A: Constraint Direction Flip (Semantic)
# =============================================================================

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="A",
    difficulty="easy",
    num_constraints=1,
    obfuscation=False,
    expected_sft_rr5=(0.90, 1.0),
    params={
        "description": "Single constraint flip, obvious naming",
        "flip_type": "simple",  # Just flip <= to >=
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="A",
    difficulty="medium",
    num_constraints=3,
    obfuscation=False,
    expected_sft_rr5=(0.70, 0.85),
    params={
        "description": "Multiple constraint flips, some interdependent",
        "flip_type": "interdependent",
        "interdependency_depth": 2,
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="A",
    difficulty="hard",
    num_constraints=5,
    obfuscation=True,
    expected_sft_rr5=(0.40, 0.60),
    params={
        "description": "Many flips with misleading names and deep dependencies",
        "flip_type": "obfuscated",
        "interdependency_depth": 3,
        "use_misleading_names": True,
    }
))


# =============================================================================
# Type B: Variable Type Modification (Bounds)
# =============================================================================

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="B",
    difficulty="easy",
    num_variables=1,
    chaining=False,
    expected_sft_rr5=(0.90, 1.0),
    params={
        "description": "Single variable type change",
        "bound_type": "simple",
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="B",
    difficulty="medium",
    num_variables=3,
    chaining=False,
    expected_sft_rr5=(0.70, 0.85),
    params={
        "description": "Multiple variable changes with dependent bounds",
        "bound_type": "dependent",
        "create_linking_constraints": True,
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="B",
    difficulty="hard",
    num_variables=5,
    chaining=True,
    expected_sft_rr5=(0.40, 0.60),
    params={
        "description": "Chained variable bounds with hidden dependencies",
        "bound_type": "chained",
        "chain_length": 4,
        "hidden_dependency": True,
    }
))


# =============================================================================
# Type C: Expression Term Removal (Logic)
# =============================================================================

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="C",
    difficulty="easy",
    num_constraints=1,
    expected_sft_rr5=(0.90, 1.0),
    params={
        "description": "Single coefficient change",
        "modification_type": "removal",
        "nested_level": 0,
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="C",
    difficulty="medium",
    num_constraints=2,
    expected_sft_rr5=(0.70, 0.85),
    params={
        "description": "Multiple coefficient changes with some interaction",
        "modification_type": "sign_flip",
        "nested_level": 1,
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="C",
    difficulty="hard",
    num_constraints=3,
    obfuscation=True,
    expected_sft_rr5=(0.40, 0.60),
    params={
        "description": "Complex coefficient changes with cascading effects",
        "modification_type": "scale_cascade",
        "nested_level": 2,
        "cascade_effect": True,
    }
))


# =============================================================================
# Type D: Contradicting Constraint (RHS Conflicts)
# =============================================================================

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="D",
    difficulty="easy",
    num_constraints=1,
    cascading=False,
    expected_sft_rr5=(0.90, 1.0),
    params={
        "description": "Single direct conflict",
        "conflict_type": "simple",
        "target_iis_size": 2,
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="D",
    difficulty="medium",
    num_constraints=3,
    cascading=False,
    expected_sft_rr5=(0.65, 0.80),
    params={
        "description": "Multiple RHS conflicts",
        "conflict_type": "multi",
        "target_iis_size": 4,
    }
))

PerTypeDifficultyConfig.register(PerTypeDifficultyConfig(
    error_type="D",
    difficulty="hard",
    num_constraints=5,
    cascading=True,
    expected_sft_rr5=(0.35, 0.55),
    params={
        "description": "Cascading RHS conflicts with chain dependencies",
        "conflict_type": "cascade",
        "target_iis_size": 6,
        "chain_length": 4,
    }
))


# =============================================================================
# Helper Functions for Per-Type Difficulty
# =============================================================================

def get_per_type_config(error_type: str, difficulty: str) -> PerTypeDifficultyConfig:
    """Get per-type difficulty configuration.

    Args:
        error_type: Error type code (A, B, C, D)
        difficulty: Difficulty level (easy, medium, hard)

    Returns:
        PerTypeDifficultyConfig instance
    """
    return PerTypeDifficultyConfig.get_config(error_type, difficulty)


def get_all_per_type_combinations() -> List[Tuple[str, str]]:
    """Get all valid (error_type, difficulty) combinations.

    Returns:
        List of (error_type, difficulty) tuples
    """
    return [
        (config.error_type, config.difficulty)
        for config in PerTypeDifficultyConfig.get_all_configs()
    ]


def get_benchmark_name(error_type: str, difficulty: str) -> str:
    """Generate standard benchmark name for a type-difficulty combination.

    Args:
        error_type: Error type code (A, B, C, D)
        difficulty: Difficulty level (easy, medium, hard)

    Returns:
        Benchmark name like "type_A_easy"
    """
    return f"type_{error_type.upper()}_{difficulty.lower()}"


def validate_per_type_config(error_type: str, difficulty: str) -> bool:
    """Check if a per-type difficulty configuration exists.

    Args:
        error_type: Error type code
        difficulty: Difficulty level

    Returns:
        True if configuration exists
    """
    try:
        PerTypeDifficultyConfig.get_config(error_type, difficulty)
        return True
    except ValueError:
        return False

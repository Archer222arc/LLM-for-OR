"""
Pytest configuration and fixtures for LLM-for-OR tests.

Research Direction: Direction A (OR-Debug-Bench)
Documentation: docs/directions/A_OR_Debug_Bench/
"""

import pytest
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture
def sample_infeasible_model():
    """Create a simple infeasible optimization model for testing."""
    return {
        "name": "test_infeasible",
        "status": "INFEASIBLE",
        "constraints": ["c1", "c2", "c3"],
        "iis": ["c1", "c2"]
    }


@pytest.fixture
def sample_state():
    """Create a sample MDP state for testing."""
    return {
        "problem": "Simple assignment problem",
        "code": "m.addConstr(x[0] + x[1] <= 1)",
        "status": "INFEASIBLE",
        "feedback": {"iis": ["c1", "c2"]},
        "history": []
    }


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create a temporary data directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir

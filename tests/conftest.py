import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hridaytwin import synth  # noqa: E402


@pytest.fixture(scope="session")
def small_cohort():
    """Small, fast cohort shared by all tests."""
    return synth.simulate(n=160, days=90, seed=123)

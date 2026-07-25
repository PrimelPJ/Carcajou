"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator — use this instead of np.random.default_rng(N) in tests."""
    return np.random.default_rng(42)

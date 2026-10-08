"""Shared fixtures: a small synthetic market on the minute grid."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.fixture(scope="session")
def small_market():
    """A 6,000-minute synthetic market: long enough for the 3750-minute beta."""
    from generate_synthetic_data import generate
    from cryptoforecast.data import to_grid

    df, details = generate(n_minutes=6_000, seed=7, signal_strength=1.0)
    weights = {int(r["Asset_ID"]): float(r["Weight"]) for r in details.iter_rows(named=True)}
    return df, to_grid(df, weights)

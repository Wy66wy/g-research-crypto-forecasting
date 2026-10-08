"""Metric tests."""

from __future__ import annotations

import numpy as np
import pytest

from cryptoforecast.metrics import pearson, per_asset_correlation, weighted_correlation


@pytest.fixture
def panel():
    rng = np.random.default_rng(0)
    assets = np.repeat([0, 1, 2], 300)
    y = rng.normal(size=900)
    return y, assets


class TestPearson:
    def test_perfect(self):
        y = np.arange(10.0)
        assert pearson(y, y) == pytest.approx(1.0)

    def test_constant_prediction_scores_zero_not_nan(self):
        """A constant carries no information, so it contributes nothing -
        and it must not turn the weighted average into nan."""
        assert pearson(np.arange(10.0), np.ones(10)) == 0.0

    def test_ignores_nan_pairs(self):
        y = np.array([1.0, 2.0, np.nan, 4.0])
        assert pearson(y, y) == pytest.approx(1.0)


class TestWeightedCorrelation:
    def test_is_computed_per_asset(self, panel):
        """A prediction that is perfect within each asset but carries a
        different offset per asset must still score 1: the metric is a
        per-asset time-series correlation, not a pooled one."""
        y, assets = panel
        pred = y + assets * 100.0
        assert weighted_correlation(y, pred, assets) == pytest.approx(1.0)

    def test_invariant_to_per_asset_rescaling(self, panel):
        """Correlation ignores scale, so a different multiplier per asset
        changes nothing. This is what lets the network train on
        standardised targets without recalibrating afterwards."""
        y, assets = panel
        rng = np.random.default_rng(1)
        p = y + rng.normal(size=len(y))
        scaled = p * np.array([1.0, 50.0, 0.01])[assets]
        assert weighted_correlation(y, scaled, assets) == pytest.approx(
            weighted_correlation(y, p, assets), rel=1e-9)

    def test_weights_matter(self, panel):
        y, assets = panel
        pred = np.where(assets == 0, y, -y)            # right on 0, wrong on 1, 2
        heavy0 = weighted_correlation(y, pred, assets, weights={0: 10, 1: 1, 2: 1})
        light0 = weighted_correlation(y, pred, assets, weights={0: 1, 1: 10, 2: 10})
        assert heavy0 > light0

    def test_components_are_reported(self, panel):
        y, assets = panel
        score, parts = weighted_correlation(y, y, assets, return_components=True)
        assert set(parts) == {0, 1, 2}

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            per_asset_correlation(np.ones(3), np.ones(4), np.ones(3))

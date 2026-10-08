"""Feature tests, centred on a perturbation test for lookahead.

Checking that each shift points the right way is fragile: it tests the code
you remembered to check. The perturbation test below is stronger because it
is indifferent to how features are computed. It replaces everything after
minute t with garbage, rebuilds the features, and requires every feature at
or before t to be unchanged. Any lookahead anywhere - a wrong shift, a
centred window, a market statistic computed over the future - fails it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptoforecast.data import MarketGrid
from cryptoforecast.features import (
    FeatureConfig,
    backward_log_return,
    build_feature_tensor,
    rolling_std,
    to_long,
)


def _corrupt_future(grid: MarketGrid, t: int, seed: int = 0) -> MarketGrid:
    rng = np.random.default_rng(seed)
    fields = {}
    for k, v in grid.fields.items():
        v = v.copy()
        v[t + 1 :] = v[t + 1 :] * rng.uniform(0.2, 5.0, v[t + 1 :].shape)
        fields[k] = v
    present = grid.present.copy()
    present[t + 1 :] = rng.random(present[t + 1 :].shape) < 0.5
    return MarketGrid(grid.timestamps, list(grid.asset_ids), fields, present,
                      grid.target, dict(grid.weights))


class TestNoLookahead:
    @pytest.mark.parametrize("t", [500, 2_000, 4_500])
    def test_features_ignore_the_future(self, small_market, t):
        _, grid = small_market
        clean, names = build_feature_tensor(grid)
        dirty, _ = build_feature_tensor(_corrupt_future(grid, t))
        a, b = clean[: t + 1], dirty[: t + 1]
        same = (a == b) | (np.isnan(a) & np.isnan(b))
        bad = [names[f] for f in range(len(names)) if not same[:, :, f].all()]
        assert not bad, f"these features changed when only the future did: {bad}"

    def test_own_only_config_also_causal(self, small_market):
        _, grid = small_market
        cfg = FeatureConfig(use_cross_asset=False)
        t = 3_000
        clean, _ = build_feature_tensor(grid, cfg)
        dirty, _ = build_feature_tensor(_corrupt_future(grid, t), cfg)
        np.testing.assert_array_equal(np.nan_to_num(clean[: t + 1]),
                                      np.nan_to_num(dirty[: t + 1]))

    def test_no_target_derived_feature_exists(self, small_market):
        _, grid = small_market
        _, names = build_feature_tensor(grid)
        assert not any("target" in n.lower() or "beta" in n.lower() for n in names)


class TestPrimitives:
    def test_backward_return_looks_back(self):
        close = np.exp(np.arange(20, dtype=float))[:, None]
        r = backward_log_return(close, 5)
        assert np.isnan(r[:5]).all()
        assert r[10, 0] == pytest.approx(5.0)

    def test_rolling_std_matches_numpy(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(100, 2))
        out = rolling_std(x, 20)
        assert out[50, 0] == pytest.approx(np.std(x[31:51, 0], ddof=1), rel=1e-9)

    def test_rolling_std_tolerates_scattered_nans(self):
        x = np.random.default_rng(2).normal(size=(100, 1))
        x[40] = np.nan
        assert np.isfinite(rolling_std(x, 20)[50, 0])


class TestCrossAssetToggle:
    def test_own_only_drops_market_and_leader_features(self, small_market):
        _, grid = small_market
        _, names = build_feature_tensor(grid, FeatureConfig(use_cross_asset=False))
        assert not any(n.startswith(("mkt_", "rel_", "lead")) for n in names)

    def test_market_feature_is_shared_across_assets(self, small_market):
        _, grid = small_market
        X, names = build_feature_tensor(grid)
        f = names.index("mkt_ret_15")
        row = X[4_000, :, f]
        assert np.allclose(row, row[0], equal_nan=True)


class TestToLong:
    def test_keeps_only_scorable_cells(self, small_market):
        _, grid = small_market
        X, _ = build_feature_tensor(grid)
        rows = np.arange(4_000, 4_100)
        Xl, y, assets, minutes = to_long(X, grid, rows)
        assert np.isfinite(y).all()
        assert grid.present[minutes, [grid.asset_ids.index(a) for a in assets]].all()

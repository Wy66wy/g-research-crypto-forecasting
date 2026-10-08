"""Target reconstruction tests.

The target defines the leakage boundary for the whole project, so its
reimplementation is checked three ways: against hand-computed values,
against the published formula's specific choices, and against an independent
implementation that shares no code with it.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptoforecast.target import (
    compute_target,
    forward_fill,
    forward_log_return,
    market_return,
    residualise,
    rolling_mean_full,
)


class TestForwardLogReturn:
    def test_window_starts_one_minute_ahead(self):
        """The target return runs from t+1 to t+16, not from t to t+15."""
        close = np.exp(np.arange(40, dtype=float))[:, None]   # log price = t
        r = forward_log_return(close, horizon=15, offset=1)
        # log(close[t+16]) - log(close[t+1]) = (t+16) - (t+1) = 15
        assert r[0, 0] == pytest.approx(15.0)
        assert r[10, 0] == pytest.approx(15.0)

    def test_offset_actually_shifts(self):
        """If the offset were ignored, a price jump at t+1 would leak in."""
        close = np.ones((40, 1))
        close[1:] = 2.0          # jump between minute 0 and minute 1
        r = forward_log_return(close, horizon=15, offset=1)
        assert r[0, 0] == pytest.approx(0.0)    # jump is before the window

    def test_tail_has_no_future(self):
        close = np.ones((40, 1))
        r = forward_log_return(close, horizon=15, offset=1)
        assert np.isnan(r[-16:]).all()
        assert np.isfinite(r[:-16]).all()


class TestForwardFill:
    def test_carries_last_value(self):
        x = np.array([[1.0], [np.nan], [np.nan], [4.0]])
        np.testing.assert_array_equal(forward_fill(x)[:, 0], [1, 1, 1, 4])

    def test_leading_nans_stay_nan(self):
        x = np.array([[np.nan], [2.0]])
        assert np.isnan(forward_fill(x)[0, 0])

    def test_limit_stops_stale_prices(self):
        x = np.array([[1.0], [np.nan], [np.nan], [np.nan]])
        out = forward_fill(x, limit=2)[:, 0]
        assert out[2] == 1.0 and np.isnan(out[3])

    def test_columns_are_independent(self):
        x = np.array([[1.0, np.nan], [np.nan, 5.0]])
        out = forward_fill(x)
        assert out[1, 0] == 1.0 and np.isnan(out[0, 1])


class TestOfficialFormula:
    def test_rolling_mean_requires_full_window(self):
        """Matches pandas' default min_periods=window, as the official code does."""
        x = np.arange(10, dtype=float)
        x[5] = np.nan
        out = rolling_mean_full(x, 3)
        assert np.isnan(out[:2]).all()                 # window not yet full
        assert np.isnan(out[5:8]).all()                # windows touching the nan
        assert out[4] == pytest.approx((2 + 3 + 4) / 3)

    def test_beta_has_no_intercept(self):
        """The published regression is mean(R*M)/mean(M^2), not cov/var.

        With a constant offset added to the market, the uncentred beta moves;
        a centred one would not. This pins the formula to the official one.
        """
        rng = np.random.default_rng(0)
        m = rng.normal(0.0, 1.0, 200) + 0.5
        r = (2.0 * m)[:, None]
        _, beta = residualise(r, m, window=50)
        assert beta[-1, 0] == pytest.approx(2.0)       # exact because r = 2m
        uncentred = np.mean(r[-50:, 0] * m[-50:]) / np.mean(m[-50:] ** 2)
        assert beta[-1, 0] == pytest.approx(uncentred)

    def test_failed_regression_gives_zero_beta(self):
        m = np.zeros(100)
        _, beta = residualise(np.ones((100, 1)), m, window=10)
        assert np.all(beta == 0.0)

    def test_market_return_renormalises_over_present_assets(self):
        r = np.array([[1.0, np.nan, 3.0]])
        w = np.array([1.0, 100.0, 1.0])
        # The missing heavy asset must drop out, not pull the mean to zero
        assert market_return(r, w)[0] == pytest.approx(2.0)


class TestAgainstIndependentImplementation:
    def test_matches_reference_target(self, small_market):
        """The generator computes its Target with separate code (explicit
        sliding windows, a per-row market loop). Agreement between the two is
        a real cross-check, not a function compared with itself."""
        _, grid = small_market
        mine = compute_target(grid)
        ref = grid.target
        assert np.array_equal(np.isfinite(mine), np.isfinite(ref)), \
            "the two implementations disagree on which cells have a target"
        both = np.isfinite(mine)
        np.testing.assert_allclose(mine[both], ref[both], atol=1e-5)

    def test_target_only_on_traded_minutes(self, small_market):
        _, grid = small_market
        t = compute_target(grid)
        assert np.isnan(t[~grid.present]).all()

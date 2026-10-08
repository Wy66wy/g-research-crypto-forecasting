"""Minute-grid tests.

The grid exists because the raw data has gaps, and on gappy data "16 rows
ahead" is not "16 minutes ahead". These tests pin that down.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from cryptoforecast.data import to_grid


def _frame(rows):
    return pl.DataFrame(
        rows,
        schema=["timestamp", "Asset_ID", "Count", "Open", "High", "Low",
                "Close", "Volume", "VWAP"],
        orient="row",
    )


class TestGrid:
    def test_gaps_become_explicit_missing_minutes(self):
        df = _frame([
            (0, 1, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0),
            (180, 1, 1.0, 1.0, 1.0, 1.0, 13.0, 1.0, 1.0),   # 3 minutes later
        ])
        g = to_grid(df, asset_ids=[1])
        assert g.n_minutes == 4                       # 0, 60, 120, 180
        np.testing.assert_array_equal(g.present[:, 0], [True, False, False, True])
        assert np.isnan(g.fields["Close"][1, 0])

    def test_timestamps_are_consecutive(self, small_market):
        _, g = small_market
        assert np.all(np.diff(g.timestamps) == 60)

    def test_a_shift_on_the_grid_is_a_shift_in_minutes(self):
        """The bug the grid prevents. On the raw frame, the next row after
        minute 0 is minute 3; on the grid, it is minute 1."""
        df = _frame([
            (0, 1, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0),
            (180, 1, 1.0, 1.0, 1.0, 1.0, 13.0, 1.0, 1.0),
        ])
        raw_next = df["Close"].to_list()[1]
        g = to_grid(df, asset_ids=[1])
        grid_next = g.fields["Close"][1, 0]
        assert raw_next == 13.0
        assert np.isnan(grid_next)

    def test_assets_land_in_their_own_columns(self):
        df = _frame([
            (0, 1, 1.0, 1.0, 1.0, 1.0, 10.0, 1.0, 1.0),
            (0, 6, 1.0, 1.0, 1.0, 1.0, 20.0, 1.0, 1.0),
        ])
        g = to_grid(df, asset_ids=[1, 6])
        assert g.fields["Close"][0, 0] == 10.0
        assert g.fields["Close"][0, 1] == 20.0

    def test_misaligned_timestamps_raise(self):
        df = _frame([(0, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                     (90, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)])
        with pytest.raises(ValueError, match="whole minutes"):
            to_grid(df, asset_ids=[1])

    def test_slice_is_independent(self, small_market):
        _, g = small_market
        s = g.slice(100, 200)
        assert s.n_minutes == 100
        assert s.timestamps[0] == g.timestamps[100]

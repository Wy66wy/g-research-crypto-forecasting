"""Live inference tests.

The training code predicts from one big feature tensor; the live code
rebuilds the grid from a rolling buffer every minute. Those are different
code paths, and the only thing that makes the training score meaningful is
that they compute the same function.

``scripts/check_consistency.py`` does this at scale. These tests pin down the
same property on a small case so a regression shows up in the test suite
rather than only when someone remembers to run the script.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptoforecast.config import ExperimentConfig
from cryptoforecast.data import to_grid
from cryptoforecast.features import build_feature_tensor
from cryptoforecast.inference import BUFFER_MINUTES, CryptoPredictor
from cryptoforecast.models import get_model


@pytest.fixture(scope="module")
def predictor_setup(small_market):
    df, grid = small_market
    tensor, names = build_feature_tensor(grid)
    model = get_model("ridge").fit(
        tensor, grid, np.arange(3_800, 5_400), feature_names=names
    )
    cfg = ExperimentConfig()
    cfg.train.model = "ridge"
    return df, grid, tensor, model, cfg


class TestLiveMatchesBatch:
    def test_predictions_are_identical(self, predictor_setup):
        """The headline property: minute-by-minute equals batch."""
        df, grid, tensor, model, cfg = predictor_setup
        p = CryptoPredictor(model, cfg, grid.weights)

        # Long enough that skipping the buffer warm-up still leaves a
        # meaningful number of cells to compare
        n_live = 260
        start_ts = int(grid.timestamps[-n_live])
        warm_from = start_ts - BUFFER_MINUTES * 60

        seed = df.filter(
            (df["timestamp"] < start_ts) & (df["timestamp"] >= warm_from)
        )
        for _, minute in seed.group_by("timestamp", maintain_order=True):
            p._buffer.append(minute.drop("Target", strict=False))

        rows = np.arange(grid.n_minutes - n_live, grid.n_minutes)
        batch = model.predict(tensor, grid, rows)

        col = {a: j for j, a in enumerate(grid.asset_ids)}
        window = df.filter(df["timestamp"] >= start_ts).sort(["timestamp", "Asset_ID"])
        live = np.full_like(batch, np.nan)
        row_of = {int(t): i for i, t in enumerate(grid.timestamps[rows])}

        for _, minute in window.group_by("timestamp", maintain_order=True):
            out = p.predict(minute.drop("Target", strict=False))
            i = row_of[int(minute["timestamp"][0])]
            for a, v in zip(minute["Asset_ID"].to_list(), out):
                live[i, col[int(a)]] = v

        # Skip the start, where the buffer is still filling and the longest
        # features are legitimately incomplete
        skip = 90
        ok = np.isfinite(batch[skip:]) & np.isfinite(live[skip:])
        # Sanity-check the comparison itself: (n_live - skip) minutes x 14
        # coins, minus the cells where a coin did not trade. An earlier
        # version asserted a fixed count larger than the window could ever
        # contain, so the test failed while the predictions were in fact
        # identical to 2e-10.
        assert ok.sum() > 0.7 * (n_live - skip) * grid.n_assets
        np.testing.assert_allclose(batch[skip:][ok], live[skip:][ok], atol=1e-6)


class TestLiveSafety:
    def test_output_length_matches_input_rows(self, predictor_setup):
        df, grid, _, model, cfg = predictor_setup
        p = CryptoPredictor(model, cfg, grid.weights)
        for _, minute in df.head(2_000).group_by("timestamp", maintain_order=True):
            out = p.predict(minute.drop("Target", strict=False))
            assert len(out) == minute.height

    def test_never_returns_nan(self, predictor_setup):
        """A nan in a submission fails it outright, so a coin that cannot be
        predicted must come back as 0 - which contributes nothing to a
        correlation - rather than nan."""
        df, grid, _, model, cfg = predictor_setup
        p = CryptoPredictor(model, cfg, grid.weights)
        for _, minute in df.head(600).group_by("timestamp", maintain_order=True):
            assert np.isfinite(p.predict(minute.drop("Target", strict=False))).all()

    def test_buffer_stays_bounded(self, predictor_setup):
        """The buffer must evict old minutes, or a full submission run would
        grow it to the length of the test set."""
        df, grid, _, model, cfg = predictor_setup
        p = CryptoPredictor(model, cfg, grid.weights)
        for _, minute in df.head(20_000).group_by("timestamp", maintain_order=True):
            p.predict(minute.drop("Target", strict=False))
        assert len(p._buffer) <= BUFFER_MINUTES + 1

    def test_empty_minute_is_handled(self, predictor_setup):
        df, grid, _, model, cfg = predictor_setup
        p = CryptoPredictor(model, cfg, grid.weights)
        empty = df.head(0).drop("Target", strict=False)
        assert len(p.predict(empty)) == 0

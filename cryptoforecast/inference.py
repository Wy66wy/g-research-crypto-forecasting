"""Inference for the evaluation API.

The API delivers one minute at a time: a small frame with a row per asset
that traded in that minute, and expects a prediction per row back.

Every feature here needs history - a 60-minute return needs the last 60
minutes - so the predictor keeps a rolling buffer of raw rows and rebuilds
the minute grid from it at each step. Rebuilding is wasteful in principle,
but the buffer is small (a few hundred minutes x 14 coins), it guarantees the
features are computed by *exactly* the same code as in training, and that
guarantee is worth far more than the microseconds.

The buffer must reach back at least as far as the longest feature window
plus the network's lookback, and :data:`BUFFER_MINUTES` is sized for that
with room to spare.

:func:`replay` drives the predictor minute by minute over historical data,
the same way the API will. It exercises the code path a vectorised backtest
never touches, and it is the check to run before submitting.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import numpy as np
import polars as pl

from .config import ExperimentConfig
from .constants import ASSET_COL, TIMESTAMP_COL
from .data import MarketGrid, to_grid
from .features import build_feature_tensor
from .metrics import weighted_correlation
from .models import load_model
from .train import CROSS_PREFIXES

logger = logging.getLogger(__name__)

__all__ = ["CryptoPredictor", "replay", "BUFFER_MINUTES"]

#: Minutes of history retained. The longest feature window is 60 minutes and
#: the network looks back 30 more, so 240 leaves comfortable headroom.
BUFFER_MINUTES = 240


class CryptoPredictor:
    """Model plus a rolling history buffer, exposed as a per-minute callback."""

    def __init__(self, model, config: ExperimentConfig, weights: dict[int, float]) -> None:
        self.model = model
        self.config = config
        self.weights = weights
        self._buffer: deque = deque()
        self._n_calls = 0

    @classmethod
    def from_artifacts(cls, artifacts_dir, name: str, weights: dict[int, float]):
        artifacts_dir = Path(artifacts_dir)
        config = ExperimentConfig.from_yaml(artifacts_dir / f"{name}_config.yaml")
        ext = {"lgbm": ".txt", "ridge": ".pkl"}.get(config.train.model, ".pt")
        model = load_model(config.train.model, artifacts_dir / f"{name}_model{ext}")
        return cls(model, config, weights)

    def _features(self, grid: MarketGrid):
        tensor, names = build_feature_tensor(grid, self.config.features)
        if self.config.train.own_only:
            keep = [i for i, n in enumerate(names) if not n.startswith(CROSS_PREFIXES)]
            tensor, names = tensor[:, :, keep], [names[i] for i in keep]
        return tensor, names

    def predict(self, minute_df: pl.DataFrame) -> np.ndarray:
        """Predict the target for every row of one minute's frame.

        Returns an array aligned with ``minute_df``'s rows. Anything that
        cannot be predicted - not enough history yet, a coin the grid has
        never seen - comes back as 0.0, which contributes nothing to a
        correlation, rather than nan, which would fail the submission.
        """
        self._n_calls += 1
        if minute_df.height == 0:
            return np.zeros(0)

        self._buffer.append(minute_df)
        ts_now = int(minute_df[TIMESTAMP_COL][0])
        cutoff = ts_now - BUFFER_MINUTES * 60
        while self._buffer and int(self._buffer[0][TIMESTAMP_COL][0]) < cutoff:
            self._buffer.popleft()

        history = pl.concat(list(self._buffer), how="diagonal_relaxed")
        grid = to_grid(history, self.weights)
        tensor, _ = self._features(grid)

        last = np.array([grid.n_minutes - 1])
        pred_row = self.model.predict(tensor, grid, last)[0]   # (A,)

        col = {a: j for j, a in enumerate(grid.asset_ids)}
        out = np.zeros(minute_df.height)
        for i, a in enumerate(minute_df[ASSET_COL].to_list()):
            j = col.get(int(a))
            if j is not None and np.isfinite(pred_row[j]):
                out[i] = float(pred_row[j])
        return out


def replay(predictor: CryptoPredictor, df: pl.DataFrame, verbose_every: int = 500) -> dict:
    """Run the predictor minute by minute and score it against the target.

    The target column is dropped from every frame before the predictor sees
    it: it does not exist live, and leaving it in would make the replay
    differ from the real thing in precisely the way that matters.
    """
    df = df.sort([TIMESTAMP_COL, ASSET_COL])
    preds, truth, assets = [], [], []

    for i, (_, minute) in enumerate(df.group_by(TIMESTAMP_COL, maintain_order=True)):
        y = minute["Target"].to_numpy() if "Target" in minute.columns else None
        p = predictor.predict(minute.drop("Target", strict=False))
        if y is not None:
            preds.append(p)
            truth.append(y)
            assets.append(minute[ASSET_COL].to_numpy())
        if verbose_every and (i + 1) % verbose_every == 0:
            logger.info("  replayed %d minutes", i + 1)

    p = np.concatenate(preds)
    y = np.concatenate(truth)
    a = np.concatenate(assets)
    ok = np.isfinite(y) & np.isfinite(p)
    score, per_asset = weighted_correlation(y[ok], p[ok], a[ok], predictor.weights,
                                            return_components=True)
    logger.info("replay score %.5f over %d minutes", score, predictor._n_calls)
    return {"score": score, "per_asset": per_asset, "n_minutes": predictor._n_calls}

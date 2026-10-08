"""Input windows for the cross-asset neural network.

The network consumes, for each sample minute ``t``, the last ``L`` minutes of
features for **all fourteen assets at once**: a tensor of shape (A, L, F).
That is what lets it relate one coin's history to the others'.

**Memory.** Materialising every window up front would cost
``n_samples x 14 x L x F x 4`` bytes. On the full dataset with L=30 and a
sample every minute, that is about 25GB. Instead the (T, A, F) feature tensor
is stored once, and windows are sliced out of it on demand inside
``__getitem__``. The tensor already exists for the gradient-boosted models,
so the network adds almost nothing on top.

**Per-asset target standardisation.** Volatility differs by an order of
magnitude across coins, so on raw targets a squared-error loss is dominated
by the wildest ones. The metric, though, is a correlation computed per asset,
and correlation is blind to scale. So targets are divided by each asset's
training-period standard deviation before the loss sees them. The network is
then pushed to get every coin's *direction* right rather than to fit the
biggest coins' magnitudes - and because correlation ignores scale, nothing
needs to be undone at prediction time.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["FeatureScaler", "WindowDataset", "sample_rows"]


class FeatureScaler:
    """Per-feature standardisation fitted on training minutes only.

    Statistics are pooled across assets, so a feature means the same thing
    for every coin. Missing values become 0 after scaling - the feature's
    training mean - and the ``is_present`` channel tells the network when that
    happened.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, tensor: np.ndarray, rows: np.ndarray) -> "FeatureScaler":
        sub = tensor[rows].reshape(-1, tensor.shape[2]).astype(np.float64)
        self.mean_ = np.nanmean(sub, axis=0)
        self.std_ = np.nanstd(sub, axis=0)
        self.mean_ = np.nan_to_num(self.mean_)
        self.std_ = np.where(np.isfinite(self.std_) & (self.std_ > 1e-12), self.std_, 1.0)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.mean_) / self.std_
        # Clip: a 50-sigma minute (they happen in crypto) should not be able
        # to dominate a batch's gradient on its own
        z = np.clip(z, -10.0, 10.0)
        return np.nan_to_num(z, nan=0.0).astype(np.float32)


def sample_rows(rows: np.ndarray, stride: int, seed: int = 0) -> np.ndarray:
    """Subsample minutes for training.

    Consecutive minutes have heavily overlapping 15-minute targets, so
    adjacent samples are nearly duplicates. Taking every ``stride``-th minute
    loses very little information and divides the cost of an epoch. A random
    offset keeps successive runs from always seeing the same phase.
    """
    rng = np.random.default_rng(seed)
    offset = int(rng.integers(0, max(stride, 1)))
    return rows[offset::stride]


class WindowDataset:
    """Lazily sliced (A, L, F) windows plus per-asset targets.

    Implements the minimal protocol torch's DataLoader needs (``__len__`` and
    ``__getitem__``) without importing torch, so the class is usable and
    testable on machines without it.
    """

    def __init__(
        self,
        scaled: np.ndarray,
        target: np.ndarray | None,
        rows: np.ndarray,
        lookback: int,
        target_scale: np.ndarray | None = None,
    ) -> None:
        self.x = scaled                         # (T, A, F), already standardised
        self.target = target                    # (T, A) or None
        self.lookback = lookback
        # A window needs `lookback` minutes of history behind it
        self.rows = rows[rows >= lookback - 1]
        self.target_scale = target_scale        # (A,)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        t = int(self.rows[i])
        window = self.x[t - self.lookback + 1 : t + 1]      # (L, A, F)
        window = np.transpose(window, (1, 0, 2))             # (A, L, F)

        if self.target is None:
            y = np.zeros(window.shape[0], dtype=np.float32)
            mask = np.ones(window.shape[0], dtype=np.float32)
        else:
            y = self.target[t].astype(np.float64)
            mask = np.isfinite(y).astype(np.float32)
            if self.target_scale is not None:
                y = y / self.target_scale
            y = np.nan_to_num(y, nan=0.0).astype(np.float32)
        return window.astype(np.float32), y, mask, t

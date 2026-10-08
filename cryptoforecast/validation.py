"""Time-based validation with an embargo sized to the target.

Random splits are wrong here for the usual time-series reasons, and there is
one reason specific to this target worth spelling out.

**The embargo must cover the target's lookahead.** The label on minute ``t``
is built from prices up to ``t + 16``. So a training row at minute 999
carries information about minutes 1000-1015. If validation starts at 1000,
the last sixteen training labels describe the validation period, and the
score is inflated by however much those overlapping minutes are worth.

The default gap here is 60 minutes, which covers the 16-minute label window
with room to spare and also covers the longest rolling feature, so a
validation row's feature window never reaches back into data the model was
fitted on.

What the embargo does *not* fix: the target's beta is estimated over a
trailing 3750-minute window, so the first ~2.6 days of any validation fold
use betas partly estimated on training-period returns. That is part of the
label definition rather than a leak a model can exploit - the beta is never a
feature - so it is noted rather than engineered around.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .constants import TARGET_LOOKAHEAD

logger = logging.getLogger(__name__)

__all__ = ["TimeSplit", "holdout", "summarise_folds"]


@dataclass
class TimeSplit:
    """Walk-forward folds over minute indices.

    Attributes
    ----------
    n_folds : int
        Number of validation folds.
    valid_minutes : int
        Length of each validation block. A week (10,080 minutes) by default.
    gap_minutes : int
        Embargo between train and validation. Must be at least
        TARGET_LOOKAHEAD (16); the default 60 also covers rolling features.
    train_minutes : int or None
        Rolling training window length. None means expanding: every fold
        trains on everything before its embargo.
    warmup_minutes : int
        Skip this many minutes at the start of the series, where rolling
        features and the target's beta have not formed yet.
    """

    n_folds: int = 3
    valid_minutes: int = 10_080
    gap_minutes: int = 60
    train_minutes: int | None = None
    warmup_minutes: int = 3_750

    def __post_init__(self) -> None:
        if self.gap_minutes < TARGET_LOOKAHEAD:
            raise ValueError(
                f"gap_minutes={self.gap_minutes} is shorter than the target's "
                f"{TARGET_LOOKAHEAD}-minute lookahead; training labels would "
                f"overlap the validation period"
            )

    def split(self, n_minutes: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield (train_rows, valid_rows) minute-index arrays."""
        usable_end = n_minutes - TARGET_LOOKAHEAD   # last minutes have no label
        need = self.warmup_minutes + self.n_folds * self.valid_minutes + self.gap_minutes
        if usable_end < need + 1_000:
            raise ValueError(
                f"only {n_minutes} minutes available, need at least "
                f"{need + 1_000} for {self.n_folds} folds"
            )

        for k in range(self.n_folds):
            v_end = usable_end - (self.n_folds - 1 - k) * self.valid_minutes
            v_start = v_end - self.valid_minutes
            t_end = v_start - self.gap_minutes
            t_start = (
                self.warmup_minutes
                if self.train_minutes is None
                else max(self.warmup_minutes, t_end - self.train_minutes)
            )
            if t_end - t_start < 1_000:
                raise ValueError(f"fold {k}: training window too short")
            logger.info(
                "fold %d: train [%d, %d)  gap %d  valid [%d, %d)",
                k, t_start, t_end, self.gap_minutes, v_start, v_end,
            )
            yield np.arange(t_start, t_end), np.arange(v_start, v_end)


def holdout(
    n_minutes: int,
    valid_minutes: int = 10_080,
    gap_minutes: int = 60,
    warmup_minutes: int = 3_750,
) -> tuple[np.ndarray, np.ndarray]:
    """Train on everything before a final validation block."""
    fold = next(
        iter(
            TimeSplit(
                n_folds=1,
                valid_minutes=valid_minutes,
                gap_minutes=gap_minutes,
                warmup_minutes=warmup_minutes,
            ).split(n_minutes)
        )
    )
    return fold


def summarise_folds(scores: list[float]) -> dict:
    arr = np.asarray([s for s in scores if np.isfinite(s)])
    if len(arr) == 0:
        return {"n_folds": 0, "mean": float("nan")}
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "n_folds": int(len(arr)),
        "mean": round(float(arr.mean()), 5),
        "std": round(std, 5),
        "min": round(float(arr.min()), 5),
        "max": round(float(arr.max()), 5),
    }

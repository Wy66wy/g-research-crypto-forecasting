"""A model interface that works at the level of the minute grid.

Tabular models (Ridge, LightGBM) think in long-format rows; the sequence
model thinks in (asset, time, feature) windows. Rather than give them
different interfaces and branch on the type everywhere, every model here
takes the same inputs - the (T, A, F) feature tensor, the grid, and a set of
minute indices - and returns predictions shaped (len(rows), A).

That single shape is what makes the comparison table honest: every model is
scored by literally the same function on literally the same cells.
"""

from __future__ import annotations

import abc
from pathlib import Path

import numpy as np

from ..data import MarketGrid

__all__ = ["BaseModel"]


class BaseModel(abc.ABC):
    name: str = "base"

    def __init__(self, **params) -> None:
        self.params = params
        self.feature_names: list[str] | None = None
        self.is_fitted = False

    @abc.abstractmethod
    def fit(
        self,
        tensor: np.ndarray,
        grid: MarketGrid,
        train_rows: np.ndarray,
        valid_rows: np.ndarray | None = None,
        feature_names: list[str] | None = None,
    ) -> "BaseModel":
        """Fit on the given minutes of the grid."""

    @abc.abstractmethod
    def predict(
        self, tensor: np.ndarray, grid: MarketGrid, rows: np.ndarray
    ) -> np.ndarray:
        """Predict for the given minutes. Returns shape (len(rows), A)."""

    @abc.abstractmethod
    def save(self, path: str | Path) -> None: ...

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> "BaseModel": ...

    def feature_importance(self) -> dict[str, float] | None:
        return None

    def _check_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError(f"{self.name} is not fitted")

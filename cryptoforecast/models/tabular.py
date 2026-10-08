"""Ridge and LightGBM, pooled across assets.

Both are fitted on every (minute, asset) row at once, with ``asset_id`` as a
feature, rather than one model per coin. The small coins have few rows and a
lot of noise, and a pooled model lets them borrow statistical strength from
the large ones.

The parameters are conservative for the usual reason in financial data: the
signal is faint, a flexible model finds patterns in the noise, and those
patterns do not survive out of sample.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np

from ..data import MarketGrid
from ..features import to_long
from .base import BaseModel

logger = logging.getLogger(__name__)

__all__ = ["RidgeModel", "LGBMModel", "DEFAULT_LGB_PARAMS"]


def _scatter(pred: np.ndarray, minutes: np.ndarray, assets: np.ndarray,
             rows: np.ndarray, grid: MarketGrid) -> np.ndarray:
    """Put long-format predictions back onto a (len(rows), A) array."""
    out = np.full((len(rows), grid.n_assets), np.nan, dtype=np.float64)
    pos = {int(r): i for i, r in enumerate(rows)}
    col = {a: j for j, a in enumerate(grid.asset_ids)}
    for p, m, a in zip(pred, minutes, assets):
        out[pos[int(m)], col[int(a)]] = p
    return out


def _all_cells(tensor: np.ndarray, grid: MarketGrid, rows: np.ndarray):
    """Every present cell of the requested minutes, target or not.

    Prediction must not depend on whether a target exists - live, there is
    none - so this ignores the target entirely.
    """
    sub = tensor[rows]
    keep = grid.present[rows]
    t_idx, a_idx = np.nonzero(keep)
    return sub[t_idx, a_idx], np.array(grid.asset_ids)[a_idx], rows[t_idx]


class RidgeModel(BaseModel):
    """Median imputation, standardisation, then Ridge."""

    name = "ridge"

    def __init__(self, alpha: float = 10.0, **params) -> None:
        super().__init__(alpha=alpha, **params)
        self.alpha = alpha
        self.pipeline = None

    def fit(self, tensor, grid, train_rows, valid_rows=None, feature_names=None):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.feature_names = feature_names
        X, y, _, _ = to_long(tensor, grid, train_rows)
        self.pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=self.alpha)),
            ]
        )
        self.pipeline.fit(X, y)
        self.is_fitted = True
        logger.info("ridge fitted on %s rows", f"{len(y):,}")
        return self

    def predict(self, tensor, grid, rows):
        self._check_fitted()
        X, assets, minutes = _all_cells(tensor, grid, rows)
        pred = self.pipeline.predict(X)
        return _scatter(pred, minutes, assets, rows, grid)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "feature_names": self.feature_names,
                         "alpha": self.alpha}, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            blob = pickle.load(f)
        m = cls(alpha=blob["alpha"])
        m.pipeline, m.feature_names, m.is_fitted = blob["pipeline"], blob["feature_names"], True
        return m

    def feature_importance(self):
        coefs = self.pipeline.named_steps["model"].coef_.ravel()
        names = self.feature_names or [f"f{i}" for i in range(len(coefs))]
        return {n: float(c) for n, c in sorted(zip(names, coefs), key=lambda kv: -abs(kv[1]))}


DEFAULT_LGB_PARAMS: dict = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 500,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.7,
    "bagging_freq": 1,
    "lambda_l2": 10.0,
    "max_bin": 127,
    "verbosity": -1,
    "seed": 42,
}


class LGBMModel(BaseModel):
    """Pooled LightGBM, early-stopped on validation correlation."""

    name = "lgbm"

    def __init__(self, n_estimators: int = 1000, early_stopping_rounds: int = 50,
                 **params) -> None:
        merged = {**DEFAULT_LGB_PARAMS, **params}
        super().__init__(**merged)
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.booster = None

    def fit(self, tensor, grid, train_rows, valid_rows=None, feature_names=None):
        import lightgbm as lgb

        self.feature_names = feature_names
        X, y, _, _ = to_long(tensor, grid, train_rows)
        names = feature_names or "auto"
        cat = ["asset_id"] if feature_names and "asset_id" in feature_names else "auto"
        dtrain = lgb.Dataset(X, label=y, feature_name=names, categorical_feature=cat)

        valid_sets, callbacks = [dtrain], [lgb.log_evaluation(0)]
        if valid_rows is not None:
            Xv, yv, _, _ = to_long(tensor, grid, valid_rows)
            valid_sets.append(lgb.Dataset(Xv, label=yv, reference=dtrain))
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))

        self.booster = lgb.train(
            self.params, dtrain, num_boost_round=self.n_estimators,
            valid_sets=valid_sets, callbacks=callbacks,
        )
        self.is_fitted = True
        logger.info("lgbm fitted: %d rounds", self.booster.best_iteration or self.n_estimators)
        return self

    def predict(self, tensor, grid, rows):
        self._check_fitted()
        X, assets, minutes = _all_cells(tensor, grid, rows)
        pred = self.booster.predict(X, num_iteration=self.booster.best_iteration or None)
        return _scatter(pred, minutes, assets, rows, grid)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        path.with_suffix(".meta.json").write_text(
            json.dumps({"feature_names": self.feature_names}), encoding="utf-8")

    @classmethod
    def load(cls, path):
        import lightgbm as lgb

        path = Path(path)
        m = cls()
        m.booster = lgb.Booster(model_file=str(path))
        meta = path.with_suffix(".meta.json")
        if meta.exists():
            m.feature_names = json.loads(meta.read_text(encoding="utf-8"))["feature_names"]
        m.is_fitted = True
        return m

    def feature_importance(self):
        g = self.booster.feature_importance(importance_type="gain")
        names = self.feature_names or self.booster.feature_name()
        return {n: float(v) for n, v in sorted(zip(names, g), key=lambda kv: -kv[1])}

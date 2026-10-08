"""Model registry.

``crossasset`` and ``gru`` are the same network with and without the
cross-asset attention block - the pair exists so attention can be ablated
with everything else held fixed. Both import torch lazily.
"""

from __future__ import annotations

from .base import BaseModel
from .tabular import DEFAULT_LGB_PARAMS, LGBMModel, RidgeModel

__all__ = ["BaseModel", "RidgeModel", "LGBMModel", "DEFAULT_LGB_PARAMS",
           "get_model", "load_model", "MODEL_NAMES"]

MODEL_NAMES = ["ridge", "lgbm", "gru", "crossasset"]


def get_model(name: str, **params) -> BaseModel:
    key = name.lower()
    if key == "ridge":
        return RidgeModel(**params)
    if key == "lgbm":
        return LGBMModel(**params)
    if key in ("gru", "crossasset"):
        from .crossasset import CrossAssetModel

        params.setdefault("use_attention", key == "crossasset")
        return CrossAssetModel(**params)
    raise ValueError(f"unknown model '{name}'; available: {MODEL_NAMES}")


def load_model(name: str, path) -> BaseModel:
    key = name.lower()
    if key == "ridge":
        return RidgeModel.load(path)
    if key == "lgbm":
        return LGBMModel.load(path)
    if key in ("gru", "crossasset"):
        from .crossasset import CrossAssetModel

        return CrossAssetModel.load(path)
    raise ValueError(f"unknown model '{name}'; available: {MODEL_NAMES}")

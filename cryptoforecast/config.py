"""Configuration as dataclasses, loadable from YAML."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .features import FeatureConfig

__all__ = ["DataConfig", "TrainConfig", "ExperimentConfig"]


@dataclass
class DataConfig:
    data_dir: str = "data"
    filename: str = "train.csv"


@dataclass
class TrainConfig:
    model: str = "lgbm"
    model_params: dict[str, Any] = field(default_factory=dict)
    #: Drop cross-asset features, for the ablation
    own_only: bool = False
    n_folds: int = 3
    valid_minutes: int = 10_080          # one week
    gap_minutes: int = 60                # >= the target's 16-minute lookahead
    warmup_minutes: int = 3_750          # until the target's beta has formed
    output_dir: str = "artifacts"
    seed: int = 42


@dataclass
class ExperimentConfig:
    name: str = "baseline"
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    #: Per-family defaults, used when a model is chosen on the command line
    ridge_params: dict[str, Any] = field(default_factory=lambda: {"alpha": 10.0})
    lgbm_params: dict[str, Any] = field(default_factory=lambda: {
        "n_estimators": 1000, "learning_rate": 0.03, "num_leaves": 31,
        "min_child_samples": 500, "feature_fraction": 0.7,
    })
    nn_params: dict[str, Any] = field(default_factory=lambda: {
        "lookback": 30, "hidden": 64, "n_heads": 4, "n_layers": 2,
        "dropout": 0.1, "lr": 1e-3, "batch_size": 256,
        "max_epochs": 12, "patience": 2, "train_stride": 4,
    })

    def model_params_for(self, model: str) -> dict[str, Any]:
        return {
            "ridge": self.ridge_params,
            "lgbm": self.lgbm_params,
            "gru": self.nn_params,
            "crossasset": self.nn_params,
        }[model]

    @classmethod
    def from_yaml(cls, path) -> "ExperimentConfig":
        return cls.from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})

    @classmethod
    def from_dict(cls, raw: dict) -> "ExperimentConfig":
        base = cls()
        cfg = cls(
            name=raw.get("name", "baseline"),
            data=DataConfig(**raw.get("data", {})),
            features=FeatureConfig(**raw.get("features", {})),
            train=TrainConfig(**raw.get("train", {})),
            ridge_params=raw.get("ridge_params", base.ridge_params),
            lgbm_params=raw.get("lgbm_params", base.lgbm_params),
            nn_params=raw.get("nn_params", base.nn_params),
        )
        if not cfg.train.model_params:
            cfg.train.model_params = dict(cfg.model_params_for(cfg.train.model))
        return cfg

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")
        else:
            path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

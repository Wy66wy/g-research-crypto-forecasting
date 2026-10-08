"""Training entry point.

Command line::

    python -m cryptoforecast.train --model lgbm --data-dir data
    python -m cryptoforecast.train --model crossasset --data-dir data
    python -m cryptoforecast.train --model lgbm --own-only
    python -m cryptoforecast.train --ablation          # the full comparison

``--ablation`` runs the experiment the project is built around. Six
configurations, each scored on identical walk-forward folds::

    ridge       own features
    ridge       own + cross-asset features
    lgbm        own features
    lgbm        own + cross-asset features
    gru         per-coin network, no attention
    crossasset  per-coin network + attention across coins

Read in pairs, they answer three separate questions:

* lgbm own vs lgbm own+cross -> how much is cross-asset information worth?
* gru vs crossasset          -> does attention extract it, all else equal?
* lgbm own+cross vs crossasset -> does learning the cross-asset structure
                                  beat engineering it by hand?

Artifacts land in ``artifacts/``: the final model fitted on all usable data,
its config, per-fold scores, and feature importances where available.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .config import ExperimentConfig
from .data import load_grid
from .features import build_feature_tensor
from .metrics import weighted_correlation
from .models import get_model
from .validation import TimeSplit, summarise_folds

logger = logging.getLogger(__name__)

CROSS_PREFIXES = ("mkt_", "rel_", "lead")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def score_predictions(pred: np.ndarray, grid, rows: np.ndarray) -> tuple[float, dict]:
    """The competition metric on the cells that have a target."""
    y = grid.target[rows]
    ok = grid.present[rows] & np.isfinite(y) & np.isfinite(pred)
    t, a = np.nonzero(ok)
    return weighted_correlation(
        y[t, a], pred[t, a], np.array(grid.asset_ids)[a],
        weights=grid.weights, return_components=True,
    )


def select_features(tensor, names, own_only: bool):
    """Drop the cross-asset columns for the own-features ablation."""
    if not own_only:
        return tensor, names
    keep = [i for i, n in enumerate(names) if not n.startswith(CROSS_PREFIXES)]
    return tensor[:, :, keep], [names[i] for i in keep]


def run_cv(config: ExperimentConfig, grid=None, tensor=None, names=None) -> dict:
    """Score one configuration across the walk-forward folds."""
    if grid is None:
        grid = load_grid(config.data.data_dir, config.data.filename)
    if tensor is None:
        tensor, names = build_feature_tensor(grid, config.features)
    tensor, names = select_features(tensor, names, config.train.own_only)

    split = TimeSplit(
        n_folds=config.train.n_folds,
        valid_minutes=config.train.valid_minutes,
        gap_minutes=config.train.gap_minutes,
        warmup_minutes=config.train.warmup_minutes,
    )

    fold_scores, fold_details = [], []
    t0 = time.time()
    for k, (tr, va) in enumerate(split.split(grid.n_minutes)):
        model = get_model(config.train.model, **config.train.model_params)
        model.fit(tensor, grid, tr, va, feature_names=names)
        pred = model.predict(tensor, grid, va)
        score, per_asset = score_predictions(pred, grid, va)
        fold_scores.append(score)
        fold_details.append({"fold": k, "score": score, "per_asset": per_asset})
        logger.info("fold %d score: %+.5f", k, score)

    summary = summarise_folds(fold_scores)
    summary["seconds"] = round(time.time() - t0, 1)
    summary["n_features"] = len(names)
    logger.info(
        "%s%s: %.5f +/- %.5f over %d folds (%.0fs)",
        config.train.model,
        " [own only]" if config.train.own_only else "",
        summary["mean"], summary["std"], summary["n_folds"], summary["seconds"],
    )
    return {"summary": summary, "folds": fold_details, "feature_names": names}


def run_training(config: ExperimentConfig) -> dict:
    """Cross-validate, then fit a final model on all usable minutes and save it."""
    out_dir = Path(config.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid(config.data.data_dir, config.data.filename)
    tensor, names = build_feature_tensor(grid, config.features)

    results = run_cv(config, grid, tensor, names)
    tensor, names = select_features(tensor, names, config.train.own_only)

    # Final model: train on everything with a label, holding out the last
    # validation block only for early stopping
    split = TimeSplit(n_folds=1, valid_minutes=config.train.valid_minutes,
                      gap_minutes=config.train.gap_minutes,
                      warmup_minutes=config.train.warmup_minutes)
    tr, va = next(iter(split.split(grid.n_minutes)))
    final = get_model(config.train.model, **config.train.model_params)
    final.fit(tensor, grid, tr, va, feature_names=names)

    ext = {"lgbm": ".txt", "ridge": ".pkl"}.get(config.train.model, ".pt")
    final.save(out_dir / f"{config.name}_model{ext}")
    config.save(out_dir / f"{config.name}_config.yaml")
    (out_dir / f"{config.name}_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )

    imp = final.feature_importance()
    if imp:
        (out_dir / f"{config.name}_importance.csv").write_text(
            "feature,importance\n" + "\n".join(f"{k},{v:.8f}" for k, v in imp.items()),
            encoding="utf-8",
        )
        logger.info("top features: %s", ", ".join(list(imp)[:8]))

    logger.info("artifacts written to %s", out_dir)
    return results


def run_ablation(config: ExperimentConfig) -> dict:
    """The six-way comparison, on shared folds and a shared feature tensor."""
    grid = load_grid(config.data.data_dir, config.data.filename)
    tensor, names = build_feature_tensor(grid, config.features)

    plan = [
        ("ridge", True), ("ridge", False),
        ("lgbm", True), ("lgbm", False),
        ("gru", False), ("crossasset", False),
    ]
    rows = []
    for model, own_only in plan:
        cfg = ExperimentConfig.from_dict(config.to_dict())
        cfg.train.model = model
        cfg.train.own_only = own_only
        cfg.train.model_params = dict(config.model_params_for(model))
        res = run_cv(cfg, grid, tensor, names)
        s = res["summary"]
        rows.append({
            "model": model,
            "features": "own" if own_only else "own+cross",
            **s,
        })

    print("\n" + "=" * 74)
    print(f"{'model':<12}{'features':<11}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}{'sec':>9}")
    print("-" * 74)
    for r in rows:
        print(f"{r['model']:<12}{r['features']:<11}{r['mean']:>10.5f}{r['std']:>10.5f}"
              f"{r['min']:>10.5f}{r['max']:>10.5f}{r['seconds']:>9.0f}")
    print("=" * 74 + "\n")

    out = Path(config.train.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(rows, indent=2, default=float),
                                       encoding="utf-8")
    return {"rows": rows}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Training for the G-Research crypto forecasting competition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--name", type=str, default=None)
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--model", type=str, default=None,
                   choices=["ridge", "lgbm", "gru", "crossasset"])
    p.add_argument("--own-only", action="store_true",
                   help="drop cross-asset features (ablation)")
    p.add_argument("--n-folds", type=int, default=None)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="override the neural network's epoch budget")
    p.add_argument("--ablation", action="store_true",
                   help="run the six-way comparison instead of one model")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)

    config = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    if args.data_dir:
        config.data.data_dir = args.data_dir
    if args.model:
        config.train.model = args.model
        config.train.model_params = dict(config.model_params_for(args.model))
    if args.own_only:
        config.train.own_only = True
    if args.n_folds:
        config.train.n_folds = args.n_folds
    if args.max_epochs:
        config.nn_params["max_epochs"] = args.max_epochs
        if config.train.model in ("gru", "crossasset"):
            config.train.model_params["max_epochs"] = args.max_epochs
    if args.output_dir:
        config.train.output_dir = args.output_dir
    config.name = args.name or (
        f"{config.train.model}{'_own' if config.train.own_only else ''}"
    )

    if args.ablation:
        run_ablation(config)
    else:
        run_training(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

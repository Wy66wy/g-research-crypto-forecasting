#!/usr/bin/env python3
"""Compare the live inference path with batch prediction, row by row.

``run_replay.py`` reports a score, and a score is a poor diagnostic: over a
few hundred minutes it is dominated by noise, so a replay that scores well
above cross-validation could be luck or could be a leak, and the number alone
cannot say which.

This script removes the ambiguity. It fixes one set of minutes and predicts
them twice:

* **batch** - the whole history on one grid, one feature tensor, one call to
  ``model.predict``, exactly as during training;
* **live** - one minute at a time through :class:`CryptoPredictor`, which
  rebuilds the grid from a rolling buffer at every step.

If the two paths implement the same thing, the predictions are identical up
to floating point. If they differ, the live path computes features
differently from the training path - and then it does not matter what the
replay score was.

The first ~90 minutes of the live run are excluded: the buffer is still
filling, so its longest features are legitimately incomplete.

Usage::

    python scripts/check_consistency.py --name lgbm
    python scripts/check_consistency.py --name crossasset --minutes 400
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptoforecast.data import load_asset_weights, load_long, to_grid  # noqa: E402
from cryptoforecast.inference import CryptoPredictor  # noqa: E402
from cryptoforecast.metrics import weighted_correlation  # noqa: E402

WARMUP = 90
CORR_THRESHOLD = 0.9999


def main() -> int:
    p = argparse.ArgumentParser(description="Batch vs live prediction, row by row")
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--name", type=str, default="lgbm")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--minutes", type=int, default=400)
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    weights = load_asset_weights(args.data_dir)
    df = load_long(args.data_dir)
    ts_all = np.sort(df["timestamp"].unique().to_numpy())
    start = int(ts_all[-args.minutes])

    # ------------------------------------------------------------- batch
    predictor = CryptoPredictor.from_artifacts(args.artifacts, args.name, weights)
    grid = to_grid(df, weights)
    tensor, _ = predictor._features(grid)
    rows = np.arange(grid.n_minutes - args.minutes, grid.n_minutes)
    batch = predictor.model.predict(tensor, grid, rows)          # (n, A)

    # ------------------------------------------------------------- live
    live_pred = CryptoPredictor.from_artifacts(args.artifacts, args.name, weights)
    # Seed the buffer with the history just before the window, as a real
    # submission would have after running for a while
    seed = df.filter((pl.col("timestamp") < start)
                     & (pl.col("timestamp") >= start - 240 * 60))
    for _, minute in seed.group_by("timestamp", maintain_order=True):
        live_pred._buffer.append(minute.drop("Target", strict=False))

    window = df.filter(pl.col("timestamp") >= start).sort(["timestamp", "Asset_ID"])
    col = {a: j for j, a in enumerate(grid.asset_ids)}
    row_of = {int(t): i for i, t in enumerate(grid.timestamps[rows])}
    live = np.full_like(batch, np.nan)

    for _, minute in window.group_by("timestamp", maintain_order=True):
        out = live_pred.predict(minute.drop("Target", strict=False))
        i = row_of[int(minute["timestamp"][0])]
        for a, v in zip(minute["Asset_ID"].to_list(), out):
            live[i, col[int(a)]] = v

    # ------------------------------------------------------------- compare
    keep = np.zeros_like(batch, dtype=bool)
    keep[WARMUP:] = True
    ok = keep & np.isfinite(batch) & np.isfinite(live)
    b, l = batch[ok], live[ok]
    diff = np.abs(b - l)
    corr = float(np.corrcoef(b, l)[0, 1])

    y = grid.target[rows]
    sc = np.isfinite(y) & ok
    t_i, a_i = np.nonzero(sc)
    assets = np.array(grid.asset_ids)[a_i]
    s_batch = weighted_correlation(y[t_i, a_i], batch[t_i, a_i], assets, weights)
    s_live = weighted_correlation(y[t_i, a_i], live[t_i, a_i], assets, weights)

    print("=" * 62)
    print(f"  cells compared       : {ok.sum():,}")
    print(f"  max abs difference   : {diff.max():.3e}")
    print(f"  mean abs difference  : {diff.mean():.3e}")
    print(f"  prediction corr      : {corr:.8f}")
    print(f"  batch score          : {s_batch:+.5f}")
    print(f"  live score           : {s_live:+.5f}")
    print("=" * 62)
    if corr >= CORR_THRESHOLD:
        print("  PASS - the live path reproduces batch prediction")
        print("  (so any gap between this score and cross-validation is the")
        print("  noise of a short window, not a leak)")
        return 0
    print("  FAIL - the live path computes something different from training")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

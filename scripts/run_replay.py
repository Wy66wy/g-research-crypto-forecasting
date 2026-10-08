#!/usr/bin/env python3
"""Replay the predictor minute by minute, as the evaluation API will.

Training scores predictions computed from one big feature tensor. The live
path rebuilds features from a rolling buffer, one minute at a time. Those are
different code paths, and this script checks that they agree.

Usage::

    python scripts/run_replay.py --name lgbm
    python scripts/run_replay.py --name crossasset --minutes 1500
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptoforecast.data import load_asset_weights, load_long  # noqa: E402
from cryptoforecast.inference import CryptoPredictor, replay  # noqa: E402
from cryptoforecast.train import setup_logging  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Minute-by-minute replay")
    p.add_argument("--artifacts", type=str, default="artifacts")
    p.add_argument("--name", type=str, default="lgbm")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--minutes", type=int, default=1_000)
    args = p.parse_args()
    setup_logging()
    logging.getLogger("cryptoforecast.data").setLevel(logging.WARNING)
    logging.getLogger("cryptoforecast.features").setLevel(logging.WARNING)

    weights = load_asset_weights(args.data_dir)
    predictor = CryptoPredictor.from_artifacts(args.artifacts, args.name, weights)

    df = load_long(args.data_dir)
    ts = np.sort(df["timestamp"].unique().to_numpy())
    start = ts[-args.minutes]
    tail = df.filter(df["timestamp"] >= int(start))

    result = replay(predictor, tail)
    print("\n" + "=" * 56)
    print(f"  replay score  : {result['score']:+.5f}")
    print(f"  minutes       : {result['n_minutes']:,}")
    print("=" * 56)
    print("  Compare with the last fold's validation score in")
    print(f"  {args.artifacts}/{args.name}_results.json. A large gap means")
    print("  the live feature path differs from the training one.\n")

    out = Path(args.artifacts) / f"{args.name}_replay.json"
    out.write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

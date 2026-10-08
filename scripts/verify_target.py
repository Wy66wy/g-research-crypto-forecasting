#!/usr/bin/env python3
"""Check the target reimplementation against the provided Target column.

On the synthetic data this compares two independent implementations. On the
real competition data it does something more valuable: it tells you whether
:mod:`cryptoforecast.target` reproduces the organisers' number. If the
correlation is not essentially 1, something about the definition - the t+1
offset, the uncentred beta, the treatment of missing minutes - has been
misread, and every leakage argument built on it is suspect.

Usage::

    python scripts/verify_target.py --data-dir data
    python scripts/verify_target.py --data-dir /path/to/real --days 30
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptoforecast.data import load_grid  # noqa: E402
from cryptoforecast.target import compute_target  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description="Verify the target reimplementation")
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--days", type=int, default=None,
                   help="use only the last N days (the real file is large)")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING)

    grid = load_grid(args.data_dir)
    if args.days:
        n = args.days * 1440
        grid = grid.slice(max(0, grid.n_minutes - n), grid.n_minutes)

    if grid.target is None:
        print("no Target column in this data; nothing to verify against")
        return 1

    mine = compute_target(grid)
    ref = grid.target
    both = np.isfinite(mine) & np.isfinite(ref)
    only_mine = int((np.isfinite(mine) & ~np.isfinite(ref)).sum())
    only_ref = int((~np.isfinite(mine) & np.isfinite(ref)).sum())

    diff = np.abs(mine[both] - ref[both])
    corr = float(np.corrcoef(mine[both], ref[both])[0, 1])

    print("=" * 60)
    print(f"  cells compared         : {both.sum():,}")
    print(f"  finite only in mine    : {only_mine:,}")
    print(f"  finite only in ref     : {only_ref:,}")
    print(f"  max abs difference     : {diff.max():.3e}")
    print(f"  median abs difference  : {np.median(diff):.3e}")
    print(f"  correlation            : {corr:.8f}")
    print("=" * 60)

    if corr > 0.9999:
        print("  MATCH - the reimplementation reproduces the provided target")
        return 0
    if corr > 0.99:
        print("  CLOSE - small disagreement, likely in missing-minute handling;")
        print("  try --ffill-limit or check the per-asset breakdown")
        return 0
    print("  MISMATCH - the target definition has been misread somewhere")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

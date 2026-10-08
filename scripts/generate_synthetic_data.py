#!/usr/bin/env python3
"""Generate synthetic minute-level data matching the competition's structure.

**Why this exists.** The real dataset is 3GB and needs a Kaggle account.
"Does the code work?" should be answerable in seconds without either.

**What is reproduced, and why each property is there**

* **A strong common factor.** Crypto assets are highly correlated - most of
  any coin's minute-to-minute move is "the market". That is the reason the
  target is residualised at all, so the synthetic data has to have it or the
  residualisation does nothing. The generator prints the resulting mean
  pairwise correlation so this can be checked rather than assumed.
* **Volatility clustering** on the market factor, with fat-tailed shocks.
* **Missing minutes**, more of them for small coins, plus occasional
  multi-hour outages. This is what makes the minute-grid reindexing in
  :mod:`cryptoforecast.data` necessary, so it must exist to be tested.
* **Two planted, realistic sources of predictability** in the residual
  return, chosen so that they discriminate between model families:

  1. *Residual mean reversion.* Each coin's idiosyncratic price is an
     Ornstein-Uhlenbeck process, so a coin that has drifted away from the
     market tends to drift back. Visible from the coin's own history, so any
     model can find it.
  2. *Lead-lag.* Altcoins respond to the market factor with a delay of a few
     minutes, while Bitcoin and Ethereum respond immediately. At minute ``t``
     the market move of the last few minutes is already visible in Bitcoin's
     price but not yet fully in, say, Dogecoin's - so Dogecoin's next
     residual move is partly predictable **from Bitcoin**. A model that sees
     only one coin at a time cannot find this. A model that sees the other
     coins can.

  Signal 2 is the real test of the cross-asset neural network: if it cannot
  beat a per-asset model here, where the structure it is designed for
  genuinely exists, it has no business being used on the real data.

**The target is computed by an independent implementation.** The function
:func:`reference_target` below shares no code with
:mod:`cryptoforecast.target`: it uses explicit sliding windows instead of
cumulative sums and a per-minute loop for the market return. The tests check
that the two agree, which is a genuine cross-check rather than a function
compared against itself.

**How this differs from reality.** The planted signals are much stronger than
anything in real crypto data, so scores here are optimistic. The data exists
to verify correctness, not to estimate achievable performance.

Usage::

    python scripts/generate_synthetic_data.py --output-dir data
    python scripts/generate_synthetic_data.py --n-minutes 200000 --signal-strength 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cryptoforecast.constants import (  # noqa: E402
    ASSET_DETAILS,
    ASSET_IDS,
    BETA_WINDOW,
    TARGET_HORIZON,
    TARGET_OFFSET,
)

#: Response delay to the market factor, in minutes. Large caps react at once;
#: smaller coins lag. Bitcoin (1) and Ethereum (6) lead.
DELAYS: dict[int, int] = {
    0: 1, 1: 0, 2: 2, 3: 2, 4: 3, 5: 3, 6: 0,
    7: 3, 8: 4, 9: 2, 10: 4, 11: 3, 12: 3, 13: 4,
}

#: Fraction of minutes in which each coin trades
TRADE_PROB: dict[int, float] = {
    0: 0.97, 1: 0.999, 2: 0.93, 3: 0.95, 4: 0.92, 5: 0.90, 6: 0.998,
    7: 0.91, 8: 0.85, 9: 0.96, 10: 0.80, 11: 0.88, 12: 0.90, 13: 0.89,
}

START_TS = 1_609_459_200   # 2021-01-01 00:00:00 UTC


def simulate_market_factor(n: int, rng: np.random.Generator) -> np.ndarray:
    """Minute-level market returns with GARCH-style volatility and t shocks."""
    # Unconditional per-minute volatility is sqrt(omega / (1 - alpha - beta)),
    # about 0.12% here. It has to be large relative to each coin's
    # idiosyncratic noise, or the coins barely co-move. An earlier version had
    # omega 7.5x smaller, and the mean pairwise 15-minute correlation came out
    # at 0.16 - far below the 0.5-0.8 real crypto shows, which left the
    # residualisation with almost nothing to remove.
    omega, alpha, beta, dof = 1.5e-8, 0.06, 0.93, 4.0
    var = omega / (1 - alpha - beta)
    scale = np.sqrt(dof / (dof - 2))
    f = np.empty(n)
    for t in range(n):
        if t > 0:
            var = omega + alpha * f[t - 1] ** 2 + beta * var
        f[t] = np.sqrt(var) * rng.standard_t(dof) / scale
    return f


def reference_target(
    close: np.ndarray, present: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """An independent implementation of the official target.

    Deliberately written differently from cryptoforecast.target: explicit
    sliding windows for the rolling means, and a per-row loop for the market
    return. If both implementations agree, both are probably right.
    """
    T, A = close.shape

    # Forward-fill each column with a simple loop
    filled = close.astype(np.float64).copy()
    for j in range(A):
        last = np.nan
        for t in range(T):
            if np.isfinite(filled[t, j]):
                last = filled[t, j]
            else:
                filled[t, j] = last

    lead = TARGET_OFFSET + TARGET_HORIZON
    R = np.full((T, A), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        R[: T - lead] = np.log(filled[lead:] / filled[TARGET_OFFSET : T - TARGET_HORIZON])
    R[~np.isfinite(R)] = np.nan

    M = np.full(T, np.nan)
    for t in range(T):
        row = R[t]
        ok = np.isfinite(row)
        if ok.any():
            M[t] = np.sum(row[ok] * weights[ok]) / np.sum(weights[ok])

    def roll_mean(x: np.ndarray) -> np.ndarray:
        out = np.full(len(x), np.nan)
        if len(x) >= BETA_WINDOW:
            win = sliding_window_view(x, BETA_WINDOW)
            means = win.mean(axis=1)   # any nan in the window -> nan
            out[BETA_WINDOW - 1 :] = means
        return out

    den = roll_mean(M * M)
    target = np.full((T, A), np.nan)
    for j in range(A):
        num = roll_mean(R[:, j] * M)
        with np.errstate(divide="ignore", invalid="ignore"):
            b = num / den
        b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
        target[:, j] = R[:, j] - b * M
    target[~present] = np.nan
    return target


def generate(
    n_minutes: int = 60_000,
    seed: int = 42,
    signal_strength: float = 1.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the synthetic train and asset_details frames.

    Parameters
    ----------
    signal_strength : float
        Multiplier on both planted effects. 1.0 is deliberately generous; 0.3
        is closer to how faint real effects are.
    """
    rng = np.random.default_rng(seed)
    A = len(ASSET_IDS)
    f = simulate_market_factor(n_minutes, rng)

    log_price = np.zeros((n_minutes, A))
    start_price = rng.uniform(0.5, 50_000, A)

    for j, a in enumerate(ASSET_IDS):
        beta_a = rng.uniform(0.8, 1.4)
        d = DELAYS[a]
        # Lead-lag: respond to the market factor d minutes late. Blending the
        # delayed and contemporaneous response keeps every coin correlated
        # with the market while leaving the lagged part predictable.
        lag_share = min(1.0, 0.3 * signal_strength) if d > 0 else 0.0
        delayed = np.concatenate([np.zeros(d), f[: n_minutes - d]]) if d > 0 else f
        market_part = beta_a * ((1 - lag_share) * f + lag_share * delayed)

        # Residual mean reversion: an Ornstein-Uhlenbeck idiosyncratic price.
        # phi < 1 pulls the deviation back toward zero; the half-life is
        # log(0.5)/log(phi) minutes.
        phi = 1.0 - 0.02 * signal_strength
        sigma = rng.uniform(0.0006, 0.0012)
        x = np.zeros(n_minutes)
        for t in range(1, n_minutes):
            x[t] = phi * x[t - 1] + sigma * rng.standard_normal()
        idio = np.diff(x, prepend=0.0)

        log_price[:, j] = np.log(start_price[j]) + np.cumsum(market_part + idio)

    close = np.exp(log_price)

    # Missingness: per-minute dropout, plus occasional multi-hour outages
    present = np.zeros((n_minutes, A), dtype=bool)
    for j, a in enumerate(ASSET_IDS):
        present[:, j] = rng.random(n_minutes) < TRADE_PROB[a]
        for _ in range(int(rng.integers(0, 4))):
            s = int(rng.integers(0, n_minutes))
            present[s : s + int(rng.integers(60, 600)), j] = False

    observed_close = np.where(present, close, np.nan)
    weights = np.array([ASSET_DETAILS[a][1] for a in ASSET_IDS])
    target = reference_target(observed_close, present, weights)

    # OHLCV around each minute's close
    prev = np.vstack([close[:1], close[:-1]])
    spread = np.abs(np.log(close / prev)) + 0.0003
    high = np.maximum(close, prev) * np.exp(rng.uniform(0, 1, close.shape) * spread)
    low = np.minimum(close, prev) * np.exp(-rng.uniform(0, 1, close.shape) * spread)
    vwap = (high + low + close) / 3.0
    activity = 1.0 + 200.0 * np.abs(np.log(close / prev))
    volume = rng.gamma(2.0, 5.0, close.shape) * activity
    count = np.maximum(1, rng.poisson(30 * activity))

    rows_t, rows_a = np.nonzero(present)
    df = pl.DataFrame(
        {
            "timestamp": START_TS + rows_t.astype(np.int64) * 60,
            "Asset_ID": np.array(ASSET_IDS)[rows_a].astype(np.int64),
            "Count": count[rows_t, rows_a].astype(np.float64),
            "Open": prev[rows_t, rows_a],
            "High": high[rows_t, rows_a],
            "Low": low[rows_t, rows_a],
            "Close": close[rows_t, rows_a],
            "Volume": volume[rows_t, rows_a],
            "VWAP": vwap[rows_t, rows_a],
            "Target": target[rows_t, rows_a],
        }
    ).sort(["timestamp", "Asset_ID"])

    details = pl.DataFrame(
        {
            "Asset_ID": ASSET_IDS,
            "Weight": [ASSET_DETAILS[a][1] for a in ASSET_IDS],
            "Asset_Name": [ASSET_DETAILS[a][0] for a in ASSET_IDS],
        }
    )
    return df, details


def main() -> int:
    p = argparse.ArgumentParser(
        description="Generate synthetic G-Research crypto data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=str, default="data")
    p.add_argument("--n-minutes", type=int, default=60_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--signal-strength", type=float, default=1.0)
    args = p.parse_args()

    print(f"generating {args.n_minutes:,} minutes x {len(ASSET_IDS)} assets "
          f"(~{args.n_minutes / 1440:.0f} days)")
    df, details = generate(args.n_minutes, args.seed, args.signal_strength)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.write_csv(out / "train.csv")
    details.write_csv(out / "asset_details.csv")

    # numpy nan becomes a float NaN in polars, not a null, so drop_nulls()
    # would keep them and the std would come out as nan. Filter on finiteness.
    tgt = df["Target"].to_numpy()
    tgt = tgt[np.isfinite(tgt)]
    coverage = df.height / (args.n_minutes * len(ASSET_IDS))
    print(f"{df.height:,} rows -> {out / 'train.csv'}")
    print(f"  grid coverage      : {coverage:.1%}  (rest are missing minutes)")
    print(f"  target std         : {tgt.std():.5f}")
    print(f"  target finite      : {len(tgt):,} of {df.height:,}")

    # Mean pairwise correlation of non-overlapping 15-minute returns. Real
    # crypto sits around 0.5-0.8; far below that, the residualised target
    # stops meaning what it is supposed to.
    piv = (df.pivot(on="Asset_ID", index="timestamp", values="Close")
             .sort("timestamp").drop("timestamp").to_numpy().astype(float))
    idx = np.where(np.isfinite(piv), np.arange(len(piv))[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    piv = piv[idx, np.arange(piv.shape[1])[None, :]]
    r15 = np.log(piv[15::15] / piv[:-15:15])
    ok = np.isfinite(r15).all(axis=1)
    c = np.corrcoef(r15[ok].T)
    pair = c[~np.eye(len(c), dtype=bool)].mean()
    print(f"  mean pairwise corr : {pair:.3f}  (15-minute returns; real crypto ~0.5-0.8)")
    print()
    print("next: python -m cryptoforecast.train --model ridge --data-dir", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

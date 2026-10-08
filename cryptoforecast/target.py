"""Reimplementing the competition target from raw prices.

The provided ``Target`` column is a black box until you can rebuild it, and
rebuilding it is the most reliable way to learn exactly where the leakage
boundary sits. The organisers published the method::

    R_a(t)  = log( Close_a(t+16) / Close_a(t+1) )         15-minute return
    M(t)    = sum_a w_a R_a(t) / sum_a w_a                weighted market return
    beta_a  = mean(R_a * M) / mean(M^2)   over a trailing 3750-minute window
    Target_a(t) = R_a(t) - beta_a(t) * M(t)

Three things are worth stating explicitly, because each one is easy to get
wrong and each changes the answer:

1. **The return starts at t+1, not t.** It is the move from one minute ahead
   to sixteen minutes ahead. A model is never asked to predict the very next
   minute.
2. **The beta regression has no intercept.** It is ``mean(R*M) / mean(M^2)``,
   not ``cov(R, M) / var(M)``. On 15-minute returns the means are tiny and
   the two are close, but they are not equal, and matching the official
   number requires the uncentred form.
3. **Every asset's target depends on every other asset's future.** ``M(t)``
   is built from all fourteen coins' forward returns. So the target for, say,
   Dogecoin at minute ``t`` encodes how Bitcoin moved between ``t+1`` and
   ``t+16``. That is why no market-level quantity computed from forward
   prices can ever be a feature - and also why a model that sees all assets
   at once has a structural reason to exist.

:func:`compute_target` is checked against an independent loop-based
implementation in the tests, and ``scripts/verify_target.py`` compares it with
the provided ``Target`` column on real data.
"""

from __future__ import annotations

import logging

import numpy as np

from .constants import BETA_WINDOW, TARGET_HORIZON, TARGET_OFFSET
from .data import MarketGrid

logger = logging.getLogger(__name__)

__all__ = [
    "forward_fill",
    "forward_log_return",
    "market_return",
    "rolling_mean_full",
    "residualise",
    "compute_target",
]


def forward_fill(x: np.ndarray, limit: int | None = None) -> np.ndarray:
    """Forward-fill nans down each column of a (T, A) array.

    Prices are forward-filled before computing returns: a coin that did not
    trade in a minute is assumed to still be worth its last traded price. With
    ``limit`` set, a gap longer than that stays nan instead of carrying a stale
    price across hours.

    Vectorised with a running maximum over "index of the last valid row". A
    plain Python loop over cells is the obvious way to write this and it is far
    too slow here: the full dataset is about 27 million cells.
    """
    x = np.asarray(x, dtype=np.float64)
    T, A = x.shape
    valid = np.isfinite(x)

    idx = np.where(valid, np.arange(T)[:, None], 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = x[idx, np.arange(A)[None, :]]

    # Rows before an asset's first observation have nothing to carry forward
    seen = np.maximum.accumulate(valid, axis=0)
    out[~seen] = np.nan

    if limit is not None:
        gap = np.arange(T)[:, None] - idx
        out[(gap > limit) & ~valid] = np.nan
    return out


def forward_log_return(
    close: np.ndarray,
    horizon: int = TARGET_HORIZON,
    offset: int = TARGET_OFFSET,
) -> np.ndarray:
    """``log(close[t + offset + horizon] / close[t + offset])`` per column.

    The last ``offset + horizon`` rows have no future to look at and are nan.
    """
    close = np.asarray(close, dtype=np.float64)
    T = close.shape[0]
    out = np.full_like(close, np.nan)
    lead = offset + horizon
    if T > lead:
        with np.errstate(divide="ignore", invalid="ignore"):
            out[: T - lead] = np.log(close[lead:] / close[offset : T - horizon])
    out[~np.isfinite(out)] = np.nan
    return out


def market_return(returns: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted mean return across the assets that have one at each minute.

    Weights are renormalised over the assets present, so a coin that is
    missing at minute ``t`` simply drops out of that minute's market return
    rather than dragging it towards zero.
    """
    r = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)[None, :]
    valid = np.isfinite(r)
    num = np.where(valid, r * w, 0.0).sum(axis=1)
    den = np.where(valid, w, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = num / den
    m[den <= 0] = np.nan
    return m


def rolling_mean_full(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean that requires every value in the window to be finite.

    Matches pandas' ``rolling(window).mean()`` with its default
    ``min_periods=window``: one nan anywhere in the window makes the result
    nan. That strictness is part of the official definition - the published
    code then turns those nans into a beta of zero - so it is reproduced
    exactly rather than "improved".
    """
    x = np.asarray(x, dtype=np.float64)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    T = x.shape[0]
    out = np.full_like(x, np.nan)
    if T < window:
        return out[:, 0] if squeeze else out

    finite = np.isfinite(x)
    vals = np.where(finite, x, 0.0)
    csum = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(vals, axis=0)])
    ccnt = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(finite, axis=0)])

    sums = csum[window:] - csum[:-window]
    cnts = ccnt[window:] - ccnt[:-window]
    means = sums / window
    means[cnts < window] = np.nan
    out[window - 1 :] = means
    return out[:, 0] if squeeze else out


def residualise(
    returns: np.ndarray,
    market: np.ndarray,
    window: int = BETA_WINDOW,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove each asset's rolling market exposure. Returns (residual, beta).

    A direct transcription of the organisers' ``ResidualizeMarket``: an
    uncentred regression coefficient over a trailing window, with any
    numerical failure (an incomplete window, a zero denominator) replaced by
    a beta of zero.
    """
    r = np.asarray(returns, dtype=np.float64)
    m = np.asarray(market, dtype=np.float64)

    num = rolling_mean_full(r * m[:, None], window)
    den = rolling_mean_full(m * m, window)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        beta = num / den
    beta = np.nan_to_num(beta, nan=0.0, posinf=0.0, neginf=0.0)

    residual = r - beta * m[:, None]
    return residual, beta


def compute_target(
    grid: MarketGrid,
    window: int = BETA_WINDOW,
    ffill_limit: int | None = None,
) -> np.ndarray:
    """Rebuild the residualised target on the grid, shape (T, A).

    Cells where the asset has no observation are set to nan, matching the
    provided column, which only exists on rows the asset actually traded.
    """
    close = forward_fill(grid.fields["Close"], limit=ffill_limit)
    R = forward_log_return(close)
    M = market_return(R, grid.weight_vector())
    target, _ = residualise(R, M, window)
    target[~grid.present] = np.nan
    return target.astype(np.float32)

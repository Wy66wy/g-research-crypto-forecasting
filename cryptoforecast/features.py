"""Feature engineering on the minute grid.

Two families, kept separate on purpose:

* **Own-asset features** use nothing but the coin's own price and volume
  history.
* **Cross-asset features** use the other coins too: the market's recent
  return, the coin's return relative to it, and the recent move of the
  leading coins.

The split exists for an ablation, not for tidiness. The target is defined
relative to the market, so the obvious question is how much of the
predictable part depends on information from *other* assets - and whether a
neural network that sees all fourteen coins at once extracts it better than
a gradient-boosted model handed the same information as engineered columns.
Comparing a GBDT on own features, a GBDT on own + cross features, and the
cross-asset network answers that cleanly. Giving the GBDT the cross-asset
features is what makes the comparison fair: without them the network would
win by default, for having been shown more.

**Every feature at minute t uses prices up to and including t.** Returns are
computed backwards (``log(close[t] / close[t-k])``), on the forward-filled
grid, so "k steps back" is genuinely k minutes. Nothing here may touch the
target's window from t+1 to t+16.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .constants import ASSET_IDS
from .data import MarketGrid
from .target import forward_fill, market_return

logger = logging.getLogger(__name__)

__all__ = [
    "FeatureConfig",
    "backward_log_return",
    "rolling_std",
    "build_feature_tensor",
    "to_long",
]

#: Coins treated as market leaders for the lead-lag features: Bitcoin and
#: Ethereum, the two largest by weight
LEADER_ASSETS: list[int] = [1, 6]


@dataclass
class FeatureConfig:
    """Feature settings.

    Attributes
    ----------
    return_windows : list[int]
        Backward return horizons in minutes. 15 matches the target's horizon.
    vol_windows : list[int]
        Windows for realised volatility.
    leader_windows : list[int]
        Horizons for the leaders' recent returns. Short on purpose: a lead-lag
        effect lives in the last few minutes, if it exists at all.
    use_cross_asset : bool
        Include the market and leader features. Switch off for the own-asset
        ablation.
    ffill_limit : int
        Longest gap, in minutes, over which a price is carried forward. Beyond
        that the coin is treated as missing rather than frozen at a stale price.
    """

    return_windows: list[int] = field(default_factory=lambda: [1, 5, 15, 60])
    vol_windows: list[int] = field(default_factory=lambda: [15, 60])
    leader_windows: list[int] = field(default_factory=lambda: [1, 3, 5])
    use_cross_asset: bool = True
    ffill_limit: int = 30


def backward_log_return(close: np.ndarray, k: int) -> np.ndarray:
    """``log(close[t] / close[t-k])`` per column. The first k rows are nan."""
    out = np.full_like(close, np.nan, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[k:] = np.log(close[k:] / close[:-k])
    out[~np.isfinite(out)] = np.nan
    return out


def rolling_std(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing standard deviation per column, tolerant of scattered nans."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    v = np.where(finite, x, 0.0)

    def csum(a):
        return np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])

    s1, s2, n = csum(v), csum(v * v), csum(finite.astype(np.float64))
    out = np.full_like(x, np.nan)
    if x.shape[0] < window:
        return out
    S1 = s1[window:] - s1[:-window]
    S2 = s2[window:] - s2[:-window]
    N = n[window:] - n[:-window]
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = S1 / N
        var = (S2 - N * mean**2) / np.maximum(N - 1, 1)
    var[N < max(3, window // 3)] = np.nan
    out[window - 1 :] = np.sqrt(np.maximum(var, 0.0))
    return out


def build_feature_tensor(
    grid: MarketGrid,
    config: FeatureConfig | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build every feature on the grid.

    Returns
    -------
    (ndarray, list[str])
        A float32 tensor of shape (T, A, F), and the F feature names.
    """
    config = config or FeatureConfig()
    T, A = grid.n_minutes, grid.n_assets

    close = forward_fill(grid.fields["Close"], limit=config.ffill_limit)
    high = grid.fields["High"].astype(np.float64)
    low = grid.fields["Low"].astype(np.float64)
    vwap = grid.fields["VWAP"].astype(np.float64)
    volume = grid.fields["Volume"].astype(np.float64)
    count = grid.fields["Count"].astype(np.float64)

    feats: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ own asset
    rets = {}
    for k in config.return_windows:
        rets[k] = backward_log_return(close, k)
        feats[f"ret_{k}"] = rets[k]

    r1 = rets.get(1, backward_log_return(close, 1))
    for w in config.vol_windows:
        feats[f"vol_{w}"] = rolling_std(r1, w)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Candle shape: how wide the minute was, and where it closed within it
        feats["log_range"] = np.log(high / low)
        rng_ = high - low
        feats["close_pos"] = np.where(rng_ > 0, (grid.fields["Close"] - low) / rng_, 0.5)
        # Distance from VWAP: a close well above the minute's average price
        # often means late buying pressure
        feats["vwap_dev"] = np.log(grid.fields["Close"] / vwap)
        feats["log_volume"] = np.log1p(volume)
        feats["log_count"] = np.log1p(count)
        # Volume relative to its own recent level - a spike matters more than
        # the absolute number, which varies by orders of magnitude across coins
        lv = np.log1p(volume)
        mu = np.where(np.isfinite(lv), lv, 0.0)
        feats["volume_z"] = lv - _rolling_mean(mu, 60)

    feats["is_present"] = grid.present.astype(np.float64)

    # ------------------------------------------------------------ cross asset
    if config.use_cross_asset:
        w = grid.weight_vector()
        for k in config.return_windows:
            m = market_return(rets[k], w)               # past returns only
            feats[f"mkt_ret_{k}"] = np.repeat(m[:, None], A, axis=1)
            feats[f"rel_ret_{k}"] = rets[k] - m[:, None]

        col = {a: j for j, a in enumerate(grid.asset_ids)}
        for leader in LEADER_ASSETS:
            if leader not in col:
                continue
            lc = close[:, col[leader]][:, None]
            for k in config.leader_windows:
                lr = backward_log_return(lc, k)[:, 0]
                feats[f"lead{leader}_ret_{k}"] = np.repeat(lr[:, None], A, axis=1)

    # Asset identity, so a single pooled model can learn per-coin behaviour
    feats["asset_id"] = np.repeat(
        np.array(grid.asset_ids, dtype=np.float64)[None, :], T, axis=0
    )

    names = list(feats)
    tensor = np.stack([feats[n] for n in names], axis=2).astype(np.float32)
    tensor[~np.isfinite(tensor)] = np.nan
    logger.info("feature tensor: %s, %d features", tensor.shape, len(names))
    return tensor, names


def _rolling_mean(x: np.ndarray, window: int) -> np.ndarray:
    c = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
    out = np.full_like(x, np.nan)
    if x.shape[0] >= window:
        out[window - 1 :] = (c[window:] - c[:-window]) / window
    return out


def to_long(
    tensor: np.ndarray,
    grid: MarketGrid,
    rows: np.ndarray,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Flatten selected minutes of the (T, A, F) tensor to long format.

    Only cells where the asset actually traded and the target is finite are
    kept: those are the only rows the metric ever scores.

    Returns
    -------
    (X, y, asset_ids, minute_index)
    """
    target = grid.target if target is None else target
    sub = tensor[rows]                          # (n, A, F)
    y = target[rows]                            # (n, A)
    keep = grid.present[rows] & np.isfinite(y)
    t_idx, a_idx = np.nonzero(keep)

    X = sub[t_idx, a_idx]
    yy = y[t_idx, a_idx]
    assets = np.array(grid.asset_ids)[a_idx]
    minutes = rows[t_idx]
    return X, yy.astype(np.float64), assets, minutes

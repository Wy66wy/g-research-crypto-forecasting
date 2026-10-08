"""The competition metric.

    score = sum_a w_a * corr_a / sum_a w_a

where ``corr_a`` is the Pearson correlation between prediction and target for
asset ``a`` **over time**, and ``w_a`` comes from ``asset_details.csv``.

Two properties of this metric shape the modelling choices elsewhere.

**It is computed per asset, over time - not across assets at each minute.**
A model that ranks the 14 coins perfectly at every minute but gets each
coin's own timing wrong scores nothing. So the cross-asset model in
:mod:`cryptoforecast.models.crossasset` uses the other assets as *context*
for each asset's own forecast, rather than predicting a cross-sectional
ranking.

**Correlation is invariant to affine rescaling within an asset.** Multiplying
an asset's predictions by 10, or adding a constant, leaves its correlation
unchanged. Prediction magnitude is therefore irrelevant, which is why the
neural network can be trained on per-asset standardised targets without any
recalibration afterwards - and why, unlike in a Sharpe-ratio competition,
shrinking predictions buys nothing here.

One consequence worth knowing: a model that predicts a constant for an asset
has an undefined correlation for that asset. :func:`weighted_correlation`
treats that as 0, which is what it should be - a constant carries no
information - but it means "predict zero" is not a usable baseline for this
metric the way it is for an MSE.
"""

from __future__ import annotations

import logging

import numpy as np

from .constants import ASSET_WEIGHTS

logger = logging.getLogger(__name__)

__all__ = ["pearson", "per_asset_correlation", "weighted_correlation"]


def pearson(a, b) -> float:
    """Pearson correlation, ignoring non-finite pairs.

    Returns 0.0 rather than nan when either side is constant: a constant
    prediction carries no information, so it should contribute nothing to the
    score instead of poisoning the weighted average.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return 0.0
    x, y = x[mask], y[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x * x).sum() * (y * y).sum())
    if denom < 1e-18:
        return 0.0
    return float((x * y).sum() / denom)


def per_asset_correlation(y_true, y_pred, asset_ids) -> dict[int, float]:
    """Pearson correlation computed separately for every asset."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assets = np.asarray(asset_ids).ravel()
    if not (len(y_true) == len(y_pred) == len(assets)):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, "
            f"asset_ids={len(assets)}"
        )
    return {
        int(a): pearson(y_true[assets == a], y_pred[assets == a])
        for a in np.unique(assets)
    }


def weighted_correlation(
    y_true,
    y_pred,
    asset_ids,
    weights: dict[int, float] | None = None,
    return_components: bool = False,
):
    """The competition score: per-asset correlation, weighted average.

    Parameters
    ----------
    y_true, y_pred : array-like
        Targets and predictions, one entry per (minute, asset) row.
    asset_ids : array-like
        The asset each row belongs to.
    weights : dict, optional
        Asset weights. Defaults to the official ones in
        :data:`cryptoforecast.constants.ASSET_WEIGHTS`.
    return_components : bool
        Also return the per-asset correlations. Worth looking at: the mean
        hides how uneven the picture usually is across coins.
    """
    weights = weights or ASSET_WEIGHTS
    corrs = per_asset_correlation(y_true, y_pred, asset_ids)

    num = sum(weights.get(a, 0.0) * c for a, c in corrs.items())
    den = sum(weights.get(a, 0.0) for a in corrs)
    score = num / den if den > 0 else 0.0

    if return_components:
        return float(score), corrs
    return float(score)

"""Loading the data onto a dense (minute x asset) grid.

**The raw data has gaps, and that breaks every rolling computation.**

``train.csv`` is long format: one row per asset per minute *in which the asset
traded*. Thinly traded coins skip minutes regularly, and early in the history
some assets skip hours. So on the raw frame, ``shift(-16)`` within an asset
means "16 rows ahead", which may be 16 minutes or may be three days. A target
or a rolling feature computed that way is silently wrong, and nothing errors.

The fix is to put everything on a complete minute grid first: one row per
minute from the first timestamp to the last, one column per asset, with
explicit "no data" markers where an asset did not trade. Every other module -
the target, the features, the neural network's input windows - then works on
that grid, where "16 steps" really is 16 minutes.

The grid is also the natural shape for the cross-asset model, which needs to
see all 14 coins side by side at each minute.

Memory: the full dataset is about 1.95M minutes x 14 assets. With eight
float32 fields that is roughly 870MB - large, but it fits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from .constants import (
    ASSET_COL,
    ASSET_DETAILS,
    ASSET_IDS,
    RAW_COLS,
    SECONDS_PER_MINUTE,
    TARGET_COL,
    TIMESTAMP_COL,
)

logger = logging.getLogger(__name__)

__all__ = [
    "MarketGrid",
    "resolve_path",
    "load_asset_weights",
    "load_long",
    "to_grid",
    "load_grid",
]


@dataclass
class MarketGrid:
    """All the data, aligned on a complete minute grid.

    Attributes
    ----------
    timestamps : ndarray, shape (T,)
        Unix seconds, strictly consecutive minutes with no gaps.
    asset_ids : list[int]
        Column order of every (T, A) array.
    fields : dict[str, ndarray]
        Each value has shape (T, A), float32, nan where the asset did not
        trade that minute.
    present : ndarray, shape (T, A), bool
        True where the asset genuinely has a row in the raw data. Kept
        separately from the nans because forward-filled prices are *not*
        observations, and the target and the metric must know the difference.
    target : ndarray or None, shape (T, A)
        The provided Target, when the data has one.
    weights : dict[int, float]
        Metric weight per asset.
    """

    timestamps: np.ndarray
    asset_ids: list[int]
    fields: dict[str, np.ndarray]
    present: np.ndarray
    target: np.ndarray | None = None
    weights: dict[int, float] = field(default_factory=dict)

    @property
    def n_minutes(self) -> int:
        return len(self.timestamps)

    @property
    def n_assets(self) -> int:
        return len(self.asset_ids)

    def weight_vector(self) -> np.ndarray:
        """Weights in column order, shape (A,)."""
        return np.array([self.weights.get(a, 0.0) for a in self.asset_ids])

    def slice(self, start: int, stop: int) -> "MarketGrid":
        """A contiguous block of minutes, as a new grid sharing no state."""
        return MarketGrid(
            timestamps=self.timestamps[start:stop],
            asset_ids=list(self.asset_ids),
            fields={k: v[start:stop] for k, v in self.fields.items()},
            present=self.present[start:stop],
            target=None if self.target is None else self.target[start:stop],
            weights=dict(self.weights),
        )

    def coverage(self) -> dict[int, float]:
        """Fraction of minutes in which each asset actually traded."""
        frac = self.present.mean(axis=0)
        return {a: round(float(f), 4) for a, f in zip(self.asset_ids, frac)}


def resolve_path(data_dir: str | Path, filename: str) -> Path:
    """Find a data file, tolerating the usual directory layouts."""
    data_dir = Path(data_dir)
    for c in (
        data_dir / filename,
        data_dir / "g-research-crypto-forecasting" / filename,
        data_dir.parent / filename,
    ):
        if c.exists():
            return c
    raise FileNotFoundError(
        f"could not find {filename} under {data_dir}. If the data has not "
        f"been downloaded, run scripts/generate_synthetic_data.py first."
    )


def load_asset_weights(data_dir: str | Path) -> dict[int, float]:
    """Read the metric weights, preferring the real ``asset_details.csv``.

    The constants module carries a copy of the weights as a fallback, but the
    file shipped with the competition is the authority - if the two ever
    disagree, the file wins.
    """
    try:
        path = resolve_path(data_dir, "asset_details.csv")
        df = pl.read_csv(path)
        weights = {int(r["Asset_ID"]): float(r["Weight"]) for r in df.iter_rows(named=True)}
        logger.info("asset weights read from %s", path)
        return weights
    except FileNotFoundError:
        logger.warning("asset_details.csv not found, using built-in weights")
        return {k: v[1] for k, v in ASSET_DETAILS.items()}


def load_long(
    data_dir: str | Path,
    filename: str = "train.csv",
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> pl.DataFrame:
    """Read the long-format CSV, optionally restricted to a time range.

    Lazy scanning keeps the 3GB real file manageable when only a slice of it
    is needed.
    """
    path = resolve_path(data_dir, filename)
    lf = pl.scan_csv(path, infer_schema_length=10_000)
    if start_ts is not None:
        lf = lf.filter(pl.col(TIMESTAMP_COL) >= start_ts)
    if end_ts is not None:
        lf = lf.filter(pl.col(TIMESTAMP_COL) <= end_ts)
    df = lf.collect()

    # Infinite VWAPs appear in the real data where Volume is zero. They are
    # not prices, so they become missing values here rather than propagating
    # inf through every downstream computation.
    float_cols = [c for c, dt in zip(df.columns, df.dtypes) if dt in (pl.Float32, pl.Float64)]
    df = df.with_columns(
        [
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in float_cols
        ]
    ).with_columns([pl.col(c).fill_nan(None).alias(c) for c in float_cols])

    logger.info("loaded %s long-format rows from %s", f"{df.height:,}", path.name)
    return df


def to_grid(
    df: pl.DataFrame,
    weights: dict[int, float] | None = None,
    asset_ids: list[int] | None = None,
) -> MarketGrid:
    """Pivot long-format data onto a complete minute grid.

    The grid spans every minute from the first timestamp to the last. A cell
    is nan where the asset has no row; ``present`` records which cells were
    real observations.
    """
    asset_ids = asset_ids or ASSET_IDS
    ts = df[TIMESTAMP_COL].to_numpy()
    t0, t1 = int(ts.min()), int(ts.max())

    if (t1 - t0) % SECONDS_PER_MINUTE != 0:
        raise ValueError("timestamps are not aligned to whole minutes")

    timestamps = np.arange(t0, t1 + SECONDS_PER_MINUTE, SECONDS_PER_MINUTE, dtype=np.int64)
    n_t, n_a = len(timestamps), len(asset_ids)

    row = ((ts - t0) // SECONDS_PER_MINUTE).astype(np.int64)
    col_of = {a: j for j, a in enumerate(asset_ids)}
    assets = df[ASSET_COL].to_numpy()
    keep = np.array([a in col_of for a in assets])
    row, assets = row[keep], assets[keep]
    col = np.array([col_of[a] for a in assets], dtype=np.int64)

    # Duplicate (timestamp, asset) rows exist in the raw data. The last one
    # wins here, which matches how a reindex would resolve them.
    present = np.zeros((n_t, n_a), dtype=bool)
    present[row, col] = True

    fields: dict[str, np.ndarray] = {}
    for c in RAW_COLS:
        if c not in df.columns:
            continue
        grid = np.full((n_t, n_a), np.nan, dtype=np.float32)
        grid[row, col] = df[c].to_numpy()[keep].astype(np.float32)
        fields[c] = grid

    target = None
    if TARGET_COL in df.columns:
        target = np.full((n_t, n_a), np.nan, dtype=np.float32)
        target[row, col] = df[TARGET_COL].to_numpy()[keep].astype(np.float32)

    grid = MarketGrid(
        timestamps=timestamps,
        asset_ids=list(asset_ids),
        fields=fields,
        present=present,
        target=target,
        weights=weights or {k: v[1] for k, v in ASSET_DETAILS.items()},
    )
    missing = 1.0 - present.mean()
    logger.info(
        "grid: %s minutes x %d assets, %.1f%% of cells have no trade",
        f"{n_t:,}",
        n_a,
        missing * 100,
    )
    return grid


def load_grid(
    data_dir: str | Path,
    filename: str = "train.csv",
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> MarketGrid:
    """Load the data and the weights, and return them on a minute grid."""
    weights = load_asset_weights(data_dir)
    df = load_long(data_dir, filename, start_ts, end_ts)
    return to_grid(df, weights)

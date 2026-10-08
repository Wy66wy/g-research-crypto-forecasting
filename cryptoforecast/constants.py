"""Column names, asset metadata and the constants that define the target.

Data schema (from the official description)::

    train.csv
        timestamp   Unix seconds, one row per asset per minute (with gaps)
        Asset_ID    0..13
        Count       number of trades in the minute
        Open, High, Low, Close
        Volume      units traded
        VWAP        volume-weighted average price
        Target      15-minute residualised log return (see target.py)

    asset_details.csv
        Asset_ID, Weight, Asset_Name

The two numbers below, ``TARGET_HORIZON`` and ``BETA_WINDOW``, are the whole
definition of the target, and they also fix the leakage boundary for every
other module. A feature on minute ``t`` may use prices up to and including
``t``. The target on minute ``t`` uses prices up to ``t + 16``. Anything in
between is off limits.
"""

from __future__ import annotations

# ---------------------------------------------------------------- columns
TIMESTAMP_COL: str = "timestamp"
ASSET_COL: str = "Asset_ID"
TARGET_COL: str = "Target"

OHLC_COLS: list[str] = ["Open", "High", "Low", "Close"]
RAW_COLS: list[str] = ["Count", *OHLC_COLS, "Volume", "VWAP"]

# ---------------------------------------------------------------- the target
#: The target looks from t+1 to t+16 - a 15-minute log return that starts one
#: minute in the future, not at t. The extra minute is easy to miss and it
#: matters: R(t) = log(Close(t+16) / Close(t+1)).
TARGET_HORIZON: int = 15
TARGET_OFFSET: int = 1

#: Rolling window, in minutes, for the beta used to residualise each asset
#: against the market. About 2.6 days.
BETA_WINDOW: int = 3750

#: Minutes in the future the target can see. The validation embargo must be at
#: least this wide, or a training row's label overlaps the validation period.
TARGET_LOOKAHEAD: int = TARGET_HORIZON + TARGET_OFFSET

SECONDS_PER_MINUTE: int = 60

# ---------------------------------------------------------------- assets
#: The 14 assets and their metric weights. The weights are the log of each
#: coin's market cap at a fixed date, as stated by the organisers; they decide
#: how much each asset's correlation counts in the final score, and also how
#: much each asset contributes to the market return the target is
#: residualised against.
ASSET_DETAILS: dict[int, tuple[str, float]] = {
    0: ("Binance Coin", 4.304065093204169),
    1: ("Bitcoin", 6.779921907472252),
    2: ("Bitcoin Cash", 2.3978952727983707),
    3: ("Cardano", 4.406719247264253),
    4: ("Dogecoin", 3.555348061489413),
    5: ("EOS.IO", 1.3862943611198906),
    6: ("Ethereum", 5.8944028342648505),
    7: ("Ethereum Classic", 2.079441541679836),
    8: ("IOTA", 1.0986122886681098),
    9: ("Litecoin", 2.3978952727983707),
    10: ("Maker", 1.0986122886681098),
    11: ("Monero", 1.6094379124341005),
    12: ("Stellar", 2.079441541679836),
    13: ("TRON", 1.791759469228055),
}

N_ASSETS: int = len(ASSET_DETAILS)
ASSET_IDS: list[int] = sorted(ASSET_DETAILS)
ASSET_WEIGHTS: dict[int, float] = {k: v[1] for k, v in ASSET_DETAILS.items()}
ASSET_NAMES: dict[int, str] = {k: v[0] for k, v in ASSET_DETAILS.items()}

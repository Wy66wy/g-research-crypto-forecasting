#!/usr/bin/env python3
"""Kaggle submission template (paste into a Kaggle Notebook).

The competition uses a time-series API: ``env.iter_test()`` yields one
minute of test rows at a time together with a submission frame, and each
prediction must be returned through ``env.predict`` before the next minute is
released.

Preparation: train locally, upload ``artifacts/`` and ``cryptoforecast/`` as
a Kaggle Dataset, attach it, paste this file, set the paths below.

.. note::

   The API module name and the column carrying the row identifier are taken
   from the competition's starter notebook (``gresearch_crypto``,
   ``row_id``). Check them against the official example before submitting.
"""

from __future__ import annotations

import os
import sys

ARTIFACTS_DIR = "/kaggle/input/gresearch-artifacts/artifacts"
CODE_DIR = "/kaggle/input/gresearch-artifacts"
MODEL_NAME = "lgbm"
COMPETITION_DIR = "/kaggle/input/g-research-crypto-forecasting"

if not os.path.isdir("/kaggle/input"):
    print(__doc__)
    print("\n  This is a Kaggle Notebook template, not a local script.")
    print("  To check the live path locally: python scripts/run_replay.py\n")
    raise SystemExit(0)

sys.path.insert(0, CODE_DIR)

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from cryptoforecast.data import load_asset_weights  # noqa: E402
from cryptoforecast.inference import CryptoPredictor  # noqa: E402

weights = load_asset_weights(COMPETITION_DIR)
PREDICTOR = CryptoPredictor.from_artifacts(ARTIFACTS_DIR, MODEL_NAME, weights)

import gresearch_crypto  # noqa: E402

env = gresearch_crypto.make_env()
for test_df, sample_prediction_df in env.iter_test():
    minute = pl.from_pandas(test_df)
    preds = PREDICTOR.predict(minute)
    preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    sample_prediction_df["Target"] = preds
    env.predict(sample_prediction_df)

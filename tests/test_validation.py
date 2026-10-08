"""Split tests. The embargo must cover the target's lookahead."""

from __future__ import annotations

import numpy as np
import pytest

from cryptoforecast.constants import TARGET_LOOKAHEAD
from cryptoforecast.validation import TimeSplit, holdout, summarise_folds


class TestTimeSplit:
    def test_train_precedes_valid_with_a_gap(self):
        for tr, va in TimeSplit(n_folds=3).split(60_000):
            assert va[0] - tr[-1] - 1 >= TARGET_LOOKAHEAD

    def test_gap_shorter_than_lookahead_is_rejected(self):
        """A gap under 16 minutes lets training labels describe the
        validation period, so it must not be constructible at all."""
        with pytest.raises(ValueError, match="lookahead"):
            TimeSplit(gap_minutes=TARGET_LOOKAHEAD - 1)

    def test_validation_blocks_do_not_overlap(self):
        seen: set[int] = set()
        for _, va in TimeSplit(n_folds=3).split(60_000):
            assert not (seen & set(va.tolist()))
            seen |= set(va.tolist())

    def test_last_minutes_are_never_validated(self):
        """The final 16 minutes have no label."""
        _, va = list(TimeSplit(n_folds=3).split(60_000))[-1]
        assert va[-1] < 60_000 - TARGET_LOOKAHEAD

    def test_warmup_is_skipped(self):
        for tr, _ in TimeSplit(n_folds=2, warmup_minutes=3_750).split(60_000):
            assert tr[0] >= 3_750

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            list(TimeSplit(n_folds=3).split(5_000))

    def test_holdout(self):
        tr, va = holdout(60_000)
        assert tr[-1] < va[0]


def test_summary():
    s = summarise_folds([0.1, 0.2, 0.3])
    assert s["mean"] == pytest.approx(0.2) and s["n_folds"] == 3

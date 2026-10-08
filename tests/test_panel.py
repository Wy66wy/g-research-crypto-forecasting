"""Neural-network input tests. None of these need torch."""

from __future__ import annotations

import numpy as np
import pytest

from cryptoforecast.panel import FeatureScaler, WindowDataset, sample_rows


@pytest.fixture
def tensor():
    T, A, F = 200, 3, 4
    x = np.arange(T * A * F, dtype=np.float32).reshape(T, A, F)
    return x


class TestWindowDataset:
    def test_window_ends_at_the_sample_minute(self, tensor):
        """The last step of the window must be minute t itself - not t+1."""
        ds = WindowDataset(tensor, None, np.array([50]), lookback=10)
        w, _, _, t = ds[0]
        assert t == 50
        np.testing.assert_array_equal(w[:, -1, :], tensor[50])
        np.testing.assert_array_equal(w[:, 0, :], tensor[41])

    def test_shape_is_asset_time_feature(self, tensor):
        w, _, _, _ = WindowDataset(tensor, None, np.array([50]), 10)[0]
        assert w.shape == (3, 10, 4)

    def test_rows_without_enough_history_are_dropped(self, tensor):
        ds = WindowDataset(tensor, None, np.arange(20), lookback=10)
        assert len(ds) == 11            # minutes 9..19

    def test_target_mask_marks_missing(self, tensor):
        target = np.ones((200, 3))
        target[50, 1] = np.nan
        _, y, m, _ = WindowDataset(tensor, target, np.array([50]), 10)[0]
        np.testing.assert_array_equal(m, [1, 0, 1])
        assert y[1] == 0.0               # nan replaced, but masked out of the loss

    def test_target_is_scaled_per_asset(self, tensor):
        target = np.full((200, 3), 2.0)
        ds = WindowDataset(tensor, target, np.array([50]), 10,
                           target_scale=np.array([1.0, 2.0, 4.0]))
        _, y, _, _ = ds[0]
        np.testing.assert_allclose(y, [2.0, 1.0, 0.5])


class TestScaler:
    def test_statistics_come_from_training_rows_only(self, tensor):
        sc = FeatureScaler().fit(tensor, np.arange(100))
        expected = tensor[:100].reshape(-1, 4).mean(axis=0)
        np.testing.assert_allclose(sc.mean_, expected, rtol=1e-6)

    def test_missing_becomes_zero(self):
        x = np.array([[[1.0]], [[np.nan]], [[3.0]]], dtype=np.float32)
        z = FeatureScaler().fit(x, np.arange(3)).transform(x)
        assert z[1, 0, 0] == 0.0

    def test_extreme_values_are_clipped(self):
        x = np.zeros((100, 1, 1), dtype=np.float32)
        x[0] = 1e6
        z = FeatureScaler().fit(x, np.arange(100)).transform(x)
        assert z.max() <= 10.0


def test_sample_rows_strides():
    out = sample_rows(np.arange(100), stride=10, seed=0)
    assert len(out) == 10 and np.all(np.diff(out) == 10)

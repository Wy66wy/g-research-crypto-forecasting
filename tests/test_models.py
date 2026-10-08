"""Model interface tests.

Every model must return predictions shaped (rows, assets), and must predict
from features alone - never from whether a target happens to exist, because
live there is none.
"""

from __future__ import annotations

import numpy as np
import pytest

from cryptoforecast.features import build_feature_tensor
from cryptoforecast.models import get_model


@pytest.fixture(scope="module")
def setup(small_market):
    _, grid = small_market
    tensor, names = build_feature_tensor(grid)
    train = np.arange(3_800, 5_000)
    valid = np.arange(5_100, 5_900)
    return grid, tensor, names, train, valid


class TestTabular:
    @pytest.mark.parametrize("name", ["ridge", "lgbm"])
    def test_prediction_shape(self, setup, name):
        grid, tensor, names, tr, va = setup
        params = {"n_estimators": 20} if name == "lgbm" else {}
        m = get_model(name, **params).fit(tensor, grid, tr, va, feature_names=names)
        pred = m.predict(tensor, grid, va)
        assert pred.shape == (len(va), grid.n_assets)

    def test_no_prediction_where_the_coin_did_not_trade(self, setup):
        grid, tensor, names, tr, va = setup
        pred = get_model("ridge").fit(tensor, grid, tr, feature_names=names).predict(
            tensor, grid, va)
        assert np.isnan(pred[~grid.present[va]]).all()
        assert np.isfinite(pred[grid.present[va]]).all()

    def test_prediction_does_not_depend_on_the_target(self, setup):
        """Wiping the target must not change a single prediction."""
        grid, tensor, names, tr, va = setup
        m = get_model("ridge").fit(tensor, grid, tr, feature_names=names)
        before = m.predict(tensor, grid, va)
        saved = grid.target
        grid.target = np.full_like(saved, np.nan)
        try:
            after = m.predict(tensor, grid, va)
        finally:
            grid.target = saved
        np.testing.assert_array_equal(np.nan_to_num(before), np.nan_to_num(after))


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="unknown model"):
        get_model("transformer-xl")


torch = pytest.importorskip("torch")


class TestCrossAsset:
    def test_all_missing_minute_does_not_produce_nan(self):
        """A minute where every coin is masked would make attention return
        nan without the guard."""
        import torch.nn as nn
        from cryptoforecast.models.crossasset import _build_net

        net = _build_net(nn, 8, 14, 16, 4, 1, 0.0, True).eval()
        x = torch.randn(3, 14, 10, 8)
        present = torch.ones(3, 14, dtype=torch.bool)
        present[1] = False
        with torch.no_grad():
            y = net(x, present)
        assert torch.isfinite(y).all()

    def test_attention_lets_coins_influence_each_other(self):
        """With attention, changing one coin's input must move another coin's
        output. Without it, it must not. This is the property the ablation
        is about, checked directly."""
        import torch.nn as nn
        from cryptoforecast.models.crossasset import _build_net

        torch.manual_seed(0)
        x = torch.randn(1, 14, 10, 8)
        x2 = x.clone()
        x2[0, 0] += 5.0                      # perturb coin 0 only
        present = torch.ones(1, 14, dtype=torch.bool)

        for use_attention, should_move in ((True, True), (False, False)):
            net = _build_net(nn, 8, 14, 16, 4, 1, 0.0, use_attention).eval()
            with torch.no_grad():
                moved = (net(x, present)[0, 5] - net(x2, present)[0, 5]).abs().item()
            assert (moved > 1e-6) == should_move

    def test_save_and_load_round_trip(self, setup, tmp_path):
        grid, tensor, names, tr, va = setup
        m = get_model("crossasset", max_epochs=1, hidden=16, n_layers=1,
                      train_stride=20, lookback=10)
        m.fit(tensor, grid, tr, va, feature_names=names)
        path = tmp_path / "m.pt"
        m.save(path)
        from cryptoforecast.models import load_model

        m2 = load_model("crossasset", path)
        np.testing.assert_allclose(
            np.nan_to_num(m.predict(tensor, grid, va)),
            np.nan_to_num(m2.predict(tensor, grid, va)), rtol=1e-5)

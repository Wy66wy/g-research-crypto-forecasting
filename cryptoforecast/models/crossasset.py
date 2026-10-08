"""A cross-asset neural network: a temporal encoder per coin, then attention
across coins.

Architecture::

    input  (B, A, L, F)    last L minutes of F features, for all A=14 coins
      |
      |  shared across coins
      v
    Linear(F -> H) -> GRU(H)          summarise each coin's own recent history
      |                               into one vector
      v
    (B, A, H)  + learned asset embedding
      |
      v
    Transformer encoder over the A axis   each coin attends to the others
      |
      v
    LayerNorm -> MLP -> (B, A)        one forecast per coin

**Why attention over assets, rather than a bigger sequence model.** The
target is the coin's return *minus its share of the market's return*. So
what is predictable about it is, by construction, relative: is this coin
ahead of or behind the others right now? A model that processes each coin in
isolation can only learn that indirectly, through features someone
engineered to compare coins. Self-attention over the asset axis lets every
coin's representation be updated by every other coin's, so relative
structure - including lead-lag, where a move in Bitcoin has not yet reached
a smaller coin - can be learned rather than hand-built.

**The ablation that matters.** ``use_attention=False`` removes the attention
block and leaves everything else identical: same GRU, same head, same
training. If the full model does not beat that, attention is not earning its
cost, whatever the absolute score looks like.

**Missing coins.** A coin with no trade at the current minute is excluded
from the attention keys via ``src_key_padding_mask``, so stale data from an
illiquid coin cannot contaminate the others' representations. A row where
every coin is missing would make attention produce nan, so that case falls
back to attending over everything.

torch is imported lazily: the rest of the package works without it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..data import MarketGrid
from ..metrics import weighted_correlation
from ..panel import FeatureScaler, WindowDataset, sample_rows
from .base import BaseModel

logger = logging.getLogger(__name__)

__all__ = ["CrossAssetModel"]


def _require_torch():
    try:
        import torch
        import torch.nn as nn

        return torch, nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "CrossAssetModel needs PyTorch: pip install torch\n"
            "The ridge and lgbm models run without it."
        ) from exc


def _build_net(nn, n_features: int, n_assets: int, hidden: int, n_heads: int,
               n_layers: int, dropout: float, use_attention: bool):
    import torch

    class CrossAssetNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.use_attention = use_attention
            self.input_proj = nn.Linear(n_features, hidden)
            self.gru = nn.GRU(hidden, hidden, batch_first=True)
            self.asset_emb = nn.Embedding(n_assets, hidden)
            if use_attention:
                layer = nn.TransformerEncoderLayer(
                    d_model=hidden,
                    nhead=n_heads,
                    dim_feedforward=hidden * 2,
                    dropout=dropout,
                    batch_first=True,
                    # Pre-norm trains more stably on noisy targets than the
                    # original post-norm arrangement
                    norm_first=True,
                )
                self.attn = nn.TransformerEncoder(
                    layer, num_layers=n_layers, enable_nested_tensor=False
                )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

        def forward(self, x, present):
            B, A, L, F = x.shape
            h = torch.relu(self.input_proj(x.reshape(B * A, L, F)))
            _, hn = self.gru(h)                       # hn: (1, B*A, H)
            h = hn[-1].reshape(B, A, -1)
            h = h + self.asset_emb.weight[None, :, :]

            if self.use_attention:
                pad = ~present                        # True = ignore this key
                everyone_missing = pad.all(dim=1, keepdim=True)
                pad = pad & ~everyone_missing         # avoid an all-masked row
                h = self.attn(h, src_key_padding_mask=pad)

            return self.head(h).squeeze(-1)           # (B, A)

    return CrossAssetNet()


class CrossAssetModel(BaseModel):
    """GRU encoder per coin, optional self-attention across coins."""

    name = "crossasset"

    def __init__(
        self,
        lookback: int = 30,
        hidden: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_attention: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 256,
        max_epochs: int = 12,
        patience: int = 3,
        train_stride: int = 3,
        seed: int = 42,
        num_threads: int | None = None,
        **params,
    ) -> None:
        super().__init__(**params)
        self.lookback = lookback
        self.hidden = hidden
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_attention = use_attention
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.train_stride = train_stride
        self.seed = seed
        self.num_threads = num_threads

        self.net = None
        self.scaler: FeatureScaler | None = None
        self.target_scale_: np.ndarray | None = None
        self.n_features_: int | None = None
        self.n_assets_: int | None = None
        self.history_: list[dict] = []

        if not use_attention:
            self.name = "gru"

    # ------------------------------------------------------------ helpers
    def _batches(self, ds: WindowDataset, shuffle: bool, rng: np.random.Generator):
        idx = np.arange(len(ds))
        if shuffle:
            rng.shuffle(idx)
        for s in range(0, len(idx), self.batch_size):
            items = [ds[int(i)] for i in idx[s : s + self.batch_size]]
            x = np.stack([it[0] for it in items])
            y = np.stack([it[1] for it in items])
            m = np.stack([it[2] for it in items])
            t = np.array([it[3] for it in items])
            yield x, y, m, t

    def _score(self, tensor_scaled, grid, rows) -> float:
        pred = self._predict_scaled(tensor_scaled, grid, rows)
        y = grid.target[rows]
        ok = grid.present[rows] & np.isfinite(y) & np.isfinite(pred)
        t, a = np.nonzero(ok)
        return weighted_correlation(y[t, a], pred[t, a], np.array(grid.asset_ids)[a])

    # ------------------------------------------------------------ fit
    def fit(self, tensor, grid, train_rows, valid_rows=None, feature_names=None):
        torch, nn = _require_torch()
        if self.num_threads:
            torch.set_num_threads(self.num_threads)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        self.feature_names = feature_names
        self.n_features_ = tensor.shape[2]
        self.n_assets_ = grid.n_assets

        # Scaling statistics come from training minutes only
        self.scaler = FeatureScaler().fit(tensor, train_rows)
        scaled = self.scaler.transform(tensor)

        # Per-asset target scale, also training-only (see panel.py for why)
        y_tr = grid.target[train_rows].astype(np.float64)
        sd = np.nanstd(y_tr, axis=0)
        self.target_scale_ = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)

        train_ds = WindowDataset(
            scaled, grid.target,
            sample_rows(train_rows, self.train_stride, self.seed),
            self.lookback, self.target_scale_,
        )

        self.net = _build_net(
            nn, self.n_features_, self.n_assets_, self.hidden, self.n_heads,
            self.n_layers, self.dropout, self.use_attention,
        )
        n_params = sum(p.numel() for p in self.net.parameters())
        logger.info(
            "%s: %s parameters, %s training windows",
            self.name, f"{n_params:,}", f"{len(train_ds):,}",
        )

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        steps = self.max_epochs * max(1, -(-len(train_ds) // self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=steps)

        # Loss weights: the metric weights coins by log market cap, so the
        # loss does too. Normalised so they average to one.
        w = grid.weight_vector()
        w_t = torch.tensor(w / w.mean(), dtype=torch.float32)

        best, best_state, bad = -np.inf, None, 0
        for epoch in range(self.max_epochs):
            self.net.train()
            total, n = 0.0, 0
            for x, y, m, t in self._batches(train_ds, True, rng):
                xb = torch.from_numpy(x)
                yb = torch.from_numpy(y)
                mb = torch.from_numpy(m)
                pres = torch.from_numpy(grid.present[t])

                pred = self.net(xb, pres)
                cell_w = mb * w_t[None, :]
                loss = (cell_w * (pred - yb) ** 2).sum() / cell_w.sum().clamp(min=1e-8)

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                sched.step()

                total += float(loss.detach()) * len(x)
                n += len(x)

            record = {"epoch": epoch, "train_loss": total / max(n, 1)}
            if valid_rows is not None:
                score = self._score(scaled, grid, valid_rows)
                record["valid_score"] = score
                logger.info(
                    "epoch %02d | train_loss %.5f | valid score %+.5f",
                    epoch, record["train_loss"], score,
                )
                if score > best:
                    best, bad = score, 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= self.patience:
                        logger.info("early stop at epoch %d, best %+.5f", epoch, best)
                        self.history_.append(record)
                        break
            else:
                logger.info("epoch %02d | train_loss %.5f", epoch, record["train_loss"])
            self.history_.append(record)

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------ predict
    def _predict_scaled(self, scaled, grid, rows):
        torch, _ = _require_torch()
        out = np.full((len(rows), grid.n_assets), np.nan, dtype=np.float64)
        ds = WindowDataset(scaled, None, rows, self.lookback)
        pos = {int(r): i for i, r in enumerate(rows)}

        self.net.eval()
        with torch.no_grad():
            for x, _, _, t in self._batches(ds, False, np.random.default_rng(0)):
                pres = torch.from_numpy(grid.present[t])
                p = self.net(torch.from_numpy(x), pres).numpy()
                for k, tt in enumerate(t):
                    out[pos[int(tt)]] = p[k]
        out[~grid.present[rows]] = np.nan
        return out

    def predict(self, tensor, grid, rows):
        self._check_fitted()
        return self._predict_scaled(self.scaler.transform(tensor), grid, rows)

    # ------------------------------------------------------------ IO
    def save(self, path):
        torch, _ = _require_torch()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "config": {
                    "lookback": self.lookback, "hidden": self.hidden,
                    "n_heads": self.n_heads, "n_layers": self.n_layers,
                    "dropout": self.dropout, "use_attention": self.use_attention,
                },
                "n_features": self.n_features_,
                "n_assets": self.n_assets_,
                "scaler_mean": self.scaler.mean_,
                "scaler_std": self.scaler.std_,
                "target_scale": self.target_scale_,
                "feature_names": self.feature_names,
            },
            str(path),
        )

    @classmethod
    def load(cls, path):
        torch, nn = _require_torch()
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        m = cls(**blob["config"])
        m.n_features_, m.n_assets_ = blob["n_features"], blob["n_assets"]
        m.net = _build_net(nn, m.n_features_, m.n_assets_, m.hidden, m.n_heads,
                           m.n_layers, m.dropout, m.use_attention)
        m.net.load_state_dict(blob["state_dict"])
        m.scaler = FeatureScaler()
        m.scaler.mean_, m.scaler.std_ = blob["scaler_mean"], blob["scaler_std"]
        m.target_scale_ = blob["target_scale"]
        m.feature_names = blob["feature_names"]
        m.is_fitted = True
        return m

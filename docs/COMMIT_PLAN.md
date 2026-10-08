# Suggested commit sequence

Pushing everything at once makes it obvious from the history that the project
was not built incrementally. The batches below follow the order the work
actually happened in.

More usefully: **read each module before committing it.** Under each batch are
the questions someone is most likely to ask about that code.

---

## Batch 1: skeleton

```bash
git init
git add .gitignore LICENSE requirements.txt README.md
git commit -m "chore: initialise project with dependencies and README"
```

---

## Batch 2: constants and the metric

```bash
git add cryptoforecast/__init__.py cryptoforecast/constants.py cryptoforecast/metrics.py
git commit -m "feat: per-asset weighted correlation metric"

git add tests/__init__.py tests/conftest.py tests/test_metrics.py
git commit -m "test: metric is per-asset and invariant to per-asset rescaling"
```

**Read this**: the score is a correlation computed per coin *over time*, then
weighted. Why does that make prediction magnitude irrelevant? Why is "predict
zero" not a usable baseline here?

---

## Batch 3: the minute grid

```bash
git add cryptoforecast/data.py
git commit -m "feat: load onto a complete minute grid with explicit missing minutes"

git add tests/test_data.py
git commit -m "test: a shift on the grid is a shift in minutes, not rows"
```

**Read this carefully**: on the raw frame, what does `shift(-16)` within an
asset actually mean when the coin skips minutes? Why does the grid carry a
separate `present` mask instead of just using nans?

---

## Batch 4: reimplementing the target

```bash
git add cryptoforecast/target.py
git commit -m "feat: reimplement the residualised target from close prices"

git add scripts/generate_synthetic_data.py
git commit -m "feat: synthetic data with lead-lag and an independent target implementation"

git add tests/test_target.py scripts/verify_target.py
git commit -m "test: target matches an independent implementation exactly"
```

**Read this carefully** - this module fixes the leakage boundary for
everything else:
- Why does the return run from t+1 to t+16 rather than t to t+15?
- Why is the beta `mean(R*M) / mean(M^2)` and not `cov / var`? Which test
  pins that down?
- Why does every coin's target depend on every other coin's future price?
- Why does the synthetic generator compute its target with *different* code?

---

## Batch 5: features

```bash
git add cryptoforecast/features.py
git commit -m "feat: causal own-asset and cross-asset features"

git add tests/test_features.py
git commit -m "test: perturbation test for lookahead across all features"
```

**Read this**: the perturbation test corrupts everything after minute t and
checks nothing at or before t changed. Why is that stronger than testing each
shift individually?

---

## Batch 6: validation

```bash
git add cryptoforecast/validation.py tests/test_validation.py
git commit -m "feat: walk-forward splits with an embargo sized to the target"
```

**Read this**: why must the gap be at least 16 minutes, and why does
constructing a shorter one raise instead of warning?

---

## Batch 7: tabular models

```bash
git add cryptoforecast/models/base.py cryptoforecast/models/tabular.py
git commit -m "feat: grid-level model interface with pooled ridge and lightgbm"
```

**Read this**: why does every model take the whole feature tensor and return
predictions shaped (rows, assets), even the tabular ones?

---

## Batch 8: the neural network

```bash
git add cryptoforecast/panel.py tests/test_panel.py
git commit -m "feat: lazily sliced (asset, time, feature) windows"

git add cryptoforecast/models/crossasset.py cryptoforecast/models/__init__.py
git commit -m "feat: GRU encoder per coin with self-attention across coins"
```

**Read this carefully**:
- Why attention over the *asset* axis rather than a bigger sequence model?
- Why are targets standardised per coin before the loss, and why does nothing
  need undoing afterwards?
- What does `src_key_padding_mask` do here, and what breaks without the guard
  for a row where every coin is missing?
- Why is `gru` (attention removed) the comparison that matters most?

---

## Batch 9: training and inference

```bash
git add cryptoforecast/config.py cryptoforecast/train.py configs/
git commit -m "feat: training CLI and the six-way ablation"

git add cryptoforecast/inference.py scripts/run_replay.py scripts/kaggle_submission.py
git commit -m "feat: rolling-buffer predictor and minute-by-minute replay"
```

---

## Batch 10: tooling and analysis

```bash
git add notebooks/01_eda.ipynb Makefile scripts/run_pipeline.sh
git commit -m "chore: EDA notebook, Makefile and pipeline script"
```

---

## Batch 11: documentation

```bash
git add README.md docs/
git commit -m "docs: design notes, ablation results and debugging record"
```

---

## Notes

- **Spread the batches over several days.**
- Messages follow [Conventional Commits](https://www.conventionalcommits.org/).
- A couple of `fix:` commits are realistic. This project had genuine ones:
  vectorising `forward_fill` after the loop version proved far too slow, and
  the NaN/null mix-up in the generator's summary.

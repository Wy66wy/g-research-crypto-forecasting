# G-Research Crypto Forecasting

A solution for the [Kaggle competition](https://www.kaggle.com/competitions/g-research-crypto-forecasting): predicting 15-minute residualised returns for 14 cryptocurrencies from minute-level market data.

The competition closed in 2022. This project is built around one question the target definition raises directly: **the target is a coin's return relative to the market, so is the predictable part relative too — and can a neural network that sees all fourteen coins at once learn that structure better than hand-engineered features?**

---

## Contents

- [The problem](#the-problem)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Core design](#core-design)
- [Measured results](#measured-results)
- [Submitting to Kaggle](#submitting-to-kaggle)
- [Things that went wrong](#things-that-went-wrong)

---

## The problem

**Data.** One row per coin per minute in which that coin traded: `Count`, `Open`, `High`, `Low`, `Close`, `Volume`, `VWAP`, and `Target`. Roughly 24 million rows over 2018–2021, about 3GB.

**Target.** A 15-minute log return, residualised against the market:

```
R_a(t)      = log( Close_a(t+16) / Close_a(t+1) )
M(t)        = sum_a w_a R_a(t) / sum_a w_a
beta_a(t)   = mean(R_a · M) / mean(M²)          over a trailing 3750-minute window
Target_a(t) = R_a(t) − beta_a(t) · M(t)
```

**Metric.** Pearson correlation between prediction and target, computed **separately for each coin over time**, then averaged with weights equal to the log of each coin's market cap.

**Submission.** A time-series API releases one minute of data at a time.

### Why the target definition matters so much

Three details in that formula are easy to misread, and each one changes the answer:

1. **The return starts at t+1, not t.** It runs from one minute ahead to sixteen minutes ahead.
2. **The beta regression has no intercept.** It is `mean(R·M) / mean(M²)`, not `cov(R, M) / var(M)`. On 15-minute returns the two are close but not equal, and only the uncentred form reproduces the official number.
3. **Every coin's target depends on every other coin's future.** `M(t)` is built from all fourteen coins' forward returns, so Dogecoin's target at minute `t` encodes how Bitcoin moved between `t+1` and `t+16`.

Point 3 is what makes this a cross-asset problem. It means no market-level quantity built from forward prices can ever be a feature, and it is also the structural argument for a model that sees all coins together.

---

## Quick start

No Kaggle account needed — the project ships a synthetic data generator with the same schema.

```bash
git clone <your-repo-url>
cd g-research-crypto
pip install -r requirements.txt

bash scripts/run_pipeline.sh
```

Step by step:

```bash
make data       # synthetic data (~1 minute)
make verify     # check the target reimplementation
make test       # 68 unit tests
make train      # LightGBM, cross-validated
make replay     # minute-by-minute inference path
make ablation   # the six-way comparison (slow on CPU: two neural networks)
```

With the real data, put `train.csv` and `asset_details.csv` in `data/` and run `make verify` first. If it does not report a match, something about the target definition has been misread and every result downstream is suspect.

---

## Project layout

```
g-research-crypto/
├── cryptoforecast/
│   ├── constants.py      # the target's horizon, offset and beta window
│   ├── data.py           # ★ loading onto a complete minute grid
│   ├── target.py         # ★ reimplementation of the residualised target
│   ├── features.py       # own-asset and cross-asset features, all causal
│   ├── metrics.py        # per-asset weighted correlation
│   ├── validation.py     # walk-forward folds with an embargo sized to the target
│   ├── panel.py          # (asset, time, feature) windows for the network
│   ├── config.py
│   ├── train.py          # training CLI and the ablation
│   ├── inference.py      # rolling-buffer predictor for the live API
│   └── models/
│       ├── base.py       # grid-level interface shared by every model
│       ├── tabular.py    # pooled Ridge and LightGBM
│       └── crossasset.py # ★ GRU per coin + self-attention across coins
├── scripts/
│   ├── generate_synthetic_data.py  # includes an independent target implementation
│   ├── verify_target.py            # reimplementation vs provided Target
│   ├── check_consistency.py        # ★ batch vs live prediction, row by row
│   ├── run_replay.py
│   ├── kaggle_submission.py
│   └── run_pipeline.sh
├── tests/                # 68 unit tests
├── configs/
└── notebooks/01_eda.ipynb
```

---

## Core design

### 1. Everything runs on a complete minute grid

The raw file only has a row when a coin traded. Thinly traded coins skip minutes regularly, and some skip hours. On that frame, `shift(-16)` within a coin means *16 rows ahead* — which might be 16 minutes, or three days.

So the first thing the loader does is pivot onto a dense grid: one row per minute, one column per coin, with a separate `present` mask recording which cells were real trades. The mask is kept apart from the nans on purpose: a forward-filled price is not an observation, and the target and the metric both need to know the difference.

Every other module — the target, the features, the network's input windows — works on this grid, where 16 steps really are 16 minutes.

### 2. The target is reimplemented and cross-checked

`target.py` rebuilds the target from close prices. The synthetic generator computes its own target with **separate code** that shares nothing with `target.py`: explicit sliding windows instead of cumulative sums, and a per-row loop for the market return.

On the synthetic data the two agree on all 767,353 cells — the same nan pattern, a maximum difference of 1.4e-7, a correlation of 1.0000000. Two implementations agreeing is a real check; one function compared against itself is not. `scripts/verify_target.py` runs the same comparison against the provided `Target` column on real data.

### 3. Lookahead is tested by perturbation

Testing that every `shift` points the right way only checks the shifts you remembered to check. The feature tests do something blunter: replace everything after minute `t` with garbage, rebuild the features, and require every feature at or before `t` to be unchanged. Any lookahead anywhere — a wrong shift, a centred window, a market statistic computed over the future — fails it, regardless of how it was written.

### 4. The cross-asset network

```
(B, A, L, F)   last L minutes of F features, for all A = 14 coins
     │
     │  shared weights across coins
     ▼
Linear → GRU            summarise each coin's recent history into one vector
     │
(B, A, H) + asset embedding
     │
     ▼
Transformer encoder over the A axis     each coin attends to the others
     │
     ▼
LayerNorm → MLP → (B, A)                one forecast per coin
```

**Why attention over coins rather than a bigger sequence model.** What is predictable about a residual return is, by construction, relative. A model that processes each coin alone can only learn relative structure through features someone engineered to compare coins. Self-attention over the asset axis lets each coin's representation be updated by the others', so that structure can be learned.

**Targets are standardised per coin before the loss.** Volatility differs by an order of magnitude across coins, so a squared-error loss on raw targets is dominated by the wildest ones. The metric is a per-coin correlation, which ignores scale — so dividing each coin's target by its training-period standard deviation pushes the network to get every coin's direction right, and nothing needs undoing at prediction time.

**Missing coins are masked out of attention**, so stale data from an illiquid coin cannot contaminate the others. A minute where every coin is missing would make attention return nan, so that case falls back to attending over everything.

**The ablation that matters** is `gru` against `crossasset`: the same network with and without the attention block, everything else identical.

---

## Measured results

All numbers below were measured on the synthetic data: three walk-forward folds of one week each, a 60-minute embargo, identical folds and an identical feature tensor for every model.

| Item | Result |
|---|---|
| Unit tests | 68 / 68 passing |
| Target reimplementation vs independent implementation | correlation 1.0000000, max diff 1.4e-7 |
| Live inference vs batch (LightGBM) | max diff 0.0, bit-identical |
| Live inference vs batch (neural network) | max diff 6e-8, float32 precision |
| Synthetic market | mean pairwise 15-minute correlation 0.675 |

### The six-way ablation

| Model | Features | mean | std |
|---|---|---:|---:|
| Ridge | own only | 0.0963 | 0.0131 |
| Ridge | own + cross | 0.2051 | 0.0052 |
| LightGBM | own only | 0.1151 | 0.0034 |
| LightGBM | own + cross | 0.2176 | 0.0011 |
| GRU, no attention | own + cross | 0.2281 | 0.0056 |
| **GRU + cross-asset attention** | own + cross | **0.2323** | 0.0058 |

"Own" features use only the coin's own history. "Cross" adds the market's recent return, the coin's return relative to it, and the recent moves of Bitcoin and Ethereum.

**Cross-asset information is the biggest single lever.** Adding the market's
recent return, the coin's return relative to it, and Bitcoin's and Ethereum's
recent moves roughly doubles every tabular model's score: Ridge 0.096 to
0.205, LightGBM 0.115 to 0.218. That is far larger than any difference
between model families, and it follows directly from the target's definition
- a residualised return is a relative quantity, so relative features are the
natural ones.

**A sequence input is worth more than attention.** The GRU beats LightGBM by
0.010 while seeing exactly the same 28 features, and wins on all three folds.
The difference is what each model is shown: LightGBM gets one minute's
features, the GRU gets thirty minutes of them. Ordering and persistence
inside that window carry information that cannot be recovered from a single
row, and engineering it away by hand would mean guessing which lags matter.

**Attention adds a small, consistent gain.** 0.2281 to 0.2323, and the
cross-asset model wins on each fold individually. Three out of three is
encouraging but not conclusive - a coin flip manages it one time in eight -
and the per-fold spread (0.0058) is larger than the gap. The honest reading
is that attention helps a little here, at roughly ten times the training
cost of LightGBM on a CPU.

**The caveat that matters most.** The lead-lag effect in the synthetic data
is *planted*: slower coins respond to the market factor with a two- to
four-minute delay, specifically so that a cross-asset model has something to
find that a per-coin model cannot. So this ablation shows the architecture
**can** recover cross-asset structure when it exists. It says nothing about
how much of it exists in real crypto data, and the honest next step is to run
the same comparison there.

> ⚠️ **The synthetic signals are much stronger than reality.** A score of 0.2 here corresponds to something well above anything achieved on the real leaderboard. The synthetic data exists to verify correctness and to compare model families under known structure, not to estimate achievable scores.

---

## Submitting to Kaggle

1. Train locally to get `artifacts/`
2. Upload `artifacts/` and `cryptoforecast/` as a Kaggle Dataset
3. Attach it to a notebook, paste `scripts/kaggle_submission.py`, set the paths
4. Submit

**Run the consistency check first:**

```bash
python scripts/check_consistency.py --name lgbm
```

Training scores predictions from one big feature tensor. The live path rebuilds the grid from a rolling buffer every minute. Those are different code paths, and this check requires them to produce identical predictions.

> The API module name and identifier column in the template come from the competition's starter notebook. Check them against the official example before submitting.

---

## Things that went wrong

### My synthetic market barely co-moved, and I described it as if it did

The first version of the generator gave the market factor a per-minute volatility of about 0.045%, against idiosyncratic noise of about 0.12% per coin. The noise swamped the factor. The EDA notebook then measured a **mean pairwise correlation of 0.16** — real crypto sits around 0.5 to 0.8.

The notebook text above that measurement said "a mean pairwise correlation this high means most of any coin's move is simply the market". I had written it before running the cell.

This was not cosmetic. The whole argument for this project runs: coins move together → so the target is residualised → so the predictable part is relative → so a cross-asset model is worth trying. My own data did not support the first link. Fixing the factor volatility brought the correlation to 0.675 and **changed the headline result**: on the weak-factor data, cross-asset features were worth +0.04; on realistic data they are worth +0.11. Had I only edited the prose, the README would have reported a value of cross-asset information understated by nearly a factor of three.

The generator now prints the pairwise correlation every time it runs, so this is checked rather than assumed.

### The replay scored far above cross-validation

The first minute-by-minute replay scored **0.31**, against a cross-validated 0.21. By the rule written in the replay script's own docstring, a replay scoring well above validation is a warning sign for leakage.

A score cannot distinguish luck from a leak, so `check_consistency.py` predicts one fixed set of minutes twice — once in batch, once through the live predictor — and compares row by row. The difference was **exactly zero** on every cell, and the batch path scored the same high number on the same window. The high score belongs to the window, not to the live path: over a few hundred minutes, each coin's correlation has a standard error of roughly 0.05.

One real bias is worth noting. The final model early-stops on the last validation block, so evaluating on that block is mildly optimistic. It affects batch and live identically, so it does not touch the consistency result.

### A test failed while the code was right

The first version of the live-vs-batch test replayed 120 minutes, skipped the
first 90 as buffer warm-up, and then asserted that more than 500 cells had
been compared. Thirty minutes times fourteen coins is 420, so the assertion
could not pass however correct the code was. The predictions it was guarding
agreed to 2e-10.

Worth recording because the failure mode is seductive in the other direction:
a test that fails for a reason unrelated to the property it checks invites
you to go looking for a bug that is not there. The threshold is now expressed
as a fraction of the window actually being compared, so it scales with the
test rather than encoding a number I guessed.

### `forward_fill` was a Python loop

The first version walked every cell. On the full dataset that is about 27 million iterations. Replaced with a running maximum over "index of the last valid row": 2 million minutes × 14 coins now fills in 0.7 seconds.

### polars keeps NaN and null apart

numpy's `nan` becomes a float NaN in polars, not a null. The generator's summary used `drop_nulls()`, which kept them, so the reported target standard deviation came out as nan. The loader already normalises NaN to null on read; the summary now filters on finiteness instead.

### `pkill -f` killed its own shell

Stopping a background training run with `pkill -f "cryptoforecast.train"` from inside a shell whose command line also contained that string killed the shell running the command. Nothing after it executed, and nothing reported an error. `-f` matches the full command line of every process, including the one doing the matching.

### Installing torch filled the disk

The CUDA build of torch from PyPI pulls several gigabytes of dependencies, and pip downloads every wheel before installing any of them, so peak usage is the sum of all of them twice over. The install ran out of space on the last package, leaving a truncated `libtorch_cpu.so` that failed with a bus error on import. Every CUDA library had in fact installed completely; removing only the broken torch and reinstalling it with `--no-deps` fixed it. On a machine with network access to PyTorch's own index, the CPU build avoids all of this.

---

## Possible next steps

- [ ] Run `verify_target.py` on the real data and confirm the reimplementation matches
- [ ] Ensemble LightGBM with the network — their errors are unlikely to be fully correlated
- [ ] Use a separate early-stopping split inside the training window, removing the mild selection bias
- [ ] Longer lookback windows for the network, now that the GRU's sequence input has been shown to help
- [ ] Per-coin attention weights as a diagnostic: which coins does each coin attend to?

## License

MIT

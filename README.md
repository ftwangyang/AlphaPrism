# AlphaPrism

Code for **AlphaPrism: Category-Guided LLM Alpha Mining with Dual-Branch Causal Combination**.

## Method Summary

AlphaPrism consists of two stages under strict market-label causality ($\tau \le t - 21$ for 20-day forward close returns):

1. **Category-Guided Residual LLM Mining (`mine`)**: Queries an LLM across five financial factor categories (reversal/trend, volatility/dispersion, liquidity/volume, price-volume correlation, and intraday/overnight patterns) using a shared 22-operator prefix grammar ($d \in [1, 40]$) and volume-normalization constraints. Round 0 runs 10 queries (2 per category, 10 candidates each) and Rounds 1 to 3 run 15 queries (1 per category per round, 12 candidates each) conditioned on unexplained training residuals $y_t^{\perp}$.
2. **Dual-Branch Causal Factor Combination (`eval`)**: Maps mined factors into cross-sectional rank-normalized space ($\mathrm{RankZ}$) and combines two causal branches:
   - **Linear branch**: Multi-scale causal moments (75% expanding average, 25% EMA with half-life 750 days), per-category top-4 orthogonal selection ($|\rho| \le 0.65$), scale-invariant Ridge regression ($\lambda = 0.1$), and in-category $\widehat{\mathrm{IC}}$ aggregation.
   - **Nonlinear branch**: Annual walk-forward shallow LightGBM (depth 3, 7 leaves, 120 trees, recency half-life 1000 days, 21-day purge gap) on the hybrid target $0.7\,\mathrm{ClipZ}(y) + 0.3\,\mathrm{RankZ}(y)$.
   - **Adaptive Max-IR fusion**: Online regularized Max-IR weighting ($\gamma = 0.05$, $w_{\min} = 0.15$) estimated from out-of-sample branch scores starting in 2021.

## Files

- `alphaprism.py`: Full pipeline implementation (Qlib data loader, 22-operator parser and evaluator, Stage 1 LLM mining, Stage 2 dual-branch combination, ablations, and sensitivity sweeps).
- `pools/`: Mined 5-seed factor pools (`seeds 42..46`) for CSI300 and CSI500.
- `requirements.txt`: Python package requirements.

## Installation and Data

```bash
pip install -r requirements.txt
```

Prepare Qlib Chinese A-share daily data (`cn_data`) covering `2010-01-01` through early `2025` with `instruments/csi300.txt` and `instruments/csi500.txt`, and pass its path via `--qlib_path /path/to/qlib_data/cn_data`.

## Usage

### Evaluate Included Factor Pools

```bash
python alphaprism.py eval \
  --market all \
  --seeds 42 43 44 45 46 \
  --pool_dir pools \
  --qlib_path /path/to/qlib_data/cn_data
```

Add `--ablations` and/or `--sensitivity` to run the ablation variants and parameter sensitivity sweeps.

### Run LLM Mining from Scratch

Start an OpenAI-compatible inference endpoint (for example, `vllm serve Qwen/Qwen3-32B-AWQ --port 8000`) and run:

```bash
python alphaprism.py mine \
  --market csi300 \
  --seeds 42 43 44 45 46 \
  --qlib_path /path/to/qlib_data/cn_data \
  --api_url <YOUR_API_URL> \
  --model Qwen3-32B-AWQ \
  --out_dir pools
```

Use the `run` subcommand instead of `mine` to execute Stage 1 mining followed immediately by Stage 2 evaluation.

## Results (2022-01-01 to 2024-12-31, 5-Seed Mean)

| Market | Variant | IC | ICIR | RankIC | RankICIR | AR (%) | MDD (%) | SR |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CSI300 | Without residual refinement | 0.068 | 0.427 | 0.097 | 0.577 | 7.54 ± 0.67 | -15.71 | 1.86 |
| CSI300 | Global top-20 (no category quota) | 0.067 | 0.436 | 0.094 | 0.579 | 6.15 ± 0.33 | -15.01 | 1.52 |
| CSI300 | Linear branch only | 0.070 | 0.438 | 0.098 | 0.601 | 6.88 ± 1.27 | -14.52 | 1.71 |
| CSI300 | Nonlinear branch only | 0.063 | 0.369 | 0.090 | 0.512 | 5.83 ± 0.97 | -16.15 | 1.32 |
| CSI300 | **AlphaPrism** | **0.074** | **0.459** | **0.103** | **0.609** | **7.85 ± 0.75** | **-14.48** | **1.94** |
| CSI500 | Without residual refinement | 0.111 | 0.779 | 0.127 | 0.836 | 12.00 ± 0.43 | -14.63 | 2.45 |
| CSI500 | Global top-20 (no category quota) | 0.112 | 0.793 | 0.127 | 0.855 | 12.41 ± 0.28 | -14.56 | 2.50 |
| CSI500 | Linear branch only | 0.089 | 0.572 | 0.116 | 0.757 | 9.05 ± 1.02 | -16.04 | 1.94 |
| CSI500 | Nonlinear branch only | 0.112 | 0.790 | 0.127 | 0.858 | 12.40 ± 0.46 | -15.02 | 2.45 |
| CSI500 | **AlphaPrism** | **0.113** | **0.800** | **0.128** | **0.861** | **12.60 ± 0.33** | **-14.16** | **2.56** |

AR, MDD, and SR follow the shared daily-equivalent 20-day forward return proxy $r_t = \frac{1}{20|S_t|}\sum_{i\in S_t} y_{i,t}$ on the equal-weight top-20% portfolio without transaction costs and with zero risk-free rate.

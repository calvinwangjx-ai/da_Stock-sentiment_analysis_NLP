# Backtesting and Trading Simulation

Evaluates prediction-driven trading strategies against historical data.

## Files

| File | Purpose |
| --- | --- |
| `walk_forward_backtest.py` | **Real backtest.** Trades the out-of-sample predictions of the models in `07_model_training/classification_models/train_ensemble_models.py` on actual daily closes, with execution delay, transaction costs, Buy & Hold and MA-crossover benchmarks, and a luck test |
| `backtesting_framework.py` | Demo only — simulates a random-walk price series (`generate_historical_data`) and, unless `advanced_analytics_integration` and its models are importable, trades random BUY/SELL/HOLD signals. Its numbers say nothing about the models |
| `live_trading_engine.py` | Simulated live trading engine — polls for new predictions, generates orders, tracks positions and P&L |
| `quick_backtest.py` | Smoke test that runs the framework end-to-end with a small dataset |
| `quick_test.py` | Smoke test that exercises the live trading engine |

## Legacy

`legacy/` contains:

- `live_trading_engine.py` — v1 (superseded by the fixed version)
- `balanced_production.py` — a specific production configuration variant

## Metrics

The framework computes:

- Sharpe ratio
- Maximum drawdown
- Win rate
- Cumulative performance
- Strategy returns vs. Buy & Hold

## `walk_forward_backtest.py`

### Method

1. **Out-of-sample signals.** Calendar dates are split into 5 expanding
   walk-forward folds with a gap equal to the label horizon. Each model is
   trained on labelled rows from the training dates only and predicts the
   next block, so every trading signal comes from a model that never saw
   that period.
2. **Positions.** At the close of day *t*: long if P(target) > threshold,
   else flat (`--mode long_short` goes short instead of flat).
3. **Execution.** Entered `--delay` closes after the signal (default 1,
   because the features use the close of day *t* itself), then earns
   close-to-close returns.
4. **Costs.** `--cost-bps` (default 5) per unit of turnover, one way.
5. **Portfolio.** Equal weight across tickers, rebalanced daily.
6. **Benchmarks** over the identical window: Buy & Hold (same costs) and an
   MA20 > MA50 trend-following rule (same delay and costs).
7. **Placebo test** (`--placebo-runs N`, the one to trust). Each model is
   retrained N times on randomly permuted labels and traded identically;
   the p-value is the share of those no-information strategies whose net
   Sharpe matches or beats the real one.
8. **Shift test** (always on, optimistic). Positions are circularly shifted
   against returns by every offset ≥ 20 days, keeping exposure, trades and
   holding periods but randomising timing. On this data, shuffled-label
   models pass it (p < 0.05) about 20 % of the time, because sitting out one
   big sell-off dominates the window, so do not rely on it alone.

### Usage

Restore the dataset first (see `07_model_training/classification_models/README.md`), then:

```bash
python 09_backtesting_trading/walk_forward_backtest.py
```

```bash
python 09_backtesting_trading/walk_forward_backtest.py --target trend_3d --cost-bps 10 --delay 2
```

```bash
python 09_backtesting_trading/walk_forward_backtest.py --placebo-runs 19
```

Options: `--target {trend_3d,trend_5d,trend_10d,sharpe_signal_10d,market_regime_10d}`,
`--models "Logistic Regression" LightGBM`, `--mode {long_flat,long_short}`,
`--threshold 0.5`, `--cost-bps 5`, `--delay 1`, `--no-pca`, `--n-splits 5`,
`--placebo-runs N`, `--shuffle-labels SEED`. About one minute per run with
all six models; `--placebo-runs N` costs N more runs. Run variants one at a
time: the tree models use every core, and parallel runs thrash.

### Results (AAPL, GOOGL, TSLA; 5 bps; 1-day delay)

`trend_5d`, window 2024-12-11 → 2025-10-03, 19 placebo runs
([report](../16_reports_results/backtests/backtest_trend_5d_long_flat.md)):

| Strategy | Total return | Sharpe | Max DD | Trades | Placebo p |
| --- | ---: | ---: | ---: | ---: | ---: |
| Buy & Hold | 18.1 % | 0.72 | −37.5 % | 3 | — |
| MA20 > MA50 rule | 16.3 % | 0.99 | −17.5 % | 9 | — |
| Logistic Regression | 40.3 % | 1.96 | −12.1 % | 69 | 0.05 |
| Voting Ensemble | 34.2 % | 1.58 | −21.2 % | 82 | 0.05 |
| XGBoost | 25.4 % | 1.19 | −23.9 % | 98 | 0.10 |
| LightGBM | 21.1 % | 0.97 | −25.0 % | 83 | 0.15 |
| Random Forest | 14.3 % | 0.70 | −27.5 % | 73 | 0.55 |
| Stacking Ensemble | −14.1 % | −0.42 | −37.5 % | 29 | 1.00 |

`market_regime_10d`, window 2025-02-12 → 2025-10-03
([report](../16_reports_results/backtests/backtest_market_regime_10d_long_flat.md)):
models reach Sharpe 2.0–2.9 against 1.16 for Buy & Hold and 2.44 for the
plain MA20 > MA50 rule, but placebo p is 0.10–0.35 for every model:
shuffled-label models already average Sharpe 1.0–1.4 over this window.

Robustness of the logistic-regression `trend_5d` strategy (Sharpe): 1.68
with a 2-day delay, 1.59 at 20 bps costs, 1.94 on `trend_3d`, but 0.89 on
`trend_10d`. XGBoost swings from 1.19 to 2.17 with the delay alone.

**Interpretation.** Most of the edge comes from being flat through the
Feb–Apr 2025 sell-off. In a separate set of 20 shuffled-label runs, 2
logistic-regression placebos beat Sharpe 1.96, so across all 39 placebo
runs p ≈ 0.075, before accounting for six models and several targets
being tried. This is a lead worth re-testing on more tickers and years,
not evidence of a tradable edge.

### Outputs

`16_reports_results/backtests/backtest_<target>_<mode>.{md,json,png}`:
metrics table (total and annual return, volatility, Sharpe, max drawdown,
Calmar, exposure, trades, trade and daily win rate, luck-test p), equity
and drawdown curves, and the fold periods.

## Status

**Not applied to live trading.** All backtests are historical
simulations on three tickers over under a year. `walk_forward_backtest.py`
includes transaction costs and an execution delay, but not market impact,
borrowing costs for shorts, or taxes.

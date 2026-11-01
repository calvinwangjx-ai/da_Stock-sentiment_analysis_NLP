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
7. **Luck test.** Each strategy's positions are circularly shifted against
   the returns by every offset ≥ 20 days. This keeps exposure, number of
   trades and holding periods but randomises timing. The p-value is the
   share of shifted copies whose net Sharpe is at least as high.

### Usage

Restore the dataset first (see `07_model_training/classification_models/README.md`), then:

```bash
python 09_backtesting_trading/walk_forward_backtest.py
```

```bash
python 09_backtesting_trading/walk_forward_backtest.py --target trend_3d --cost-bps 10 --delay 2
```

Options: `--target {trend_3d,trend_5d,trend_10d,sharpe_signal_10d,market_regime_10d}`,
`--models "Logistic Regression" LightGBM`, `--mode {long_flat,long_short}`,
`--threshold 0.5`, `--cost-bps 5`, `--delay 1`, `--no-pca`, `--n-splits 5`.
About one minute per run.

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

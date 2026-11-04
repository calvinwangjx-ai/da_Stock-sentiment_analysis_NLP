# 17 — Scripts and Executables

Top-level entry points that orchestrate multi-stage runs.

## Files

| File | Purpose |
| --- | --- |
| `main.py` | Runs the full pipeline |
| `run_data_quality_assessment.py` | Runs data-quality checks in isolation |
| `run_feature_engineering.py` | Runs feature engineering in isolation |
| `run_missing_values_pipeline.py` | Runs missing-value handling in isolation |

## Usage

```bash
python main.py                    # full pipeline
python run_feature_engineering.py # single stage
Note
Individual stages also have their own runnable scripts at the top of
their folders. These entry points are conveniences for running the
whole thing or a specific slice.

## `run_price_pipeline.sh`

Collects ~10 years of Yahoo Finance prices for the 30-stock universe,
builds features, trains and evaluates the traditional ML models
(walk-forward and shuffled), and backtests them with placebo tests:

```bash
bash 17_scripts_executables/run_price_pipeline.sh
```

Settings via environment variables: `START` (default `2015-01-01`),
`PLACEBO_RUNS` (default 19), `BACKTEST_TARGETS` (default
`"trend_5d market_regime_10d"`), `BACKTEST_MODELS` (`|`-separated, default
`"Logistic Regression|LightGBM"`), `PYTHON`. Reports go to
`16_reports_results/extended/`, separate from the 3-ticker reports. Needs
HTTPS access to `query1.finance.yahoo.com`, `query2.finance.yahoo.com` and
`fc.yahoo.com`; no API keys.

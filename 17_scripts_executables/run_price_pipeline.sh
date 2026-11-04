#!/usr/bin/env bash
# End-to-end run on a wide, multi-year price dataset from Yahoo Finance:
#   collect prices -> build features -> train/evaluate models -> backtest.
#
# Needs HTTPS access to query1.finance.yahoo.com, query2.finance.yahoo.com
# and fc.yahoo.com. No API keys.
#
# Usage (from the repository root):
#   bash 17_scripts_executables/run_price_pipeline.sh
#   START=2010-01-01 PLACEBO_RUNS=9 bash 17_scripts_executables/run_price_pipeline.sh
#
# Outputs (reports are kept apart from the 3-ticker reports):
#   14_data_storage/raw/yfinance_daily.csv                        (gitignored)
#   14_data_storage/integrated_datasets/market_features.csv        (gitignored)
#   16_reports_results/extended/classification/traditional_ml_*.{md,json}
#   16_reports_results/extended/backtests/backtest_*.{md,json,png}
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
START=${START:-2015-01-01}
PLACEBO_RUNS=${PLACEBO_RUNS:-19}
BACKTEST_TARGETS=${BACKTEST_TARGETS:-"trend_5d market_regime_10d"}
# Placebo runs retrain the models PLACEBO_RUNS times; keep this list short on large data.
BACKTEST_MODELS=${BACKTEST_MODELS:-"Logistic Regression|LightGBM"}

RAW=14_data_storage/raw/yfinance_daily.csv
DATA=14_data_storage/integrated_datasets/market_features.csv
OUT=16_reports_results/extended
IFS='|' read -r -a MODELS <<< "$BACKTEST_MODELS"

echo "== 1/4 Collect prices (from $START)"
$PY 01_data_collection/collect_prices.py --start "$START" --out "$RAW"

echo "== 2/4 Build features"
$PY 03_feature_engineering/build_market_dataset.py --raw "$RAW" --out "$DATA"

echo "== 3/4 Train and evaluate models"
$PY 07_model_training/classification_models/train_ensemble_models.py \
    --data "$DATA" --out-dir "$OUT/classification"
$PY 07_model_training/classification_models/train_ensemble_models.py \
    --data "$DATA" --out-dir "$OUT/classification" --split random

echo "== 4/4 Backtest"
for target in $BACKTEST_TARGETS; do
    $PY 09_backtesting_trading/walk_forward_backtest.py \
        --data "$DATA" --out-dir "$OUT/backtests" --target "$target" \
        --models "${MODELS[@]}" --placebo-runs "$PLACEBO_RUNS"
done

echo "Done. Reports in $OUT/"

# Classification Models

Traditional machine-learning models for predicting stock movement.

## Files

| File | Purpose |
| --- | --- |
| `train_ensemble_models.py` | Logistic Regression, Random Forest, XGBoost, LightGBM, soft-voting and stacking ensembles on five forward-looking targets, with walk-forward evaluation and naive baselines |
| `train_classification_models.py` | Earlier per-ticker classifiers (expects `final_dataset/…csv`, not in the repo) |
| `train_regression_models.py` | Earlier per-ticker return regressors (expects `final_dataset/…csv`, not in the repo) |
| `classification_stock_news_experiment.py` | Earlier stock + news experiment (expects `integrated_stock_news/…csv`, not in the repo) |

## `train_ensemble_models.py`

### Targets

| Target | Definition (per ticker, daily bars) |
| --- | --- |
| `trend_3d` / `trend_5d` / `trend_10d` | `Close[t+h] > Close[t]` |
| `sharpe_signal_10d` | annualised Sharpe of daily returns `t+1 … t+10` > 1.0 |
| `market_regime_10d` | `MA20 > MA50` at `t+10` (bullish regime ahead) |

### Pipeline

1. Keep daily bars only (the integrated dataset also appends weekly bars),
   sort by ticker and date, drop duplicates.
2. Use the same 1,795 candidate features as `pca_analysis/pca_tsne_analysis.py`.
3. Inside each training fold: drop columns > 50 % missing, median-impute,
   drop constant columns, standard-scale, PCA to 35 components.
4. Fit each model and score the next block of dates.

### Leakage controls

- Walk-forward evaluation (default): 5 expanding-window folds over calendar
  dates, with a gap equal to the label horizon between train and test.
- All preprocessing, including PCA, is fit on the training fold only.
- The stacking ensemble's meta-learner is fit only on *forward* out-of-fold
  predictions inside the training fold, with non-negative weights. (sklearn's
  `StackingClassifier` trains base models on later rows to predict earlier
  ones; on this data that produced ROC-AUC < 0.5.)
- Each target is compared with naive baselines: training-fold majority class,
  always-up, and (for the regime target) persistence.

### Usage

The dataset is gitignored. Restore the copy kept in git history, then run:

```bash
mkdir -p 14_data_storage/integrated_datasets && git cat-file -p e75356b4c5c980fc84a9703b1b5984acc3102045 > 14_data_storage/integrated_datasets/integrated_simple_20251202_213953.csv
```

```bash
pip install -r 18_documentation/requirements.txt
```

```bash
python 07_model_training/classification_models/train_ensemble_models.py
```

```bash
python 07_model_training/classification_models/train_ensemble_models.py --split random
```

Options: `--targets trend_5d market_regime_10d`, `--no-pca`,
`--n-components 35`, `--n-splits 5`, `--seed 42`, `--data <csv>`,
`--out-dir <dir>`. A full run takes about 3–4 minutes on CPU.

`--split random` uses shuffled stratified K-fold. It lets future rows into
training and exists only to show how much that inflates results.

### Outputs

`16_reports_results/classification_reports/traditional_ml_<split>.md` and
`.json` — per-target tables for every model and baseline, plus fold periods.

### Results on the committed dataset (AAPL, GOOGL, TSLA; Oct 2024 – Oct 2025)

| Target | Walk-forward best | Best naive baseline | Shuffled K-fold best |
| --- | ---: | ---: | ---: |
| `trend_3d` | 57.7 % (LogReg) | 54.8 % | 68.7 % (LightGBM) |
| `trend_5d` | 56.5 % (LogReg) | 52.3 % | 70.1 % (LightGBM) |
| `trend_10d` | 53.0 % (LightGBM) | 57.8 % | 78.3 % (XGBoost) |
| `sharpe_signal_10d` | 52.8 % (LightGBM) | 52.0 % | 76.1 % (LightGBM) |
| `market_regime_10d` | 81.1 % (LogReg) | 80.9 % (persistence) | 93.4 % (Voting) |

Under walk-forward evaluation no model clearly beats a naive baseline; the
62.6–70.3 % range quoted in the root README is only reached with shuffled
splits, which leak future information.

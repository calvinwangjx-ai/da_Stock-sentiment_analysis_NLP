# Traditional ML & Ensemble Results (walk-forward)

Generated 2026-11-01 04:01 by `07_model_training/classification_models/train_ensemble_models.py`.

- Data: `14_data_storage/integrated_datasets/integrated_simple_20251202_213953.csv` — 750 daily rows, tickers AAPL, GOOGL, TSLA, 2024-10-04 to 2025-10-03
- Features: 1795 candidate columns → PCA(35) fit inside each training fold
- Evaluation: walk-forward, 5 expanding-window folds over calendar dates with a gap equal to the label horizon; metrics pooled over all test folds

## Best model per target

| Target | Best model | Accuracy | F1 (weighted) | ROC-AUC | Best baseline | Baseline acc. | Edge |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| trend_3d | Logistic Regression | 57.7% | 0.576 | 0.588 | majority of train | 54.8% | +2.9% |
| trend_5d | Logistic Regression | 56.5% | 0.565 | 0.575 | always up | 52.3% | +4.2% |
| trend_10d | LightGBM | 53.0% | 0.530 | 0.549 | majority of train | 57.8% | -4.8% |
| sharpe_signal_10d | LightGBM | 52.8% | 0.527 | 0.556 | always up | 52.0% | +0.8% |
| market_regime_10d | Logistic Regression | 81.1% | 0.811 | 0.891 | persistence | 80.9% | +0.2% |

## All models

### trend_3d (horizon 3d, 615 test rows, 54.8% positive)

Baselines: majority of train 54.8%, always up 54.8%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 57.7% | 0.576 | 0.588 | 0.583 | 57.7% ± 7.0% |
| Voting Ensemble | 55.9% | 0.551 | 0.565 | 0.561 | 55.9% ± 6.6% |
| Random Forest | 55.8% | 0.530 | 0.539 | 0.538 | 55.8% ± 9.5% |
| XGBoost | 55.1% | 0.544 | 0.560 | 0.547 | 55.1% ± 5.7% |
| LightGBM | 55.0% | 0.543 | 0.546 | 0.547 | 55.0% ± 7.6% |
| Stacking Ensemble | 49.1% | 0.471 | 0.439 | 0.550 | 49.1% ± 8.8% |

### trend_5d (horizon 5d, 600 test rows, 52.3% positive)

Baselines: majority of train 49.0%, always up 52.3%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 56.5% | 0.565 | 0.575 | 0.527 | 56.5% ± 5.5% |
| Random Forest | 53.7% | 0.532 | 0.532 | 0.556 | 53.7% ± 8.3% |
| Voting Ensemble | 53.2% | 0.532 | 0.543 | 0.553 | 53.2% ± 3.0% |
| XGBoost | 52.8% | 0.529 | 0.539 | 0.568 | 52.8% ± 4.0% |
| LightGBM | 48.7% | 0.487 | 0.520 | 0.563 | 48.7% ± 5.4% |
| Stacking Ensemble | 40.8% | 0.408 | 0.406 | 0.556 | 40.8% ± 7.1% |

### trend_10d (horizon 10d, 600 test rows, 57.8% positive)

Baselines: majority of train 57.8%, always up 57.8%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 53.0% | 0.530 | 0.549 | 0.586 | 53.0% ± 16.2% |
| Logistic Regression | 51.8% | 0.520 | 0.551 | 0.579 | 51.8% ± 10.0% |
| XGBoost | 51.5% | 0.516 | 0.532 | 0.577 | 51.5% ± 14.7% |
| Voting Ensemble | 51.5% | 0.515 | 0.532 | 0.582 | 51.5% ± 14.6% |
| Random Forest | 50.7% | 0.451 | 0.465 | 0.555 | 50.7% ± 15.3% |
| Stacking Ensemble | 41.5% | 0.419 | 0.387 | 0.575 | 41.5% ± 15.9% |

### sharpe_signal_10d (horizon 10d, 600 test rows, 52.0% positive)

Baselines: majority of train 35.0%, always up 52.0%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 52.8% | 0.527 | 0.556 | 0.588 | 52.8% ± 10.8% |
| Logistic Regression | 51.7% | 0.504 | 0.536 | 0.542 | 51.7% ± 12.4% |
| XGBoost | 51.3% | 0.512 | 0.541 | 0.586 | 51.3% ± 12.7% |
| Voting Ensemble | 51.2% | 0.508 | 0.540 | 0.584 | 51.2% ± 12.1% |
| Random Forest | 46.8% | 0.467 | 0.476 | 0.572 | 46.8% ± 15.1% |
| Stacking Ensemble | 42.0% | 0.409 | 0.400 | 0.581 | 42.0% ± 17.9% |

### market_regime_10d (horizon 10d, 465 test rows, 54.6% positive)

Baselines: majority of train 28.4%, always up 54.6%, persistence 80.9%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 81.1% | 0.811 | 0.891 | 0.792 | 81.1% ± 9.3% |
| Voting Ensemble | 67.1% | 0.671 | 0.767 | 0.830 | 67.1% ± 21.9% |
| LightGBM | 64.5% | 0.645 | 0.723 | 0.794 | 64.5% ± 19.8% |
| XGBoost | 62.6% | 0.627 | 0.696 | 0.809 | 62.6% ± 23.2% |
| Stacking Ensemble | 57.0% | 0.501 | 0.781 | 0.788 | 57.0% ± 28.0% |
| Random Forest | 56.6% | 0.565 | 0.525 | 0.797 | 56.6% ± 32.0% |

## Caveats

- Small sample: three tickers over roughly one year. Adjacent rows share most of
  their label window, so effective sample size is far below the row count and
  differences of a few points are within noise.
- News-sentiment columns are populated on only a handful of rows at the very end
  of the period and are dropped as sparse; no Reddit features are present in this
  dataset. The models are therefore effectively price/volume/technical-only.
- Some `*_scaled` / outlier-flag columns in the integrated dataset were computed
  upstream on the full period, a mild look-ahead this script cannot undo.
- `Edge` compares the best model against the strongest naive baseline. Picking
  the best of six models per target is itself optimistic.
- Pooled ROC-AUC ranks predictions across all test folds together, so a model
  whose scores mostly track the recent base rate (e.g. the stacking ensemble
  when its meta-weights shrink to ~0) can score below 0.5 when base rates flip
  between periods. The fold-mean ROC-AUC isolates within-period ranking skill.

# Traditional ML & Ensemble Results (random)

Generated 2026-11-01 04:05 by `07_model_training/classification_models/train_ensemble_models.py`.

- Data: `14_data_storage/integrated_datasets/integrated_simple_20251202_213953.csv` — 750 daily rows, tickers AAPL, GOOGL, TSLA, 2024-10-04 to 2025-10-03
- Features: 1795 candidate columns → PCA(35) fit inside each training fold
- Evaluation: shuffled stratified 5-fold (NOT time-aware; for comparison only)

## Best model per target

| Target | Best model | Accuracy | F1 (weighted) | ROC-AUC | Best baseline | Baseline acc. | Edge |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| trend_3d | LightGBM | 68.7% | 0.683 | 0.715 | majority of train | 56.4% | +12.3% |
| trend_5d | LightGBM | 70.1% | 0.699 | 0.745 | majority of train | 55.1% | +15.0% |
| trend_10d | XGBoost | 78.3% | 0.778 | 0.846 | majority of train | 60.8% | +17.5% |
| sharpe_signal_10d | LightGBM | 76.1% | 0.759 | 0.840 | majority of train | 55.3% | +20.8% |
| market_regime_10d | Voting Ensemble | 93.4% | 0.933 | 0.979 | persistence | 80.8% | +12.6% |

## All models

### trend_3d (horizon 3d, 741 test rows, 56.4% positive)

Baselines: majority of train 56.4%, always up 56.4%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 68.7% | 0.683 | 0.715 | 0.716 | 68.7% ± 2.8% |
| XGBoost | 67.1% | 0.669 | 0.703 | 0.702 | 67.1% ± 2.2% |
| Voting Ensemble | 66.4% | 0.660 | 0.707 | 0.710 | 66.4% ± 3.0% |
| Random Forest | 63.6% | 0.615 | 0.683 | 0.686 | 63.6% ± 4.1% |
| Stacking Ensemble | 61.8% | 0.591 | 0.672 | 0.704 | 61.8% ± 3.4% |
| Logistic Regression | 60.2% | 0.596 | 0.643 | 0.650 | 60.2% ± 5.3% |

### trend_5d (horizon 5d, 735 test rows, 55.1% positive)

Baselines: majority of train 55.1%, always up 55.1%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 70.1% | 0.699 | 0.745 | 0.746 | 70.1% ± 2.0% |
| Voting Ensemble | 69.3% | 0.689 | 0.735 | 0.737 | 69.3% ± 3.0% |
| XGBoost | 67.8% | 0.675 | 0.733 | 0.735 | 67.8% ± 3.1% |
| Random Forest | 66.0% | 0.646 | 0.723 | 0.725 | 66.0% ± 3.0% |
| Logistic Regression | 62.2% | 0.616 | 0.654 | 0.657 | 62.2% ± 3.9% |
| Stacking Ensemble | 59.6% | 0.531 | 0.692 | 0.746 | 59.6% ± 3.9% |

### trend_10d (horizon 10d, 720 test rows, 60.8% positive)

Baselines: majority of train 60.8%, always up 60.8%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| XGBoost | 78.3% | 0.778 | 0.846 | 0.848 | 78.3% ± 3.0% |
| Voting Ensemble | 78.3% | 0.776 | 0.846 | 0.849 | 78.3% ± 2.8% |
| LightGBM | 78.2% | 0.777 | 0.854 | 0.856 | 78.2% ± 2.5% |
| Random Forest | 72.1% | 0.695 | 0.820 | 0.823 | 72.1% ± 4.2% |
| Logistic Regression | 70.8% | 0.698 | 0.759 | 0.756 | 70.8% ± 4.3% |
| Stacking Ensemble | 64.3% | 0.543 | 0.751 | 0.785 | 64.3% ± 5.1% |

### sharpe_signal_10d (horizon 10d, 720 test rows, 55.3% positive)

Baselines: majority of train 55.3%, always up 55.3%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM | 76.1% | 0.759 | 0.840 | 0.843 | 76.1% ± 2.4% |
| Voting Ensemble | 74.9% | 0.746 | 0.821 | 0.824 | 74.9% ± 2.8% |
| Random Forest | 73.8% | 0.729 | 0.819 | 0.823 | 73.8% ± 4.9% |
| XGBoost | 72.8% | 0.727 | 0.809 | 0.811 | 72.8% ± 3.3% |
| Logistic Regression | 69.3% | 0.688 | 0.730 | 0.730 | 69.3% ± 4.4% |
| Stacking Ensemble | 65.1% | 0.636 | 0.687 | 0.695 | 65.1% ± 8.7% |

### market_regime_10d (horizon 10d, 573 test rows, 57.9% positive)

Baselines: majority of train 57.9%, always up 57.9%, persistence 80.8%

| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) | Fold accuracy mean ± std |
| --- | ---: | ---: | ---: | ---: | ---: |
| Voting Ensemble | 93.4% | 0.933 | 0.979 | 0.980 | 93.4% ± 0.9% |
| Logistic Regression | 92.3% | 0.923 | 0.978 | 0.977 | 92.3% ± 3.0% |
| LightGBM | 92.3% | 0.923 | 0.977 | 0.977 | 92.3% ± 1.2% |
| XGBoost | 92.1% | 0.921 | 0.970 | 0.972 | 92.1% ± 1.0% |
| Stacking Ensemble | 89.2% | 0.890 | 0.970 | 0.973 | 89.2% ± 1.3% |
| Random Forest | 86.9% | 0.867 | 0.950 | 0.953 | 86.9% ± 1.6% |

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
- **This run uses shuffled K-fold**: training folds contain rows from after
  the test rows. The numbers are inflated and are shown only for comparison
  with the walk-forward report.

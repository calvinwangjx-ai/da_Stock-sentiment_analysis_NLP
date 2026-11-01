# 16 — Reports and Results

Human-readable outputs from the pipeline: classification reports,
metric summaries, and comparison tables.

## Contents

| Folder | Contains |
| --- | --- |
| `classification_reports/` | Per-model classification reports (precision, recall, F1) |
| `classification_reports/traditional_ml_walk_forward.{md,json}` | XGBoost / LightGBM / RF / LogReg / voting / stacking on five targets, walk-forward evaluation (from `07_model_training/classification_models/train_ensemble_models.py`) |
| `classification_reports/traditional_ml_random.{md,json}` | Same models with shuffled K-fold, for comparison only (leaks future data) |

## Related

- Model training: `07_model_training/`
- Evaluation code: `08_model_evaluation/`
- Experiment artifacts: `13_experiments_tests/`

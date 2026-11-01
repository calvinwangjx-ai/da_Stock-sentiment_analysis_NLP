"""
Traditional ML + ensemble training for stock-movement classification.

Trains Logistic Regression, Random Forest, XGBoost, LightGBM, a soft-voting
ensemble and a stacking ensemble on five forward-looking targets:

    trend_3d            Close[t+3]  > Close[t]
    trend_5d            Close[t+5]  > Close[t]
    trend_10d           Close[t+10] > Close[t]
    sharpe_signal_10d   annualised Sharpe of daily returns t+1..t+10 > 1.0
    market_regime_10d   (MA20 > MA50) at t+10   (bullish regime ahead)

Leakage controls
----------------
* Only daily bars are used (the integrated dataset also appends weekly bars),
  sorted by ticker and date, duplicates removed.
* Every target looks strictly forward; features are what is known at close t.
* Imputation, sparse-column dropping, scaling and PCA are fit inside each
  training fold (sklearn Pipeline), never on test data.
* Default evaluation is walk-forward over calendar dates (expanding window,
  sklearn TimeSeriesSplit on unique dates) with a gap of `horizon` dates
  between train and test so overlapping label windows cannot leak.
* The stacking ensemble's meta-learner is fit only on forward (time-ordered)
  out-of-fold predictions inside the training fold, with non-negative weights.
* Every model is compared with naive baselines: the training fold's majority
  class, always predicting 1 ("up" / "bullish"), and for market_regime_10d a
  persistence baseline (today's regime still holds in 10 days).

`--split random` reproduces the shuffled K-fold evaluation used elsewhere in
this repository. It lets future rows into training and is reported only to
show how much that inflates the numbers.

Usage
-----
    python 07_model_training/classification_models/train_ensemble_models.py
    python 07_model_training/classification_models/train_ensemble_models.py --split random
    python 07_model_training/classification_models/train_ensemble_models.py --no-pca --targets trend_5d
"""

import argparse
import json
import os
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - reported at runtime
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover - reported at runtime
    LGBMClassifier = None

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "14_data_storage" / "integrated_datasets" / "integrated_simple_20251202_213953.csv"
DEFAULT_OUT = REPO_ROOT / "16_reports_results" / "classification_reports"

# Same name filter as 07_model_training/pca_analysis/pca_tsne_analysis.py,
# which yields the 1,795 features quoted in the README.
TARGET_LIKE = ["close", "price", "target", "label", "return", "next_", "future_", "_target"]

TARGETS = {
    "trend_3d": 3,
    "trend_5d": 5,
    "trend_10d": 10,
    "sharpe_signal_10d": 10,
    "market_regime_10d": 10,
}
SHARPE_THRESHOLD = 1.0


# --------------------------------------------------------------------------- data
def load_daily(path):
    """Load the integrated dataset, keep daily bars only, sorted per ticker."""
    df = pd.read_csv(path, low_memory=False)
    if "interval" in df.columns:
        df = df[df["interval"] == "1d"]
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df = df.drop_duplicates(subset=["ticker", "Date"]).sort_values(["ticker", "Date"])
    df = df.reset_index(drop=True)
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace([np.inf, -np.inf], np.nan)
    return df


def select_features(df):
    num = df.select_dtypes(include=[np.number]).columns
    return [c for c in num if not any(t in c.lower() for t in TARGET_LIKE)]


def build_targets(df):
    """Forward-looking binary targets (NaN where the future is unknown)."""
    close = df["Close"]
    g = close.groupby(df["ticker"])
    out = {}
    for h in (3, 5, 10):
        fut = g.shift(-h)
        out[f"trend_{h}d"] = (fut > close).astype(float).where(fut.notna())

    daily_ret = g.pct_change()
    fwd = pd.concat(
        [daily_ret.groupby(df["ticker"]).shift(-k) for k in range(1, 11)], axis=1
    )
    sharpe = fwd.mean(axis=1) / fwd.std(axis=1, ddof=1) * np.sqrt(252)
    complete = fwd.notna().all(axis=1)
    out["sharpe_signal_10d"] = (sharpe > SHARPE_THRESHOLD).astype(float).where(complete)

    ma20 = g.transform(lambda s: s.rolling(20).mean())
    ma50 = g.transform(lambda s: s.rolling(50).mean())
    regime = (ma20 > ma50).astype(float).where(ma50.notna())
    out["market_regime_10d"] = regime.groupby(df["ticker"]).shift(-10)
    return out, regime


# ------------------------------------------------------------------------ models
class DropSparseColumns(BaseEstimator, TransformerMixin):
    """Drop columns whose missing share in the *training* data exceeds a limit."""

    def __init__(self, max_missing=0.5):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.keep_ = np.isnan(X).mean(axis=0) <= self.max_missing
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


class NonNegativeLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression on base-model logits with weights constrained >= 0.

    Unconstrained meta-learners fitted on a few hundred noisy out-of-fold
    predictions readily assign negative weights (i.e. bet *against* a base
    model), which does not generalise. Non-negative weights are the standard
    remedy for stacked generalisation (Breiman, 1996).
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    @staticmethod
    def _logit(P):
        P = np.clip(np.asarray(P, dtype=float), 1e-6, 1 - 1e-6)
        return np.log(P / (1 - P))

    def fit(self, X, y):
        Z, y = self._logit(X), np.asarray(y, dtype=float)
        n, k = Z.shape
        self.classes_ = np.array([0, 1])

        def loss(theta):
            w, b = theta[:k], theta[k]
            s = Z @ w + b
            p = 1.0 / (1.0 + np.exp(-s))
            nll = np.mean(np.logaddexp(0, s) - y * s)
            grad_s = (p - y) / n
            g = np.concatenate([Z.T @ grad_s + self.alpha * w / n, [grad_s.sum()]])
            return nll + 0.5 * self.alpha * w @ w / n, g

        theta0 = np.concatenate([np.full(k, 1.0 / k), [0.0]])
        bounds = [(0, None)] * k + [(None, None)]
        opt = minimize(loss, theta0, jac=True, method="L-BFGS-B", bounds=bounds)
        self.coef_, self.intercept_ = opt.x[:k][None, :], opt.x[k:]
        return self

    def predict_proba(self, X):
        s = self._logit(X) @ self.coef_[0] + self.intercept_[0]
        p = 1.0 / (1.0 + np.exp(-s))
        return np.column_stack([1 - p, p])


class TimeSeriesStackingClassifier(BaseEstimator, ClassifierMixin):
    """Stacking whose meta-learner only sees *forward* out-of-fold predictions.

    sklearn's StackingClassifier needs a CV that partitions the data, i.e. base
    models are trained on later rows to predict earlier ones. On drifting price
    data that teaches the meta-learner the wrong weights (ROC-AUC < 0.5 in
    practice). Here rows must arrive in chronological order; base models are
    fit on an expanding window and predict the next block, the meta-learner is
    fit on those predictions, and the base models are then refit on all rows.
    """

    def __init__(self, estimators, final_estimator, n_splits=4):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.n_splits = n_splits

    def _meta_features(self, models, X):
        return np.column_stack([m.predict_proba(X)[:, 1] for m in models])

    def fit(self, X, y):
        X, y = np.asarray(X), np.asarray(y)
        self.classes_ = np.unique(y)
        oof = np.full((len(y), len(self.estimators)), np.nan)
        for tr, te in TimeSeriesSplit(n_splits=self.n_splits).split(X):
            if len(np.unique(y[tr])) < 2:
                continue
            fitted = [clone(e).fit(X[tr], y[tr]) for _, e in self.estimators]
            oof[te] = self._meta_features(fitted, X[te])
        have = ~np.isnan(oof).any(axis=1)
        self.estimators_ = [clone(e).fit(X, y) for _, e in self.estimators]
        if have.sum() == 0 or len(np.unique(y[have])) < 2:
            # Not enough forward predictions to learn weights: fall back to averaging.
            self.final_estimator_ = None
        else:
            self.final_estimator_ = clone(self.final_estimator).fit(oof[have], y[have])
        return self

    def predict_proba(self, X):
        Z = self._meta_features(self.estimators_, np.asarray(X))
        if self.final_estimator_ is None:
            p = Z.mean(axis=1)
            return np.column_stack([1 - p, p])
        return self.final_estimator_.predict_proba(Z)

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]


def preprocessing(use_pca, n_components, seed):
    steps = [
        ("drop_sparse", DropSparseColumns(0.5)),
        ("impute", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold(0.0)),
        ("scale", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=n_components, random_state=seed)))
    return steps


def base_estimators(seed):
    est = {
        "Logistic Regression": LogisticRegression(C=0.1, max_iter=5000),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=seed, n_jobs=-1
        ),
    }
    if XGBClassifier is not None:
        est["XGBoost"] = XGBClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1,
        )
    if LGBMClassifier is not None:
        est["LightGBM"] = LGBMClassifier(
            n_estimators=300, num_leaves=15, learning_rate=0.05, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, min_child_samples=10,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
    return est


def build_models(use_pca, n_components, seed):
    base = base_estimators(seed)
    members = [(k.lower().replace(" ", "_"), v) for k, v in base.items()]
    trees = [m for m in members if m[0] != "logistic_regression"]
    models = dict(base)
    models["Voting Ensemble"] = VotingClassifier(
        estimators=[(n, clone(e)) for n, e in members], voting="soft"
    )
    # Meta-learner sees only forward (time-ordered) out-of-fold predictions made
    # inside the training fold, so the test fold is never seen.
    models["Stacking Ensemble"] = TimeSeriesStackingClassifier(
        estimators=[(n, clone(e)) for n, e in trees],
        final_estimator=NonNegativeLogisticRegression(alpha=1.0),
        n_splits=4,
    )
    return {
        name: Pipeline(preprocessing(use_pca, n_components, seed) + [("model", est)])
        for name, est in models.items()
    }


# ------------------------------------------------------------------------ splits
def walk_forward_splits(dates, n_splits, gap):
    """Expanding-window splits over unique dates with a `gap`-date embargo."""
    codes, uniques = pd.factorize(dates, sort=True)
    tss = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    for tr_d, te_d in tss.split(np.arange(len(uniques))):
        yield np.flatnonzero(np.isin(codes, tr_d)), np.flatnonzero(np.isin(codes, te_d))


def random_splits(y, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(y)), y)


# -------------------------------------------------------------------- evaluation
def metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    auc = roc_auc_score(y_true, prob) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1_weighted": float(f1_score(y_true, pred, average="weighted")),
        "roc_auc": float(auc),
    }


def evaluate_target(name, X, y, dates, persistence, models, split, n_splits, seed):
    horizon = TARGETS[name]
    if split == "walk_forward":
        folds = list(walk_forward_splits(dates, n_splits, gap=horizon))
    else:
        folds = list(random_splits(y, n_splits, seed))

    test_idx = np.concatenate([te for _, te in folds])
    y_test = y[test_idx]
    probs = {m: np.empty(len(test_idx)) for m in models}
    fold_acc = {m: [] for m in models}
    fold_auc = {m: [] for m in models}
    base_pred = np.empty(len(test_idx))
    fold_meta = []

    pos = 0
    for k, (tr, te) in enumerate(folds):
        sl = slice(pos, pos + len(te))
        majority = int(y[tr].mean() >= 0.5)
        base_pred[sl] = majority
        fold_meta.append({
            "fold": k + 1,
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
            "train_period": [str(dates[tr].min().date()), str(dates[tr].max().date())],
            "test_period": [str(dates[te].min().date()), str(dates[te].max().date())],
            "test_positive_rate": float(y[te].mean()),
        })
        for m, pipe in models.items():
            est = clone(pipe)
            if len(np.unique(y[tr])) < 2:
                p = np.full(len(te), float(majority))
            else:
                est.fit(X[tr], y[tr])
                p = est.predict_proba(X[te])[:, 1]
            probs[m][sl] = p
            fold_acc[m].append(float(accuracy_score(y[te], (p >= 0.5).astype(int))))
            if len(np.unique(y[te])) > 1:
                fold_auc[m].append(float(roc_auc_score(y[te], p)))
        pos += len(te)

    res = {
        "horizon_days": horizon,
        "n_test": int(len(test_idx)),
        "test_positive_rate": float(y_test.mean()),
        "folds": fold_meta,
        "baselines": {
            "majority_of_train": {
                "accuracy": float(accuracy_score(y_test, base_pred)),
                "f1_weighted": float(f1_score(y_test, base_pred, average="weighted")),
            },
            "always_up": {
                "accuracy": float(y_test.mean()),
                "f1_weighted": float(f1_score(y_test, np.ones_like(y_test), average="weighted")),
            },
        },
        "models": {},
    }
    if persistence is not None:
        pp = persistence[test_idx]
        res["baselines"]["persistence"] = {
            "accuracy": float(accuracy_score(y_test, pp)),
            "f1_weighted": float(f1_score(y_test, pp, average="weighted")),
        }
    for m in models:
        r = metrics(y_test, probs[m])
        r["fold_accuracy_mean"] = float(np.mean(fold_acc[m]))
        r["fold_accuracy_std"] = float(np.std(fold_acc[m]))
        r["fold_roc_auc_mean"] = float(np.mean(fold_auc[m])) if fold_auc[m] else float("nan")
        res["models"][m] = r
    return res


# ------------------------------------------------------------------------ report
def best_baseline(res):
    return max(res["baselines"].items(), key=lambda kv: kv[1]["accuracy"])


def write_report(results, meta, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"traditional_ml_{meta['split']}{'' if meta['pca'] else '_nopca'}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    with open(json_path, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    lines = [
        f"# Traditional ML & Ensemble Results ({meta['split'].replace('_', '-')})",
        "",
        f"Generated {meta['generated']} by "
        "`07_model_training/classification_models/train_ensemble_models.py`.",
        "",
        f"- Data: `{meta['data']}` — {meta['rows']} daily rows, tickers "
        f"{', '.join(meta['tickers'])}, {meta['date_range'][0]} to {meta['date_range'][1]}",
        f"- Features: {meta['n_features']} candidate columns"
        + (f" → PCA({meta['n_components']}) fit inside each training fold" if meta["pca"] else " (no PCA)"),
        f"- Evaluation: {meta['split_description']}",
        "",
        "## Best model per target",
        "",
        "| Target | Best model | Accuracy | F1 (weighted) | ROC-AUC | Best baseline | Baseline acc. | Edge |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for t, res in results.items():
        bm, br = max(res["models"].items(), key=lambda kv: kv[1]["accuracy"])
        bn, bb = best_baseline(res)
        edge = br["accuracy"] - bb["accuracy"]
        lines.append(
            f"| {t} | {bm} | {br['accuracy']:.1%} | {br['f1_weighted']:.3f} | {br['roc_auc']:.3f} "
            f"| {bn.replace('_', ' ')} | {bb['accuracy']:.1%} | {edge:+.1%} |"
        )
    lines += ["", "## All models", ""]
    for t, res in results.items():
        bl = ", ".join(f"{k.replace('_', ' ')} {v['accuracy']:.1%}" for k, v in res["baselines"].items())
        lines += [
            f"### {t} (horizon {res['horizon_days']}d, {res['n_test']} test rows, "
            f"{res['test_positive_rate']:.1%} positive)",
            "",
            f"Baselines: {bl}",
            "",
            "| Model | Accuracy | F1 (weighted) | ROC-AUC (pooled) | ROC-AUC (fold mean) "
            "| Fold accuracy mean ± std |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for m, r in sorted(res["models"].items(), key=lambda kv: -kv[1]["accuracy"]):
            lines.append(
                f"| {m} | {r['accuracy']:.1%} | {r['f1_weighted']:.3f} | {r['roc_auc']:.3f} "
                f"| {r['fold_roc_auc_mean']:.3f} "
                f"| {r['fold_accuracy_mean']:.1%} ± {r['fold_accuracy_std']:.1%} |"
            )
        lines.append("")
    lines += [
        "## Caveats",
        "",
        "- Small sample: three tickers over roughly one year. Adjacent rows share most of",
        "  their label window, so effective sample size is far below the row count and",
        "  differences of a few points are within noise.",
        "- News-sentiment columns are populated on only a handful of rows at the very end",
        "  of the period and are dropped as sparse; no Reddit features are present in this",
        "  dataset. The models are therefore effectively price/volume/technical-only.",
        "- Some `*_scaled` / outlier-flag columns in the integrated dataset were computed",
        "  upstream on the full period, a mild look-ahead this script cannot undo.",
        "- `Edge` compares the best model against the strongest naive baseline. Picking",
        "  the best of six models per target is itself optimistic.",
        "- Pooled ROC-AUC ranks predictions across all test folds together, so a model",
        "  whose scores mostly track the recent base rate (e.g. the stacking ensemble",
        "  when its meta-weights shrink to ~0) can score below 0.5 when base rates flip",
        "  between periods. The fold-mean ROC-AUC isolates within-period ranking skill.",
    ]
    if meta["split"] == "random":
        lines += [
            "- **This run uses shuffled K-fold**: training folds contain rows from after",
            "  the test rows. The numbers are inflated and are shown only for comparison",
            "  with the walk-forward report.",
        ]
    md_path.write_text("\n".join(lines) + "\n")
    return md_path, json_path


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DEFAULT_DATA), help="integrated dataset CSV")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT), help="where to write the .md/.json report")
    ap.add_argument("--split", choices=["walk_forward", "random"], default="walk_forward")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-components", type=int, default=35)
    ap.add_argument("--no-pca", action="store_true", help="train on scaled raw features instead of PCA")
    ap.add_argument("--targets", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if XGBClassifier is None or LGBMClassifier is None:
        missing = [n for n, c in (("xgboost", XGBClassifier), ("lightgbm", LGBMClassifier)) if c is None]
        print(f"WARNING: {', '.join(missing)} not installed; those models (and ensemble members) are skipped.")

    if not os.path.exists(args.data):
        raise SystemExit(
            f"Dataset not found: {args.data}\n"
            "It is gitignored. Regenerate it with stages 01-06, or restore the copy from "
            "git history:\n  mkdir -p 14_data_storage/integrated_datasets && git cat-file -p "
            "e75356b4c5c980fc84a9703b1b5984acc3102045 > "
            "14_data_storage/integrated_datasets/integrated_simple_20251202_213953.csv"
        )

    df = load_daily(args.data)
    feats = select_features(df)
    targets, regime_now = build_targets(df)
    models = build_models(not args.no_pca, args.n_components, args.seed)
    print(f"Loaded {len(df)} daily rows, {df['ticker'].nunique()} tickers, {len(feats)} candidate features")
    print(f"Models: {', '.join(models)}")
    print(f"Split: {args.split} ({args.n_splits} folds)\n")

    # Targets were built per ticker; rows are now put in chronological order so
    # that row order inside any training fold is time order (needed by the
    # time-series stacking ensemble).
    order = df.sort_values(["Date", "ticker"], kind="stable").index

    results = {}
    for name in args.targets:
        t = targets[name]
        ok = t.notna()
        if name == "market_regime_10d":
            ok &= regime_now.notna()
        idx = order[ok.loc[order].to_numpy()]
        sub = df.loc[idx]
        X = sub[feats].to_numpy(dtype=float)
        y = t.loc[idx].astype(int).to_numpy()
        dates = pd.DatetimeIndex(sub["Date"])
        persistence = None
        if name == "market_regime_10d":
            persistence = regime_now.loc[idx].astype(int).to_numpy()
        print(f"▶ {name}: {len(y)} rows, {y.mean():.1%} positive")
        res = evaluate_target(name, X, y, dates, persistence, models, args.split, args.n_splits, args.seed)
        results[name] = res
        bl = ", ".join(f"{k} {v['accuracy']:.3f}" for k, v in res["baselines"].items())
        print(f"   baselines: {bl}")
        for m, r in sorted(res["models"].items(), key=lambda kv: -kv[1]["accuracy"]):
            print(f"   {m:20s} acc {r['accuracy']:.3f}  f1 {r['f1_weighted']:.3f}  auc {r['roc_auc']:.3f}")
        print()

    split_desc = (
        f"walk-forward, {args.n_splits} expanding-window folds over calendar dates with a "
        "gap equal to the label horizon; metrics pooled over all test folds"
        if args.split == "walk_forward"
        else f"shuffled stratified {args.n_splits}-fold (NOT time-aware; for comparison only)"
    )
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data": os.path.relpath(args.data, REPO_ROOT) if str(args.data).startswith(str(REPO_ROOT)) else args.data,
        "rows": int(len(df)),
        "tickers": sorted(df["ticker"].unique().tolist()),
        "date_range": [str(df["Date"].min().date()), str(df["Date"].max().date())],
        "n_features": len(feats),
        "pca": not args.no_pca,
        "n_components": args.n_components,
        "split": args.split,
        "split_description": split_desc,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "sharpe_threshold": SHARPE_THRESHOLD,
    }
    md_path, json_path = write_report(results, meta, Path(args.out_dir))
    print(f"Report: {md_path}\nJSON:   {json_path}")


if __name__ == "__main__":
    main()

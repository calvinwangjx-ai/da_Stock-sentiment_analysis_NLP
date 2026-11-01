"""
Walk-forward backtest of the trained classifiers on real prices.

Unlike `backtesting_framework.py` (a demo on a simulated random-walk price
with random BUY/SELL/HOLD signals), this script trades the actual model
predictions from `07_model_training/classification_models/train_ensemble_models.py`
on the actual AAPL / GOOGL / TSLA closes in the integrated dataset.

How it works
------------
1. Walk-forward: calendar dates are split into expanding-window folds
   (sklearn TimeSeriesSplit, gap = label horizon). Each model is trained on
   labelled rows from the training dates only and predicts every row in the
   following test block, so every signal is out-of-sample.
2. Signal: at the close of day t a position is chosen from the predicted
   probability (long if p > threshold, else flat; or long/short).
3. Execution: the position is entered `--delay` closes later (default 1:
   you cannot trade at the very close used to compute the features) and
   earns close-to-close returns from then on.
4. Costs: `--cost-bps` per unit of turnover, one way (covers commission and
   slippage); a flat->long->flat round trip pays it twice.
5. Portfolio: equal weight across tickers, rebalanced daily (rebalancing
   costs between tickers are ignored; position-change costs are not).
6. Benchmarks over the identical window: buy & hold (same costs), and a
   rule-based MA20 > MA50 trend-following strategy with the same delay/costs.
7. Placebo test (`--placebo-runs N`, the one to trust): every model is
   retrained N times on randomly permuted labels and traded the same way;
   the p-value is the share of those no-information strategies whose net
   Sharpe matches or beats the real one.
8. Shift test (always on, optimistic): each position series is circularly
   shifted against the returns by every offset of at least 20 days, which
   keeps exposure, trades and holding periods but randomises timing. On this
   data, models trained on shuffled labels pass it (p < 0.05) about 20% of
   the time, because sitting out one big sell-off dominates the window.

Usage
-----
    python 09_backtesting_trading/walk_forward_backtest.py
    python 09_backtesting_trading/walk_forward_backtest.py --target trend_3d --cost-bps 10
    python 09_backtesting_trading/walk_forward_backtest.py --target market_regime_10d --no-pca
    python 09_backtesting_trading/walk_forward_backtest.py --mode long_short --threshold 0.55
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "07_model_training" / "classification_models"))
import train_ensemble_models as tem  # noqa: E402

warnings.filterwarnings("ignore")
TRADING_DAYS = 252
DEFAULT_OUT = REPO_ROOT / "16_reports_results" / "backtests"
MIN_SHIFT = 20


# ------------------------------------------------------------ out-of-sample signals
def oos_probabilities(df, feats, label, horizon, models, n_splits):
    """Walk-forward out-of-sample P(label=1) for every row in the test blocks.

    `df` must be in (Date, ticker) order. Folds are defined on all dates, so
    the most recent rows (whose labels are not yet known) still get a
    prediction; training uses only labelled rows from the training dates.
    """
    codes, uniques = pd.factorize(pd.DatetimeIndex(df["Date"]), sort=True)
    X = df[feats].to_numpy(dtype=float)
    y = label.to_numpy()
    labelled = ~np.isnan(y)
    probs = {m: np.full(len(df), np.nan) for m in models}
    folds = []
    tss = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    for tr_d, te_d in tss.split(np.arange(len(uniques))):
        tr = np.flatnonzero(np.isin(codes, tr_d) & labelled)
        te = np.flatnonzero(np.isin(codes, te_d))
        ytr = y[tr].astype(int)
        folds.append({
            "train_period": [str(uniques[tr_d[0]].date()), str(uniques[tr_d[-1]].date())],
            "test_period": [str(uniques[te_d[0]].date()), str(uniques[te_d[-1]].date())],
            "train_rows": int(len(tr)),
            "test_rows": int(len(te)),
        })
        for m, pipe in models.items():
            if len(np.unique(ytr)) < 2:
                probs[m][te] = float(ytr.mean())
                continue
            est = clone(pipe).fit(X[tr], ytr)
            probs[m][te] = est.predict_proba(X[te])[:, 1]
    return probs, folds


# --------------------------------------------------------------------- simulation
def signals_to_positions(p, threshold, mode):
    if mode == "long_short":
        pos = np.where(p > threshold, 1.0, -1.0)
    else:
        pos = np.where(p > threshold, 1.0, 0.0)
    return np.where(np.isnan(p), 0.0, pos)


def simulate(decided, returns, cost, delay):
    """decided: positions chosen at each close (dates x tickers).
    Returns per-ticker held positions and net daily returns."""
    held = decided.shift(1 + delay).fillna(0.0)
    turnover = held.diff().abs()
    turnover.iloc[0] = held.iloc[0].abs()
    net = held * returns - cost * turnover
    return held, net


def max_drawdown(r):
    eq = (1 + r).cumprod()
    return float((eq / eq.cummax() - 1).min())


def trade_returns(held, net):
    """Compound net returns over each contiguous run of a non-zero position."""
    out = []
    for col in held.columns:
        h, r = held[col].to_numpy(), net[col].to_numpy()
        start = None
        for i in range(len(h) + 1):
            cur = h[i] if i < len(h) else 0.0
            prev = h[i - 1] if i > 0 else 0.0
            if start is not None and cur != prev:
                out.append(float(np.prod(1 + r[start:i]) - 1))
                start = None
            if cur != 0 and cur != prev:
                start = i
    return out


def sharpe(r):
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0


def performance(held, net):
    port = net.mean(axis=1)
    n = len(port)
    total = float((1 + port).prod() - 1)
    downside = port[port < 0].std(ddof=1)
    trades = trade_returns(held, net)
    in_mkt = held.to_numpy() != 0
    mdd = max_drawdown(port)
    ann = float((1 + total) ** (TRADING_DAYS / n) - 1) if n else 0.0
    return {
        "total_return": total,
        "annual_return": ann,
        "annual_volatility": float(port.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "sharpe": sharpe(port),
        "sortino": float(port.mean() / downside * np.sqrt(TRADING_DAYS)) if downside > 0 else float("nan"),
        "max_drawdown": mdd,
        "calmar": float(ann / abs(mdd)) if mdd < 0 else float("nan"),
        "exposure": float(in_mkt.mean()),
        "daily_win_rate": float((net.to_numpy()[in_mkt] > 0).mean()) if in_mkt.any() else float("nan"),
        "n_trades": len(trades),
        "trade_win_rate": float(np.mean([t > 0 for t in trades])) if trades else float("nan"),
        "avg_trade_return": float(np.mean(trades)) if trades else float("nan"),
        "turnover_per_year": float(held.diff().abs().sum().sum() / held.shape[1] / n * TRADING_DAYS),
    }


def shift_test(held, returns, cost, actual_sharpe):
    """Circular-shift null distribution of net portfolio Sharpe."""
    n = len(held)
    H = held.to_numpy()
    R = returns.to_numpy()
    null = []
    for k in range(MIN_SHIFT, n - MIN_SHIFT + 1):
        h = np.roll(H, k, axis=0)
        turn = np.abs(np.diff(h, axis=0, prepend=np.zeros((1, h.shape[1]))))
        port = (h * R - cost * turn).mean(axis=1)
        sd = port.std(ddof=1)
        null.append(port.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else 0.0)
    null = np.array(null)
    if len(null) == 0:
        return {"p_value": float("nan"), "n_shifts": 0}
    return {
        "p_value": float((np.sum(null >= actual_sharpe - 1e-12) + 1) / (len(null) + 1)),
        "n_shifts": int(len(null)),
        "null_sharpe_median": float(np.median(null)),
        "null_sharpe_95th": float(np.percentile(null, 95)),
    }


# ------------------------------------------------------------------------ report
def plot(curves, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.3]})
    for name, r in curves.items():
        eq = (1 + r).cumprod()
        style = {"color": "black", "lw": 2.2} if name == "Buy & Hold" else (
            {"color": "gray", "lw": 1.6, "ls": "--"} if name.startswith("MA20") else {"lw": 1.3})
        ax1.plot(eq.index, eq.values, label=name, **style)
        ax2.plot(eq.index, (eq / eq.cummax() - 1).values * 100, **style)
    ax1.axhline(1.0, color="lightgray", lw=0.8)
    ax1.set_ylabel("Growth of $1 (net of costs)")
    ax1.set_title(title)
    ax1.legend(fontsize=8, ncol=2, frameon=False)
    ax2.set_ylabel("Drawdown %")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def write_report(rows, meta, folds, out_dir, stem):
    md = out_dir / f"{stem}.md"
    js = out_dir / f"{stem}.json"
    with open(js, "w") as f:
        json.dump({"meta": meta, "folds": folds, "strategies": rows}, f, indent=2)

    def pct(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1%}"

    def num(x, d=2):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"

    lines = [
        f"# Walk-Forward Backtest — `{meta['target']}`",
        "",
        f"Generated {meta['generated']} by `09_backtesting_trading/walk_forward_backtest.py`.",
        "",
        f"- Prices: {', '.join(meta['tickers'])} daily closes, equal-weighted portfolio",
        f"- Window: {meta['window'][0]} to {meta['window'][1]} ({meta['n_days']} trading days, "
        "all signals out-of-sample)",
        f"- Signal: {meta['mode'].replace('_', '/')} when P({meta['target']}) > {meta['threshold']}; "
        f"executed {meta['delay']} close(s) after the signal",
        f"- Costs: {meta['cost_bps']} bps per unit turnover, one way",
        f"- Features: {'PCA(%d) fit per fold' % meta['n_components'] if meta['pca'] else 'scaled raw features'}",
        *([f"- **PLACEBO RUN**: training labels randomly permuted (seed {meta['shuffle_labels']}); "
           "model strategies carry no information"] if meta.get("shuffle_labels") is not None else []),
        "",
        f"![equity curves]({meta['plot']})",
        "",
        "| Strategy | Total return | Ann. return | Ann. vol | Sharpe | Max DD | Calmar "
        "| Exposure | Trades | Trade win rate | Daily win rate | Shift-test p | Placebo p |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        p = r.get("luck_test", {}).get("p_value")
        pp = r.get("placebo_test", {}).get("p_value")
        lines.append(
            f"| {r['strategy']} | {pct(r['total_return'])} | {pct(r['annual_return'])} "
            f"| {pct(r['annual_volatility'])} | {num(r['sharpe'])} | {pct(r['max_drawdown'])} "
            f"| {num(r['calmar'])} | {pct(r['exposure'])} | {r['n_trades']} "
            f"| {pct(r['trade_win_rate'])} | {pct(r['daily_win_rate'])} | {num(p, 3)} | {num(pp, 3)} |"
        )
    lines += [
        "",
        "## Reading this table",
        "",
        "- **Placebo p** (the test to trust; only with `--placebo-runs N`): each model is",
        "  retrained N times on randomly permuted labels and traded identically. p is the",
        "  share of those no-information strategies whose net Sharpe matches or beats the",
        "  real one. A model with genuine skill should beat nearly all of them.",
        "- **Shift-test p**: share of circularly shifted copies of the strategy (same",
        "  exposure, trades and holding periods, timing randomised) whose net Sharpe is at",
        "  least as high. It is **optimistic**: feature-driven positions that happen to",
        "  sit out one large sell-off pass it easily. In placebo checks on this data,",
        "  models trained on shuffled labels scored p < 0.05 about 20% of the time.",
        "- Compare every model with **Buy & Hold** and the rule-based **MA20 > MA50**",
        "  strategy. A model that only matches them adds nothing.",
        "- Picking the best of several models on one window is optimistic; three",
        "  correlated tech stocks over under a year is a very small sample.",
        "",
        "## Walk-forward folds",
        "",
        "| Fold | Train period | Test period | Train rows | Test rows |",
        "| ---: | --- | --- | ---: | ---: |",
    ]
    for i, f in enumerate(folds, 1):
        lines.append(f"| {i} | {f['train_period'][0]} → {f['train_period'][1]} "
                     f"| {f['test_period'][0]} → {f['test_period'][1]} "
                     f"| {f['train_rows']} | {f['test_rows']} |")
    md.write_text("\n".join(lines) + "\n")
    return md, js


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(tem.DEFAULT_DATA))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--target", choices=list(tem.TARGETS), default="trend_5d")
    ap.add_argument("--models", nargs="+", default=None,
                    help="subset of model names (default: all from train_ensemble_models.py)")
    ap.add_argument("--mode", choices=["long_flat", "long_short"], default="long_flat")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--cost-bps", type=float, default=5.0)
    ap.add_argument("--delay", type=int, default=1, help="closes between signal and execution")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-components", type=int, default=35)
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle-labels", type=int, default=None, metavar="SEED",
                    help="placebo: randomly permute the training labels with this seed; "
                         "any 'edge' that remains comes from the setup, not the predictions")
    ap.add_argument("--placebo-runs", type=int, default=0, metavar="N",
                    help="also retrain each model N times on permuted labels and report "
                         "where the real Sharpe falls in that distribution (slow: N full runs)")
    args = ap.parse_args()

    if not os.path.exists(args.data):
        raise SystemExit(f"Dataset not found: {args.data}\n"
                         "See 07_model_training/classification_models/README.md for how to restore it.")

    df = tem.load_daily(args.data)
    feats = tem.select_features(df)
    targets, regime_now = tem.build_targets(df)
    label = targets[args.target]
    if args.target == "market_regime_10d":
        label = label.where(regime_now.notna())
    order = df.sort_values(["Date", "ticker"], kind="stable").index
    dfo, label = df.loc[order].reset_index(drop=True), label.loc[order].reset_index(drop=True)
    if args.shuffle_labels is not None:
        known = label.notna().to_numpy()
        rng = np.random.default_rng(args.shuffle_labels)
        label[known] = rng.permutation(label[known].to_numpy())

    models = tem.build_models(not args.no_pca, args.n_components, args.seed)
    if args.models:
        unknown = set(args.models) - set(models)
        if unknown:
            raise SystemExit(f"Unknown model(s): {unknown}. Choose from: {list(models)}")
        models = {m: models[m] for m in args.models}

    horizon = tem.TARGETS[args.target]
    print(f"Target {args.target} (horizon {horizon}d) | models: {', '.join(models)}")
    probs, folds = oos_probabilities(dfo, feats, label, horizon, models, args.n_splits)

    # Wide (date x ticker) frames
    close = dfo.pivot(index="Date", columns="ticker", values="Close").sort_index()
    returns = close.pct_change().fillna(0.0)
    cost = args.cost_bps / 1e4

    first_signal = dfo.loc[~np.isnan(next(iter(probs.values()))), "Date"].min()
    sig_dates = close.index[close.index >= first_signal]
    start = sig_dates[1 + args.delay]  # first day a signal-driven position earns a return
    window = close.index[close.index >= start]

    def run(decided):
        decided = decided.reindex(close.index).fillna(0.0)
        held, net = simulate(decided, returns, cost, args.delay)
        return held.loc[window], net.loc[window]

    rows, curves = [], {}
    # Buy & hold: long from the first day of the window (entry cost charged once)
    bh_decided = pd.DataFrame(1.0, index=close.index, columns=close.columns)
    bh_decided[bh_decided.index < sig_dates[0]] = 0.0
    held, net = run(bh_decided)
    rows.append({"strategy": "Buy & Hold", **performance(held, net)})
    curves["Buy & Hold"] = net.mean(axis=1)

    # Rule-based trend following on the same delay/costs
    g = dfo.groupby("ticker")["Close"]
    ma20 = g.transform(lambda s: s.rolling(20).mean())
    ma50 = g.transform(lambda s: s.rolling(50).mean())
    rule = pd.Series(np.where(ma20 > ma50, 1.0, 0.0 if args.mode == "long_flat" else -1.0))
    rule[ma50.isna()] = 0.0
    rule_decided = pd.DataFrame({"Date": dfo["Date"], "ticker": dfo["ticker"], "pos": rule}) \
        .pivot(index="Date", columns="ticker", values="pos")
    rule_decided[rule_decided.index < sig_dates[0]] = 0.0
    held, net = run(rule_decided)
    r = {"strategy": "MA20 > MA50 rule", **performance(held, net)}
    r["luck_test"] = shift_test(held, returns.loc[window], cost, r["sharpe"])
    rows.append(r)
    curves["MA20 > MA50 rule"] = net.mean(axis=1)

    def run_model(p):
        pos = signals_to_positions(p, args.threshold, args.mode)
        decided = pd.DataFrame({"Date": dfo["Date"], "ticker": dfo["ticker"], "pos": pos}) \
            .pivot(index="Date", columns="ticker", values="pos")
        return run(decided)

    model_rows = {}
    for m, p in probs.items():
        held, net = run_model(p)
        r = {"strategy": m, **performance(held, net)}
        r["luck_test"] = shift_test(held, returns.loc[window], cost, r["sharpe"])
        rows.append(r)
        model_rows[m] = r
        curves[m] = net.mean(axis=1)

    # Placebo test: retrain every model on randomly permuted labels and compare
    # the real net Sharpe with the resulting distribution.
    if args.placebo_runs > 0:
        known = label.notna().to_numpy()
        null = {m: [] for m in models}
        for s in range(args.placebo_runs):
            lab = label.copy()
            lab[known] = np.random.default_rng(10_000 + s).permutation(lab[known].to_numpy())
            pp, _ = oos_probabilities(dfo, feats, lab, horizon, models, args.n_splits)
            for m in models:
                held, net = run_model(pp[m])
                null[m].append(sharpe(net.mean(axis=1)))
            print(f"   placebo run {s + 1}/{args.placebo_runs} done")
        for m, r in model_rows.items():
            dist = np.array(null[m])
            r["placebo_test"] = {
                "p_value": float((np.sum(dist >= r["sharpe"]) + 1) / (len(dist) + 1)),
                "n_runs": int(len(dist)),
                "placebo_sharpe_median": float(np.median(dist)),
                "placebo_sharpe_95th": float(np.percentile(dist, 95)),
                "placebo_sharpes": [float(x) for x in dist],
            }

    print(f"\nWindow {window[0].date()} → {window[-1].date()} ({len(window)} days), "
          f"costs {args.cost_bps} bps, delay {args.delay}\n")
    print(f"{'Strategy':22s} {'Total':>8s} {'Sharpe':>7s} {'MaxDD':>8s} {'Expo':>6s} "
          f"{'Trades':>6s} {'TradeWin':>8s} {'p(shift)':>8s} {'p(placebo)':>10s}")
    for r in rows:
        p = r.get("luck_test", {}).get("p_value", float("nan"))
        pp = r.get("placebo_test", {}).get("p_value", float("nan"))
        print(f"{r['strategy']:22s} {r['total_return']:8.1%} {r['sharpe']:7.2f} {r['max_drawdown']:8.1%} "
              f"{r['exposure']:6.0%} {r['n_trades']:6d} {r['trade_win_rate']:8.1%} {p:8.3f} {pp:10.3f}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"backtest_{args.target}_{args.mode}{'' if not args.no_pca else '_nopca'}"
    if args.shuffle_labels is not None:
        stem += f"_placebo{args.shuffle_labels}"
    png = out_dir / f"{stem}.png"
    plot(curves, png, f"Walk-forward backtest: {args.target} signals, equal-weight "
                      f"{'/'.join(close.columns)} ({args.cost_bps:g} bps costs)")
    meta = {
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "target": args.target,
        "tickers": list(close.columns),
        "window": [str(window[0].date()), str(window[-1].date())],
        "n_days": int(len(window)),
        "mode": args.mode,
        "threshold": args.threshold,
        "delay": args.delay,
        "cost_bps": args.cost_bps,
        "pca": not args.no_pca,
        "n_components": args.n_components,
        "n_splits": args.n_splits,
        "seed": args.seed,
        "shuffle_labels": args.shuffle_labels,
        "plot": png.name,
    }
    md, js = write_report(rows, meta, folds, out_dir, stem)
    print(f"\nReport: {md}\nJSON:   {js}\nPlot:   {png}")


if __name__ == "__main__":
    main()

"""
Build a multi-ticker, multi-year integrated dataset from raw daily prices.

Input : long CSV from 01_data_collection/collect_prices.py
        (Date, ticker, Open, High, Low, Close, Volume), optionally including
        market symbols such as ^GSPC and ^VIX.
Output: 14_data_storage/integrated_datasets/market_features_<start>_<end>.csv
        with Date, ticker, interval, OHLCV and feature columns prefixed `f_`.

The `f_` prefix is an explicit feature contract: train_ensemble_models.py and
walk_forward_backtest.py use exactly the `f_` columns when any are present,
instead of guessing features from column names.

Every feature is computed per ticker from data up to and including day t
(rolling windows, EWMs, lagged changes), so nothing looks ahead. Features are
scale-free (ratios, returns, z-scores) so one model can be pooled across
tickers with very different price levels. Technical indicators come from
03_feature_engineering/technical_indicators.py.

Usage
-----
    python 03_feature_engineering/build_market_dataset.py --raw 14_data_storage/raw/yfinance_daily_2015-01-01_2026-11-04.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from technical_indicators import TechnicalIndicatorCalculator  # noqa: E402

OUT_DIR = REPO_ROOT / "14_data_storage" / "integrated_datasets"
MARKET_INDEX = "^GSPC"
VOL_INDEX = "^VIX"


def stock_features(g, calc):
    """Scale-free, backward-looking features for one ticker (sorted by date)."""
    g = g.sort_values("Date").reset_index(drop=True)
    t = calc.calculate_all_indicators(g, price_col="Close", volume_col="Volume")
    c = t["Close"]
    f = pd.DataFrame({"Date": g["Date"], "ticker": g["ticker"]})

    # Trend: distance from moving averages, MA spreads, MACD — all relative to price
    for p in (5, 10, 20, 50, 200):
        f[f"f_dist_sma{p}"] = c / t[f"SMA_{p}"] - 1
        f[f"f_dist_ema{p}"] = c / t[f"EMA_{p}"] - 1
    f["f_sma20_vs_50"] = t["SMA_20_vs_50"] / c
    f["f_sma50_vs_200"] = t["SMA_50_vs_200"] / c
    f["f_macd"] = t["MACD"] / c
    f["f_macd_signal"] = t["MACD_signal"] / c
    f["f_macd_hist"] = t["MACD_histogram"] / c
    f["f_macd_above_signal"] = t["MACD_cross_above"]

    # Momentum / oscillators
    f["f_rsi14"] = t["RSI_14"]
    f["f_rsi_oversold"] = t["RSI_oversold"]
    f["f_rsi_overbought"] = t["RSI_overbought"]
    f["f_bb_position"] = t["BB_position"]
    f["f_bb_width"] = t["BB_width"]
    f["f_ret_1d"] = t["daily_return"]
    for p in (5, 20, 60, 120, 252):
        f[f"f_mom_{p}d"] = c / c.shift(p) - 1

    # Volatility and range
    f["f_vol_5d"] = t["volatility_5d"]
    f["f_vol_20d"] = t["volatility_20d"]
    f["f_vol_60d"] = t["daily_return"].rolling(60).std()
    f["f_vol_ratio_20_60"] = f["f_vol_20d"] / f["f_vol_60d"]
    f["f_hl_range"] = (g["High"] - g["Low"]) / c
    f["f_gap"] = g["Open"] / c.shift(1) - 1
    f["f_dist_resistance20"] = t["dist_from_resistance"]
    f["f_dist_support20"] = t["dist_from_support"]
    f["f_dist_high252"] = c / c.rolling(252).max() - 1

    # Volume
    f["f_volume_ratio5"] = t["volume_ratio_5"]
    f["f_volume_ratio20"] = t["volume_ratio_20"]
    lv = np.log1p(g["Volume"].astype(float))
    f["f_volume_z252"] = (lv - lv.rolling(252).mean()) / lv.rolling(252).std()

    # Calendar (known in advance)
    d = pd.DatetimeIndex(g["Date"])
    f["f_dayofweek"] = d.dayofweek
    f["f_month"] = d.month
    return f.replace([np.inf, -np.inf], np.nan)


def market_features(raw):
    """Market-wide context per date from the index and volatility series, if present."""
    out = None
    if MARKET_INDEX in set(raw["ticker"]):
        m = raw[raw["ticker"] == MARKET_INDEX].sort_values("Date").set_index("Date")["Close"]
        r = m.pct_change()
        out = pd.DataFrame({
            "f_mkt_ret_1d": r,
            "f_mkt_mom_20d": m / m.shift(20) - 1,
            "f_mkt_mom_60d": m / m.shift(60) - 1,
            "f_mkt_dist_sma200": m / m.rolling(200).mean() - 1,
            "f_mkt_vol_20d": r.rolling(20).std(),
        })
    if VOL_INDEX in set(raw["ticker"]):
        v = raw[raw["ticker"] == VOL_INDEX].sort_values("Date").set_index("Date")["Close"]
        vf = pd.DataFrame({
            "f_vix": v,
            "f_vix_chg_5d": v / v.shift(5) - 1,
            "f_vix_vs_sma60": v / v.rolling(60).mean() - 1,
        })
        out = vf if out is None else out.join(vf, how="outer")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="CSV from 01_data_collection/collect_prices.py")
    ap.add_argument("--out", default=None, help="output CSV (default: 14_data_storage/integrated_datasets/...)")
    ap.add_argument("--warmup", type=int, default=200,
                    help="drop each ticker's first N rows, where long-window features are undefined")
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    raw["Date"] = pd.to_datetime(raw["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    raw = raw.drop_duplicates(["ticker", "Date"]).dropna(subset=["Close"])
    stocks = raw[~raw["ticker"].str.startswith("^")]
    if stocks.empty:
        sys.exit("No stock rows found in the raw file.")

    calc = TechnicalIndicatorCalculator()
    parts = []
    for tk, g in stocks.groupby("ticker"):
        f = stock_features(g, calc).iloc[args.warmup:]
        parts.append(f)
        print(f"  {tk:6s} {len(f):5d} rows after {args.warmup}-day warm-up")
    feats = pd.concat(parts, ignore_index=True)

    mkt = market_features(raw)
    if mkt is not None:
        feats = feats.merge(mkt, left_on="Date", right_index=True, how="left")
        if "f_mkt_mom_20d" in feats:
            feats["f_rel_mom_20d"] = feats["f_mom_20d"] - feats["f_mkt_mom_20d"]
            # 60-day rolling beta to the market, per ticker
            def beta(g):
                cov = g["f_ret_1d"].rolling(60).cov(g["f_mkt_ret_1d"])
                return cov / g["f_mkt_ret_1d"].rolling(60).var()
            feats = feats.sort_values(["ticker", "Date"])
            feats["f_beta_60d"] = feats.groupby("ticker", group_keys=False).apply(beta)
        print(f"  market context: {', '.join(c for c in mkt.columns)}")
    else:
        print("  market context: none (no ^GSPC / ^VIX rows in the raw file)")

    out = stocks.merge(feats, on=["Date", "ticker"], how="inner")
    out.insert(2, "interval", "1d")
    out = out.sort_values(["ticker", "Date"]).reset_index(drop=True)
    fcols = [c for c in out.columns if c.startswith("f_")]

    start, end = out["Date"].min().date(), out["Date"].max().date()
    path = Path(args.out) if args.out else OUT_DIR / f"market_features_{start}_{end}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} rows, {out['ticker'].nunique()} tickers, {len(fcols)} features, "
          f"{start} → {end}\n→ {path}")


if __name__ == "__main__":
    main()

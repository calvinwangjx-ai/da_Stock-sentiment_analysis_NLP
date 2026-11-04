"""
Download daily OHLCV price history from Yahoo Finance (no API key needed).

Writes one long-format CSV (Date, ticker, Open, High, Low, Close, Volume)
with split- and dividend-adjusted prices, for the project's stock universe
plus market context series (S&P 500 and VIX by default).

The output feeds 03_feature_engineering/build_market_dataset.py.

Usage
-----
    python 01_data_collection/collect_prices.py
    python 01_data_collection/collect_prices.py --start 2010-01-01 --tickers AAPL MSFT NVDA

Network: needs HTTPS access to query1.finance.yahoo.com, query2.finance.yahoo.com
and fc.yahoo.com.
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "14_data_storage" / "raw"

# Same 30-stock universe as DEFAULT_SETTINGS in 12_config_utilities/config.py
DEFAULT_TICKERS = [
    "AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA", "JPM", "JNJ",
    "V", "WMT", "PG", "DIS", "BAC", "XOM", "INTC", "CSCO", "PFE", "VZ",
    "ADBE", "CRM", "NKE", "T", "ABT", "COST", "TMO", "LLY", "AVGO", "UNH",
]
DEFAULT_MARKET = ["^GSPC", "^VIX"]


def fetch_one(yf, ticker, start, end, retries=4):
    """Daily adjusted history for one symbol, with exponential-backoff retries."""
    last_err = None
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end, interval="1d",
                                             auto_adjust=True, actions=False)
            if hist is None or hist.empty:
                raise ValueError("empty response")
            hist = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
            idx = pd.DatetimeIndex(hist.index)
            if idx.tz is not None:  # exchange-local timestamps -> plain trading dates
                idx = idx.tz_convert("America/New_York").tz_localize(None)
            hist.index = idx.normalize()
            hist.index.name = "Date"
            hist["ticker"] = ticker
            return hist.reset_index()
        except Exception as e:  # network errors, rate limits, delisted symbols
            last_err = e
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"{ticker}: {last_err}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--market", nargs="*", default=DEFAULT_MARKET,
                    help="market context symbols (stored with the same columns)")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default=date.today().isoformat(), help="exclusive end date")
    ap.add_argument("--out", default=None, help="output CSV (default: 14_data_storage/raw/...)")
    args = ap.parse_args()

    try:
        import yfinance as yf
    except ImportError:
        sys.exit("yfinance is not installed: pip install -r 18_documentation/requirements.txt")

    symbols = list(dict.fromkeys(args.tickers + list(args.market)))
    frames, failed = [], []
    for i, sym in enumerate(symbols, 1):
        try:
            df = fetch_one(yf, sym, args.start, args.end)
            frames.append(df)
            print(f"[{i:2d}/{len(symbols)}] {sym:6s} {len(df):5d} rows "
                  f"{df['Date'].min().date()} → {df['Date'].max().date()}")
        except RuntimeError as e:
            failed.append(sym)
            print(f"[{i:2d}/{len(symbols)}] {sym:6s} FAILED: {e}")

    if not frames:
        sys.exit("No data downloaded. If every symbol failed, check network access to Yahoo Finance.")

    out = pd.concat(frames, ignore_index=True)
    out = out[["Date", "ticker", "Open", "High", "Low", "Close", "Volume"]]
    out = out.drop_duplicates(["ticker", "Date"]).sort_values(["ticker", "Date"])
    path = Path(args.out) if args.out else RAW_DIR / f"yfinance_daily_{args.start}_{args.end}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"\nSaved {len(out)} rows for {out['ticker'].nunique()} symbols → {path}")
    if failed:
        print(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

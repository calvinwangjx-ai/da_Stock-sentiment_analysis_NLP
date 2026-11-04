
03 — Feature Engineering
Transforms cleaned data into predictive features. The largest single
stage by output size — the final feature space has ~2,700 columns
across stocks, news sentiment, and Reddit sentiment.

Files
File	Purpose
feature_config.py	Configuration for which features to generate
feature_engineer.py	Main feature engineering pipeline
feature_pipeline.py	Orchestrates the pipeline end-to-end
run_feature_engineering.py	Entry point
sentiment_preparer.py	Prepares sentiment scores from news and Reddit
technical_indicators.py	Generates 50+ technical indicators
Technical indicators
Trend, momentum, volatility, volume, and market-structure indicators:

SMA, EMA, MACD, RSI

Bollinger Bands, ATR

OBV, volume ratios

Support/resistance features

Sentiment features
News and Reddit text produce sentiment features that are joined to the
stock price series by (ticker, date).

Legacy
legacy/ contains earlier versions of feature_engineer.py and
technical_indicators.py.

Inputs
Cleaned data from 02_data_processing/.

Output
Feature-engineered CSVs, per source (stock, news, Reddit).

Stage boundary
Input: cleaned data.
Output: feature matrices ready for outlier handling and scaling.

## Multi-ticker price dataset: `build_market_dataset.py`

Turns the raw prices from `01_data_collection/collect_prices.py` into an
integrated dataset for `07_model_training/classification_models/` and
`09_backtesting_trading/walk_forward_backtest.py`:

```bash
python 03_feature_engineering/build_market_dataset.py --raw 14_data_storage/raw/yfinance_daily.csv
```

- About 40 stock features per ticker, built with `technical_indicators.py`
  and normalised to be scale-free (distance from SMAs/EMAs, MA spreads and
  MACD relative to price, RSI, Bollinger position/width, 5–252-day momentum,
  volatility, high–low range, gaps, volume ratios and z-score, calendar).
- About 10 market features when `^GSPC` / `^VIX` are present (index return,
  momentum, distance from 200-day average, volatility, VIX level and changes,
  relative momentum, 60-day beta).
- Feature columns are prefixed `f_`. The training and backtest scripts use
  exactly these columns when present, rather than guessing from names.
- All features use data up to day *t* only. Tested by building features on
  a truncated history and checking that every overlapping value is
  identical to the full-history build.
- The first 200 rows per ticker are dropped (`--warmup`) because the
  200-day features are undefined there.
- No news or Reddit sentiment: free APIs do not provide multi-year history.

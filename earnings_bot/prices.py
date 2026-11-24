"""Minute-by-minute prices (including after-hours) from Alpaca's free market data."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config


def get_minute_bars(ticker: str, start_utc: pd.Timestamp, end_utc: pd.Timestamp, cache_path: Path) -> pd.DataFrame:
    """Return a DataFrame indexed by bar start time (UTC) with open/high/low/close/volume."""
    if cache_path.exists():
        df = pd.read_csv(cache_path, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        return df

    if not (config.ALPACA_API_KEY and config.ALPACA_SECRET_KEY):
        raise RuntimeError("Alpaca keys missing. Add ALPACA_API_KEY and ALPACA_SECRET_KEY to your .env file.")

    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = StockHistoricalDataClient(config.ALPACA_API_KEY, config.ALPACA_SECRET_KEY)
    req = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Minute,
        start=start_utc.to_pydatetime(),
        end=end_utc.to_pydatetime(),
        feed=DataFeed.SIP,  # full consolidated tape; free accounts can use it for data older than 15 min
        adjustment=Adjustment.RAW,
    )
    df = client.get_stock_bars(req).df
    if df.empty:
        raise RuntimeError(f"Alpaca returned no bars for {ticker} between {start_utc} and {end_utc}.")
    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(ticker, level="symbol")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].sort_index()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path)
    return df


def last_regular_close(bars: pd.DataFrame, before_utc: pd.Timestamp) -> float | None:
    """Close of the last regular-session (9:30-16:00 ET) bar before a time, i.e. the price before the release."""
    et = bars.index.tz_convert("America/New_York")
    minutes = et.hour * 60 + et.minute
    mask = (bars.index < before_utc) & (minutes >= 9 * 60 + 30) & (minutes < 16 * 60)
    regular = bars[mask]
    return float(regular["close"].iloc[-1]) if len(regular) else None

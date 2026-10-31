"""Market data retrieval and validation."""
from __future__ import annotations
import logging
from typing import Iterable
import pandas as pd

logger = logging.getLogger(__name__)

def clean_prices(prices: pd.DataFrame, min_observations: int = 100) -> pd.DataFrame:
    if prices is None or prices.empty:
        raise ValueError("No price data supplied.")
    frame = prices.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        # yfinance commonly returns (Price field, ticker)
        if "Close" in frame.columns.get_level_values(0):
            frame = frame["Close"]
        else:
            frame.columns = [str(c[-1]) for c in frame.columns]
    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.where(frame > 0).ffill(limit=3).dropna(axis=1, how="all")
    frame = frame.dropna(axis=0, how="any")
    if frame.shape[1] < 1:
        raise ValueError("No usable assets remain after cleaning.")
    if len(frame) < min_observations:
        raise ValueError(f"Only {len(frame)} overlapping observations; need at least {min_observations}.")
    return frame

def fetch_prices(tickers: Iterable[str], period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    symbols = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not symbols:
        raise ValueError("Enter at least one ticker.")
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("yfinance is missing. Run pip install -r requirements.txt.") from exc
    try:
        raw = yf.download(symbols, period=period, interval=interval, auto_adjust=True,
                          progress=False, threads=True)
        if raw.empty:
            raise ValueError("The provider returned no market data.")
        if isinstance(raw.columns, pd.MultiIndex):
            field = "Close" if "Close" in raw.columns.get_level_values(0) else raw.columns.get_level_values(0)[0]
            prices = raw[field]
        else:
            prices = raw[["Close"]] if "Close" in raw.columns else raw
            if len(symbols) == 1:
                prices.columns = symbols
        return clean_prices(prices)
    except Exception as exc:
        logger.exception("Price download failed")
        raise RuntimeError(f"Market data retrieval failed: {exc}") from exc

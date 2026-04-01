"""Data Service — fetches and caches OHLCV data from yfinance."""
from __future__ import annotations

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class DataService:
    """Handles all market data fetching and preprocessing."""

    _cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
    CACHE_TTL = timedelta(minutes=15)

    @classmethod
    def _is_fresh(cls, symbol: str) -> bool:
        if symbol not in cls._cache:
            return False
        ts, _ = cls._cache[symbol]
        return (datetime.now() - ts) < cls.CACHE_TTL

    @classmethod
    def fetch_ohlcv(cls, symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
        cache_key = f"{symbol}_{period}_{interval}"
        if cls._is_fresh(cache_key):
            return cls._cache[cache_key][1].copy()
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                logger.warning(f"No data for {symbol}")
                return pd.DataFrame()
            df = cls._clean(df)
            cls._cache[cache_key] = (datetime.now(), df)
            return df.copy()
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_multiple(cls, symbols: list[str], period: str = "1y", interval: str = "1d") -> dict[str, pd.DataFrame]:
        result = {}
        for sym in symbols:
            df = cls.fetch_ohlcv(sym, period, interval)
            if not df.empty:
                result[sym] = df
        return result

    @classmethod
    def fetch_index(cls, period: str = "1y") -> pd.DataFrame:
        from ..config import NIFTY_INDEX
        return cls.fetch_ohlcv(NIFTY_INDEX, period)

    @classmethod
    def fetch_vix(cls, period: str = "1y") -> pd.DataFrame:
        from ..config import INDIA_VIX
        return cls.fetch_ohlcv(INDIA_VIX, period)

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.ffill().dropna()
        return df

    @classmethod
    def get_current_price(cls, symbol: str) -> Optional[float]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.fast_info
            return float(info.get("lastPrice", info.get("previousClose", 0)))
        except Exception:
            df = cls.fetch_ohlcv(symbol, period="5d")
            if not df.empty:
                return float(df["Close"].iloc[-1])
            return None

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()

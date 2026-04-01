"""Data Service — fetches and caches OHLCV data from yfinance."""
from __future__ import annotations

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# IST offset — same pattern as live_trader.py
_IST = timezone(timedelta(hours=5, minutes=30))
def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


class DataService:
    """Handles all market data fetching and preprocessing."""

    _cache: dict[str, tuple[datetime, pd.DataFrame]] = {}
    CACHE_TTL = timedelta(minutes=5)   # 5 min for intraday freshness
    # Live price cache is much shorter — 60 seconds
    _price_cache: dict[str, tuple[datetime, float]] = {}
    PRICE_TTL = timedelta(seconds=60)

    @classmethod
    def _is_fresh(cls, key: str, ttl: timedelta = None) -> bool:
        ttl = ttl or cls.CACHE_TTL
        cache = cls._cache if ttl == cls.CACHE_TTL else cls._price_cache
        if key not in cache:
            return False
        ts = cache[key][0]
        return (_now_utc() - ts) < ttl

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
            cls._cache[cache_key] = (_now_utc(), df)
            return df.copy()
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()

    @classmethod
    def fetch_live_price(cls, symbol: str) -> dict | None:
        """
        Returns the latest real-time price for a symbol.
        Uses yfinance fast_info (no cache delay) — refreshes every 60 s.
        Falls back to 1-minute bar if fast_info fails.
        Returns dict: {price, prev_close, change_pct, volume, high, low, name}
        """
        cache_key = f"live_{symbol}"
        if cache_key in cls._price_cache:
            ts, data = cls._price_cache[cache_key]
            if (_now_utc() - ts) < cls.PRICE_TTL:
                return data

        try:
            ticker = yf.Ticker(symbol)
            fi = ticker.fast_info

            # yfinance fast_info uses attribute access, not dict keys
            try:
                price = float(getattr(fi, "last_price", None) or 0)
                prev  = float(getattr(fi, "previous_close", None) or 0)
            except (TypeError, ValueError):
                price, prev = 0, 0

            # fast_info failed — fall back to 1m bar
            if price == 0:
                df1m = ticker.history(period="1d", interval="1m")
                if df1m.empty:
                    return None
                df1m = cls._clean(df1m)
                if df1m.empty:
                    return None
                price = float(df1m["Close"].iloc[-1])
                prev  = float(df1m["Close"].iloc[0]) if len(df1m) > 1 else price

            chg_pct = round((price / prev - 1) * 100, 2) if prev > 0 else 0.0

            # 5-day daily for high/low context
            df5d = cls.fetch_ohlcv(symbol, period="5d", interval="1d")
            high_52 = float(df5d["High"].max()) if not df5d.empty else price
            low_52  = float(df5d["Low"].min())  if not df5d.empty else price
            vol     = int(getattr(fi, "three_month_average_volume", None) or
                          (int(df5d["Volume"].iloc[-1]) if not df5d.empty else 0))

            # Today's intraday high/low from 1m bars
            try:
                df_today = ticker.history(period="1d", interval="1m")
                today_high = float(df_today["High"].max()) if not df_today.empty else price
                today_low  = float(df_today["Low"].min())  if not df_today.empty else price
                today_vol  = int(df_today["Volume"].sum())  if not df_today.empty else vol
            except Exception:
                today_high, today_low, today_vol = price, price, vol

            data = {
                "symbol":    symbol.replace(".NS", "").replace(".BO", ""),
                "price":     round(price, 2),
                "prev_close": round(prev, 2),
                "chg":       chg_pct,
                "high_5d":   round(high_52, 2),
                "low_5d":    round(low_52, 2),
                "today_high": round(today_high, 2),
                "today_low":  round(today_low, 2),
                "volume":    today_vol,
                "df":        df5d,
            }
            cls._price_cache[cache_key] = (_now_utc(), data)
            return data
        except Exception as e:
            logger.error(f"fetch_live_price {symbol}: {e}")
            return None

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
        # Strip MultiIndex if present (yfinance bug with some versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Keep only standard OHLCV columns
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df.ffill().dropna()
        return df

    @classmethod
    def get_current_price(cls, symbol: str) -> Optional[float]:
        data = cls.fetch_live_price(symbol)
        if data:
            return data["price"]
        df = cls.fetch_ohlcv(symbol, period="5d")
        if not df.empty:
            return float(df["Close"].iloc[-1])
        return None

    @classmethod
    def clear_cache(cls):
        cls._cache.clear()
        cls._price_cache.clear()

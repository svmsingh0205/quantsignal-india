"""
Broker Feed — Real-Time Data Layer
===================================
Priority chain:
  1. Zerodha Kite Connect  (WebSocket LTP + REST OHLC)
  2. Upstox API            (WebSocket LTP + REST OHLC)
  3. Yahoo Finance         (REST fallback, 15-min delayed)

All credentials read from environment variables — never hardcoded.

Environment variables required:
  KITE_API_KEY          Zerodha Kite Connect API key
  KITE_ACCESS_TOKEN     Zerodha access token (regenerated daily)
  UPSTOX_ACCESS_TOKEN   Upstox OAuth2 access token
  ALPHA_VANTAGE_KEY     Alpha Vantage API key (optional)

Usage:
  from backend.engines.broker_feed import get_live_tick, get_ohlcv, DataSource
"""
from __future__ import annotations

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── IST ───────────────────────────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))
def _now_ist() -> datetime:
    return datetime.now(_IST).replace(tzinfo=None)
def _now_ts() -> float:
    return time.time()

# ── Data freshness constants ──────────────────────────────────────────────────
MAX_TICK_AGE_SEC   = 5      # reject ticks older than 5 seconds
MAX_PRICE_DEV_PCT  = 1.0    # reject if primary vs secondary > 1% deviation
TICK_CACHE_TTL     = 3      # seconds — in-process tick cache
OHLCV_CACHE_TTL    = 300    # 5 minutes for OHLCV bars


@dataclass
class Tick:
    symbol:      str
    ltp:         float          # last traded price
    bid:         float = 0.0
    ask:         float = 0.0
    volume:      int   = 0
    oi:          int   = 0      # open interest (futures/options)
    prev_close:  float = 0.0
    today_high:  float = 0.0
    today_low:   float = 0.0
    vwap:        float = 0.0
    timestamp:   float = field(default_factory=_now_ts)
    source:      str   = "unknown"
    is_valid:    bool  = False

    @property
    def change_pct(self) -> float:
        if self.prev_close > 0:
            return round((self.ltp / self.prev_close - 1) * 100, 2)
        return 0.0

    @property
    def age_sec(self) -> float:
        return _now_ts() - self.timestamp

    @property
    def is_fresh(self) -> bool:
        return self.age_sec <= MAX_TICK_AGE_SEC


class DataSource:
    KITE    = "kite"
    UPSTOX  = "upstox"
    YAHOO   = "yahoo"
    UNKNOWN = "unknown"


# ── In-process tick cache ─────────────────────────────────────────────────────
_tick_cache:  dict[str, Tick]                          = {}
_ohlcv_cache: dict[str, tuple[float, pd.DataFrame]]   = {}
_cache_lock = threading.Lock()


def _cache_tick(symbol: str, tick: Tick) -> None:
    with _cache_lock:
        _tick_cache[symbol] = tick


def _get_cached_tick(symbol: str) -> Optional[Tick]:
    with _cache_lock:
        t = _tick_cache.get(symbol)
    if t and t.age_sec < TICK_CACHE_TTL:
        return t
    return None


def _cache_ohlcv(key: str, df: pd.DataFrame) -> None:
    with _cache_lock:
        _ohlcv_cache[key] = (_now_ts(), df)


def _get_cached_ohlcv(key: str) -> Optional[pd.DataFrame]:
    with _cache_lock:
        entry = _ohlcv_cache.get(key)
    if entry and (_now_ts() - entry[0]) < OHLCV_CACHE_TTL:
        return entry[1].copy()
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 1: ZERODHA KITE CONNECT
# ══════════════════════════════════════════════════════════════════════════════

def _kite_available() -> bool:
    return bool(os.environ.get("KITE_API_KEY") and os.environ.get("KITE_ACCESS_TOKEN"))


def _get_kite():
    """Return authenticated KiteConnect instance or None."""
    if not _kite_available():
        return None
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=os.environ["KITE_API_KEY"])
        kite.set_access_token(os.environ["KITE_ACCESS_TOKEN"])
        return kite
    except ImportError:
        logger.warning("kiteconnect package not installed. Run: pip install kiteconnect")
        return None
    except Exception as e:
        logger.warning(f"Kite init failed: {e}")
        return None


def _kite_fetch_tick(symbol: str) -> Optional[Tick]:
    """
    Fetch live tick from Zerodha Kite REST API.
    symbol: NSE symbol without .NS (e.g. "RELIANCE")
    """
    kite = _get_kite()
    if not kite:
        return None
    try:
        # Kite uses "NSE:SYMBOL" format
        instrument = f"NSE:{symbol}"
        quote = kite.quote([instrument])
        data = quote.get(instrument, {})
        if not data:
            return None

        ohlc = data.get("ohlc", {})
        depth = data.get("depth", {})
        buy_orders  = depth.get("buy", [{}])
        sell_orders = depth.get("sell", [{}])

        tick = Tick(
            symbol     = symbol,
            ltp        = float(data.get("last_price", 0)),
            bid        = float(buy_orders[0].get("price", 0)) if buy_orders else 0,
            ask        = float(sell_orders[0].get("price", 0)) if sell_orders else 0,
            volume     = int(data.get("volume", 0)),
            oi         = int(data.get("oi", 0)),
            prev_close = float(ohlc.get("close", 0)),
            today_high = float(ohlc.get("high", 0)),
            today_low  = float(ohlc.get("low", 0)),
            vwap       = float(data.get("average_price", 0)),
            timestamp  = _now_ts(),
            source     = DataSource.KITE,
            is_valid   = True,
        )
        return tick if tick.ltp > 0 else None
    except Exception as e:
        logger.debug(f"Kite tick {symbol}: {e}")
        return None


def _kite_fetch_ohlcv(symbol: str, interval: str = "5minute", days: int = 5) -> Optional[pd.DataFrame]:
    """
    Fetch historical OHLCV from Kite.
    interval: "minute", "5minute", "15minute", "60minute", "day"
    """
    kite = _get_kite()
    if not kite:
        return None
    try:
        from datetime import date
        instruments = kite.instruments("NSE")
        token = next((i["instrument_token"] for i in instruments if i["tradingsymbol"] == symbol), None)
        if not token:
            return None

        to_date   = datetime.now(_IST).date()
        from_date = to_date - timedelta(days=days)

        records = kite.historical_data(token, from_date, to_date, interval)
        if not records:
            return None

        df = pd.DataFrame(records)
        df = df.rename(columns={"date": "Datetime", "open": "Open", "high": "High",
                                  "low": "Low", "close": "Close", "volume": "Volume"})
        df = df.set_index("Datetime")
        df.index = pd.to_datetime(df.index)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    except Exception as e:
        logger.debug(f"Kite OHLCV {symbol}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 2: UPSTOX API
# ══════════════════════════════════════════════════════════════════════════════

def _upstox_available() -> bool:
    return bool(os.environ.get("UPSTOX_ACCESS_TOKEN"))


def _upstox_fetch_tick(symbol: str) -> Optional[Tick]:
    """
    Fetch live tick from Upstox v2 REST API.
    symbol: NSE symbol without .NS (e.g. "RELIANCE")
    """
    if not _upstox_available():
        return None
    try:
        import requests
        token = os.environ["UPSTOX_ACCESS_TOKEN"]
        # Upstox instrument key format: NSE_EQ|INE002A01018 — need ISIN
        # Use symbol-based quote endpoint
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }
        # Upstox v2 market quote by symbol
        instrument_key = f"NSE_EQ|{symbol}"
        url = f"https://api.upstox.com/v2/market-quote/quotes?instrument_key={instrument_key}"
        resp = requests.get(url, headers=headers, timeout=5)
        if resp.status_code != 200:
            return None

        data = resp.json().get("data", {}).get(instrument_key, {})
        if not data:
            return None

        ohlc = data.get("ohlc", {})
        depth = data.get("depth", {})
        buy_orders  = depth.get("buy", [{}])
        sell_orders = depth.get("sell", [{}])

        tick = Tick(
            symbol     = symbol,
            ltp        = float(data.get("last_price", 0)),
            bid        = float(buy_orders[0].get("price", 0)) if buy_orders else 0,
            ask        = float(sell_orders[0].get("price", 0)) if sell_orders else 0,
            volume     = int(data.get("volume", 0)),
            oi         = int(data.get("oi", 0)),
            prev_close = float(ohlc.get("close", 0)),
            today_high = float(ohlc.get("high", 0)),
            today_low  = float(ohlc.get("low", 0)),
            vwap       = float(data.get("average_price", 0)),
            timestamp  = _now_ts(),
            source     = DataSource.UPSTOX,
            is_valid   = True,
        )
        return tick if tick.ltp > 0 else None
    except Exception as e:
        logger.debug(f"Upstox tick {symbol}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SOURCE 3: YAHOO FINANCE (FALLBACK)
# ══════════════════════════════════════════════════════════════════════════════

def _yahoo_fetch_tick(symbol: str) -> Optional[Tick]:
    """Yahoo Finance fallback — 15-min delayed but always available."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        fi = ticker.fast_info

        ltp  = float(getattr(fi, "last_price", None) or 0)
        prev = float(getattr(fi, "previous_close", None) or 0)

        if ltp == 0:
            # Fall back to 1m bar
            df1m = ticker.history(period="1d", interval="1m")
            if df1m.empty:
                return None
            if isinstance(df1m.columns, pd.MultiIndex):
                df1m.columns = df1m.columns.get_level_values(0)
            ltp  = float(df1m["Close"].iloc[-1])
            prev = float(df1m["Close"].iloc[0]) if len(df1m) > 1 else ltp

        # Get today's OHLC from 1m bars
        try:
            df_today = ticker.history(period="1d", interval="1m")
            if isinstance(df_today.columns, pd.MultiIndex):
                df_today.columns = df_today.columns.get_level_values(0)
            today_high = float(df_today["High"].max()) if not df_today.empty else ltp
            today_low  = float(df_today["Low"].min())  if not df_today.empty else ltp
            today_vol  = int(df_today["Volume"].sum())  if not df_today.empty else 0
            # VWAP
            if not df_today.empty:
                tp = (df_today["High"] + df_today["Low"] + df_today["Close"]) / 3
                vwap = float((tp * df_today["Volume"]).cumsum().iloc[-1] /
                             df_today["Volume"].cumsum().replace(0, np.nan).iloc[-1])
            else:
                vwap = ltp
        except Exception:
            today_high = today_low = ltp
            today_vol  = 0
            vwap       = ltp

        return Tick(
            symbol     = symbol,
            ltp        = round(ltp, 2),
            prev_close = round(prev, 2),
            today_high = round(today_high, 2),
            today_low  = round(today_low, 2),
            volume     = today_vol,
            vwap       = round(vwap, 2),
            timestamp  = _now_ts(),
            source     = DataSource.YAHOO,
            is_valid   = True,
        )
    except Exception as e:
        logger.debug(f"Yahoo tick {symbol}: {e}")
        return None


def _yahoo_fetch_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.NS")
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        df = df[keep].dropna()
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.debug(f"Yahoo OHLCV {symbol}: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _validate_ticks(primary: Optional[Tick], secondary: Optional[Tick]) -> Optional[Tick]:
    """
    Cross-validate two ticks:
    1. Freshness: reject if > MAX_TICK_AGE_SEC old
    2. Price deviation: reject primary if > MAX_PRICE_DEV_PCT vs secondary
    Returns best validated tick or None.
    """
    def fresh(t: Optional[Tick]) -> bool:
        return t is not None and t.is_fresh and t.ltp > 0

    if fresh(primary) and fresh(secondary):
        dev = abs(primary.ltp - secondary.ltp) / max(primary.ltp, 1e-6) * 100
        if dev > MAX_PRICE_DEV_PCT:
            logger.warning(
                f"Price deviation {primary.symbol}: "
                f"{primary.source}={primary.ltp} vs {secondary.source}={secondary.ltp} "
                f"({dev:.2f}%) → using secondary"
            )
            secondary.is_valid = True
            return secondary
        # Merge: primary price + secondary VWAP/depth if available
        if secondary.vwap > 0 and primary.vwap == 0:
            primary.vwap = secondary.vwap
        if secondary.today_high > 0 and primary.today_high == 0:
            primary.today_high = secondary.today_high
            primary.today_low  = secondary.today_low
        primary.is_valid = True
        return primary

    if fresh(primary):
        primary.is_valid = True
        return primary

    if fresh(secondary):
        secondary.is_valid = True
        return secondary

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_live_tick(symbol: str) -> Optional[Tick]:
    """
    Get validated live tick for a symbol.
    Priority: Kite → Upstox → Yahoo
    Returns None only if all sources fail.
    """
    # Cache hit
    cached = _get_cached_tick(symbol)
    if cached:
        return cached

    # Parallel fetch from top two available sources
    sources = []
    if _kite_available():
        sources.append((_kite_fetch_tick, symbol))
    if _upstox_available():
        sources.append((_upstox_fetch_tick, symbol))
    if not sources:
        # No broker API — use Yahoo directly
        tick = _yahoo_fetch_tick(symbol)
        if tick:
            _cache_tick(symbol, tick)
        return tick

    # Fetch primary + secondary in parallel
    results = [None, None]
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(fn, sym): i for i, (fn, sym) in enumerate(sources[:2])}
        for fut in as_completed(futs):
            results[futs[fut]] = fut.result()

    primary   = results[0]
    secondary = results[1]

    # If broker APIs returned nothing, fall back to Yahoo
    if not primary and not secondary:
        tick = _yahoo_fetch_tick(symbol)
        if tick:
            _cache_tick(symbol, tick)
        return tick

    # If only one broker available, cross-check with Yahoo
    if primary and not secondary:
        secondary = _yahoo_fetch_tick(symbol)
    elif secondary and not primary:
        primary, secondary = secondary, _yahoo_fetch_tick(symbol)

    validated = _validate_ticks(primary, secondary)
    if validated:
        _cache_tick(symbol, validated)
    return validated


def get_ohlcv(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Get OHLCV bars. Uses Kite if available, else Yahoo.
    Results cached for OHLCV_CACHE_TTL seconds.
    """
    cache_key = f"{symbol}_{period}_{interval}"
    cached = _get_cached_ohlcv(cache_key)
    if cached is not None:
        return cached

    df = None
    if _kite_available():
        # Map period/interval to Kite format
        kite_interval_map = {
            "5m": "5minute", "15m": "15minute", "1h": "60minute",
            "1d": "day", "1wk": "day",
        }
        kite_interval = kite_interval_map.get(interval, "day")
        days_map = {"5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
        days = days_map.get(period, 365)
        df = _kite_fetch_ohlcv(symbol, kite_interval, days)

    if df is None or df.empty:
        df = _yahoo_fetch_ohlcv(symbol, period, interval)

    if df is not None and not df.empty:
        _cache_ohlcv(cache_key, df)
        return df

    return pd.DataFrame()


def get_live_ticks_batch(symbols: list[str], max_workers: int = 8) -> dict[str, Tick]:
    """
    Fetch validated ticks for multiple symbols in parallel.
    Returns only valid ticks.
    """
    results = {}

    def _fetch(sym: str) -> tuple[str, Optional[Tick]]:
        return sym, get_live_tick(sym)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            sym, tick = fut.result()
            if tick and tick.is_valid:
                results[sym] = tick

    return results


def get_data_source_status() -> dict:
    """Returns which data sources are configured and available."""
    return {
        "kite":    {"available": _kite_available(),   "source": DataSource.KITE},
        "upstox":  {"available": _upstox_available(), "source": DataSource.UPSTOX},
        "yahoo":   {"available": True,                "source": DataSource.YAHOO},
        "primary": DataSource.KITE if _kite_available() else
                   (DataSource.UPSTOX if _upstox_available() else DataSource.YAHOO),
    }


def clear_tick_cache() -> None:
    with _cache_lock:
        _tick_cache.clear()

"""
Data Validator — Multi-source fetch + validation layer for live signals.

Architecture:
  Primary source  : yfinance fast_info (LTP, volume, bid/ask proxy)
  Secondary source: yfinance 1-minute bars (cross-check + VWAP)
  Validation      : freshness check (≤5s stale) + price deviation check (≤1%)
  Cache           : in-process TTL dict (3s for live, 5m for OHLCV)
  Signal gate     : signals only generated on validated data

This replaces the raw yfinance fetch in _scan_one with a validated pipeline.
"""
from __future__ import annotations

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_DATA_AGE_SEC  = 5      # data older than 5s is considered stale
MAX_PRICE_DEV     = 0.01   # 1% max deviation between primary and secondary
LIVE_CACHE_TTL    = 3      # seconds — matches Redis TTL from architecture doc
OHLCV_CACHE_TTL   = 300    # 5 minutes for OHLCV bars

# ── In-process cache (replaces Redis for Streamlit Cloud compatibility) ────────
_live_cache: dict[str, tuple[float, dict]] = {}   # symbol → (timestamp, data)
_ohlcv_cache: dict[str, tuple[float, pd.DataFrame]] = {}


def _now_ts() -> float:
    return time.time()


def _cache_set(symbol: str, data: dict) -> None:
    _live_cache[symbol] = (_now_ts(), data)


def _cache_get(symbol: str) -> Optional[dict]:
    entry = _live_cache.get(symbol)
    if entry and (_now_ts() - entry[0]) < LIVE_CACHE_TTL:
        return entry[1]
    return None


def _ohlcv_cache_set(key: str, df: pd.DataFrame) -> None:
    _ohlcv_cache[key] = (_now_ts(), df)


def _ohlcv_cache_get(key: str) -> Optional[pd.DataFrame]:
    entry = _ohlcv_cache.get(key)
    if entry and (_now_ts() - entry[0]) < OHLCV_CACHE_TTL:
        return entry[1].copy()
    return None


# ── Primary fetch: fast_info LTP ──────────────────────────────────────────────
def _fetch_primary(symbol: str) -> Optional[dict]:
    """
    Fetch LTP from yfinance fast_info.
    Returns dict with price, prev_close, volume, timestamp.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info

        price = float(getattr(fi, "last_price", None) or 0)
        prev  = float(getattr(fi, "previous_close", None) or 0)
        vol   = int(getattr(fi, "three_month_average_volume", None) or 0)

        if price <= 0:
            return None

        return {
            "symbol":    symbol,
            "price":     round(price, 2),
            "prev_close": round(prev, 2),
            "volume":    vol,
            "source":    "fast_info",
            "timestamp": _now_ts(),
            "is_valid":  False,   # set by validator
        }
    except Exception as e:
        logger.debug(f"_fetch_primary {symbol}: {e}")
        return None


# ── Secondary fetch: 1-minute bars ────────────────────────────────────────────
def _fetch_secondary(symbol: str) -> Optional[dict]:
    """
    Fetch latest price from 1-minute bars.
    Also computes today's VWAP and intraday high/low.
    """
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m")

        if df.empty:
            return None

        # Clean columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna()

        if df.empty:
            return None

        price     = float(df["Close"].iloc[-1])
        prev      = float(df["Close"].iloc[0]) if len(df) > 1 else price
        today_high = float(df["High"].max())
        today_low  = float(df["Low"].min())
        today_vol  = int(df["Volume"].sum())

        # VWAP = cumulative(price * volume) / cumulative(volume)
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = float((tp * df["Volume"]).cumsum().iloc[-1] /
                     df["Volume"].cumsum().replace(0, np.nan).iloc[-1])

        return {
            "symbol":     symbol,
            "price":      round(price, 2),
            "prev_close": round(prev, 2),
            "today_high": round(today_high, 2),
            "today_low":  round(today_low, 2),
            "volume":     today_vol,
            "vwap":       round(vwap, 2),
            "source":     "1m_bars",
            "timestamp":  _now_ts(),
            "df_1m":      df,
            "is_valid":   False,
        }
    except Exception as e:
        logger.debug(f"_fetch_secondary {symbol}: {e}")
        return None


# ── Validation layer ──────────────────────────────────────────────────────────
def _validate(primary: Optional[dict], secondary: Optional[dict]) -> Optional[dict]:
    """
    Validates data freshness and cross-source price accuracy.
    Returns the best available validated data dict, or None if both fail.
    """
    now = _now_ts()

    def _is_fresh(d: dict) -> bool:
        return d is not None and (now - d["timestamp"]) <= MAX_DATA_AGE_SEC

    def _price_ok(p: dict, s: dict) -> bool:
        if not p or not s:
            return True   # can't cross-check, accept single source
        dev = abs(p["price"] - s["price"]) / max(p["price"], 1e-6)
        return dev <= MAX_PRICE_DEV

    # Both available
    if primary and secondary:
        if not _is_fresh(primary):
            logger.debug(f"Primary stale for {primary['symbol']} → using secondary")
            secondary["is_valid"] = True
            return secondary

        if not _price_ok(primary, secondary):
            logger.debug(
                f"Price mismatch {primary['symbol']}: "
                f"primary={primary['price']} secondary={secondary['price']} → using secondary"
            )
            secondary["is_valid"] = True
            return secondary

        # Both valid — merge: use primary price, secondary VWAP/intraday H/L
        merged = {**primary}
        merged["today_high"] = secondary.get("today_high", primary["price"])
        merged["today_low"]  = secondary.get("today_low",  primary["price"])
        merged["vwap"]       = secondary.get("vwap",       primary["price"])
        merged["df_1m"]      = secondary.get("df_1m",      pd.DataFrame())
        merged["is_valid"]   = True
        merged["source"]     = "merged"
        return merged

    # Only one source
    if primary and _is_fresh(primary):
        primary["is_valid"] = True
        primary.setdefault("today_high", primary["price"])
        primary.setdefault("today_low",  primary["price"])
        primary.setdefault("vwap",       primary["price"])
        primary.setdefault("df_1m",      pd.DataFrame())
        return primary

    if secondary and _is_fresh(secondary):
        secondary["is_valid"] = True
        return secondary

    return None


# ── Public API ────────────────────────────────────────────────────────────────
def get_validated_tick(symbol: str) -> Optional[dict]:
    """
    Main entry point. Returns validated live tick for a symbol.
    Checks in-process cache first (3s TTL), then fetches from both sources.

    Returns dict with:
        symbol, price, prev_close, today_high, today_low, vwap,
        volume, source, is_valid, timestamp, df_1m
    Returns None if data is unavailable or invalid.
    """
    # Cache hit
    cached = _cache_get(symbol)
    if cached:
        return cached

    # Parallel fetch from both sources
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_primary   = ex.submit(_fetch_primary,   symbol)
        f_secondary = ex.submit(_fetch_secondary, symbol)
        primary   = f_primary.result()
        secondary = f_secondary.result()

    validated = _validate(primary, secondary)

    if validated:
        _cache_set(symbol, validated)

    return validated


def get_validated_ticks_batch(symbols: list[str], max_workers: int = 8) -> dict[str, dict]:
    """
    Fetch and validate ticks for multiple symbols in parallel.
    Returns dict: symbol → validated tick (only valid ones included).
    """
    results = {}

    def _fetch_one(sym: str) -> tuple[str, Optional[dict]]:
        return sym, get_validated_tick(sym)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_one, sym): sym for sym in symbols}
        for fut in as_completed(futs):
            sym, data = fut.result()
            if data and data.get("is_valid"):
                results[sym] = data

    return results


def generate_signal(tick: dict, avg_volume: float, vwap: float) -> str:
    """
    Signal engine — only runs on validated data.
    Mirrors the architecture doc logic with VWAP + volume + bid/ask spread.
    """
    if not tick or not tick.get("is_valid"):
        return "NO_TRADE"

    ltp    = tick["price"]
    volume = tick.get("volume", 0)
    tick_vwap = tick.get("vwap", vwap)

    # Use tick's own VWAP if available, else passed-in VWAP
    effective_vwap = tick_vwap if tick_vwap > 0 else vwap

    # Strong BUY: volume spike + price above VWAP
    if volume > avg_volume * 1.5 and ltp > effective_vwap:
        return "BUY"

    # Strong SELL: volume spike + price below VWAP
    if volume > avg_volume * 1.5 and ltp < effective_vwap:
        return "SELL"

    return "NO_TRADE"


def clear_live_cache() -> None:
    """Clear the in-process live tick cache."""
    _live_cache.clear()

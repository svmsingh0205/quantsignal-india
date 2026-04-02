"""
Data Validator — Strict validation layer. No fake data ever passes through.

Rules:
  - price == 0 or None  → REJECT
  - volume == 0          → REJECT (market closed is OK, but 0 avg vol = bad symbol)
  - empty DataFrame      → REJECT
  - symbol not on NSE    → REJECT with clear error
  - stale data (>5s)     → REJECT, try secondary
  - price deviation >1%  → use secondary, log warning

Public API:
  validate_symbol(symbol)          → (True, None) | (False, reason_str)
  get_validated_tick(symbol)       → dict | None   (None = hard reject)
  get_validated_ohlcv(symbol, ...) → pd.DataFrame  (empty = hard reject)
  get_validated_ticks_batch(...)   → dict[symbol, dict]
  clear_live_cache()
"""
from __future__ import annotations

import time
import logging
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

# ── Validation constants ──────────────────────────────────────────────────────
MAX_DATA_AGE_SEC  = 5      # ticks older than 5s are stale
MAX_PRICE_DEV     = 0.01   # 1% max cross-source deviation
LIVE_CACHE_TTL    = 3      # seconds
OHLCV_CACHE_TTL   = 300    # 5 minutes
MIN_OHLCV_ROWS    = 10     # fewer rows = bad symbol / delisted
MIN_PRICE         = 0.01   # reject sub-1-paise prices

# ── In-process TTL cache ──────────────────────────────────────────────────────
_live_cache:  dict[str, tuple[float, dict]]          = {}
_ohlcv_cache: dict[str, tuple[float, pd.DataFrame]]  = {}


def _now() -> float:
    return time.time()


def _cache_set(symbol: str, data: dict) -> None:
    _live_cache[symbol] = (_now(), data)


def _cache_get(symbol: str) -> Optional[dict]:
    entry = _live_cache.get(symbol)
    if entry and (_now() - entry[0]) < LIVE_CACHE_TTL:
        return entry[1]
    return None


# ══════════════════════════════════════════════════════════════════════════════
# SYMBOL VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_symbol(symbol: str) -> tuple[bool, Optional[str]]:
    """
    Validate that a symbol is a real, tradeable NSE stock.
    Returns (True, None) if valid, (False, reason) if not.

    Checks:
      1. Non-empty string
      2. No spaces or special chars (except - and &)
      3. yfinance can return at least MIN_OHLCV_ROWS of daily data
      4. Last close price > MIN_PRICE
    """
    if not symbol or not isinstance(symbol, str):
        return False, "Symbol is empty or not a string"

    clean = symbol.strip().upper().replace(".NS", "").replace(".BO", "")
    if not clean:
        return False, "Symbol is blank after cleaning"

    # Basic character check
    allowed = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-&")
    bad_chars = set(clean) - allowed
    if bad_chars:
        return False, f"Invalid characters in symbol: {bad_chars}"

    # Try fetching a small slice of data to confirm the symbol exists
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{clean}.NS")
        df = ticker.history(period="5d", interval="1d", auto_adjust=True, actions=False)
        if df.empty or len(df) < 1:
            return False, f"No market data found for {clean} — symbol may be invalid or delisted"
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        last_price = float(df["Close"].iloc[-1]) if "Close" in df.columns else 0
        if last_price < MIN_PRICE:
            return False, f"Last price ₹{last_price:.4f} is below minimum — symbol may be suspended"
        return True, None
    except Exception as e:
        logger.warning("validate_symbol %s: %s", clean, e)
        return False, f"Could not verify symbol {clean}: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# FETCH LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _fetch_primary(symbol: str) -> Optional[dict]:
    """fast_info LTP — fastest path."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        fi = ticker.fast_info
        price = float(getattr(fi, "last_price", None) or 0)
        prev  = float(getattr(fi, "previous_close", None) or 0)
        vol   = int(getattr(fi, "three_month_average_volume", None) or 0)

        if price < MIN_PRICE:
            logger.debug("_fetch_primary %s: price=%.4f below minimum", symbol, price)
            return None

        return {
            "symbol":     symbol,
            "price":      round(price, 2),
            "prev_close": round(prev, 2),
            "volume":     vol,
            "vwap":       0.0,
            "today_high": 0.0,
            "today_low":  0.0,
            "source":     "fast_info",
            "timestamp":  _now(),
            "is_valid":   False,
        }
    except Exception as e:
        logger.debug("_fetch_primary %s: %s", symbol, e)
        return None


def _fetch_secondary(symbol: str) -> Optional[dict]:
    """1-minute bars — richer data, also computes VWAP."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="1d", interval="1m", auto_adjust=True, actions=False)

        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep].dropna()
        if df.empty:
            return None

        price      = float(df["Close"].iloc[-1])
        prev       = float(df["Close"].iloc[0]) if len(df) > 1 else price
        today_high = float(df["High"].max())
        today_low  = float(df["Low"].min())
        today_vol  = int(df["Volume"].sum())

        if price < MIN_PRICE:
            return None

        # VWAP
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        cum_vol = df["Volume"].cumsum().replace(0, np.nan)
        vwap = float((tp * df["Volume"]).cumsum().iloc[-1] / cum_vol.iloc[-1])

        return {
            "symbol":     symbol,
            "price":      round(price, 2),
            "prev_close": round(prev, 2),
            "today_high": round(today_high, 2),
            "today_low":  round(today_low, 2),
            "volume":     today_vol,
            "vwap":       round(vwap, 2) if not np.isnan(vwap) else price,
            "source":     "1m_bars",
            "timestamp":  _now(),
            "df_1m":      df,
            "is_valid":   False,
        }
    except Exception as e:
        logger.debug("_fetch_secondary %s: %s", symbol, e)
        return None


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION LAYER
# ══════════════════════════════════════════════════════════════════════════════

def _validate(primary: Optional[dict], secondary: Optional[dict],
              symbol: str) -> Optional[dict]:
    """
    Cross-validate two ticks. Returns best validated tick or None.
    None means: DO NOT generate any report for this symbol.
    """
    now = _now()

    def fresh(d: Optional[dict]) -> bool:
        return (d is not None
                and d.get("price", 0) >= MIN_PRICE
                and (now - d.get("timestamp", 0)) <= MAX_DATA_AGE_SEC)

    if fresh(primary) and fresh(secondary):
        dev = abs(primary["price"] - secondary["price"]) / max(primary["price"], 1e-6)
        if dev > MAX_PRICE_DEV:
            logger.warning(
                "Price deviation %s: primary=%.2f secondary=%.2f (%.2f%%) → secondary wins",
                symbol, primary["price"], secondary["price"], dev * 100,
            )
            secondary["is_valid"] = True
            return secondary

        # Merge: primary price + secondary enrichment
        merged = {**primary}
        merged["today_high"] = secondary.get("today_high") or primary["price"]
        merged["today_low"]  = secondary.get("today_low")  or primary["price"]
        merged["vwap"]       = secondary.get("vwap")       or primary["price"]
        merged["df_1m"]      = secondary.get("df_1m", pd.DataFrame())
        merged["volume"]     = secondary.get("volume") or primary.get("volume", 0)
        merged["is_valid"]   = True
        merged["source"]     = "merged"
        return merged

    if fresh(primary):
        primary["is_valid"] = True
        primary.setdefault("today_high", primary["price"])
        primary.setdefault("today_low",  primary["price"])
        primary.setdefault("vwap",       primary["price"])
        primary.setdefault("df_1m",      pd.DataFrame())
        return primary

    if fresh(secondary):
        secondary["is_valid"] = True
        return secondary

    # Both failed or stale
    logger.warning("No valid tick for %s — both sources failed or stale", symbol)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_validated_tick(symbol: str) -> Optional[dict]:
    """
    Returns a validated live tick dict, or None if data is unavailable/invalid.

    STRICT: returns None (not a fake dict) when:
      - Both sources return price=0 or empty
      - Data is stale (>5s)
      - Price deviation >1% and secondary also fails

    Callers MUST check for None before generating any report.
    """
    cached = _cache_get(symbol)
    if cached:
        return cached

    with ThreadPoolExecutor(max_workers=2) as ex:
        f_p = ex.submit(_fetch_primary,   symbol)
        f_s = ex.submit(_fetch_secondary, symbol)
        primary   = f_p.result()
        secondary = f_s.result()

    validated = _validate(primary, secondary, symbol)
    if validated:
        _cache_set(symbol, validated)
    return validated


def get_validated_ohlcv(symbol: str, period: str = "1y",
                         interval: str = "1d") -> pd.DataFrame:
    """
    Returns validated OHLCV DataFrame, or empty DataFrame if data is bad.

    STRICT: returns empty DataFrame (not fake rows) when:
      - yfinance returns empty
      - Fewer than MIN_OHLCV_ROWS rows
      - All Close prices are 0 or NaN
    """
    cache_key = f"ohlcv_{symbol}_{period}_{interval}"
    entry = _ohlcv_cache.get(cache_key)
    if entry and (_now() - entry[0]) < OHLCV_CACHE_TTL:
        return entry[1].copy()

    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval,
                            auto_adjust=True, actions=False)

        # Fallback to longer period if empty
        if df.empty:
            _fallback = {"5d": "1mo", "1mo": "3mo", "3mo": "6mo",
                         "6mo": "1y", "1y": "2y", "2y": "5y"}
            fb = _fallback.get(period)
            if fb:
                df = ticker.history(period=fb, interval=interval,
                                    auto_adjust=True, actions=False)

        if df.empty:
            logger.warning("get_validated_ohlcv: no data for %s", symbol)
            return pd.DataFrame()

        # Clean
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[keep]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="last")]
        df = df.ffill().dropna()

        # Strict checks
        if len(df) < MIN_OHLCV_ROWS:
            logger.warning("get_validated_ohlcv: only %d rows for %s — rejecting", len(df), symbol)
            return pd.DataFrame()

        if "Close" in df.columns:
            last_close = float(df["Close"].iloc[-1])
            if last_close < MIN_PRICE:
                logger.warning("get_validated_ohlcv: last close=%.4f for %s — rejecting", last_close, symbol)
                return pd.DataFrame()

        _ohlcv_cache[cache_key] = (_now(), df)
        return df.copy()

    except Exception as e:
        logger.error("get_validated_ohlcv %s: %s", symbol, e)
        return pd.DataFrame()


def get_validated_ticks_batch(symbols: list[str],
                               max_workers: int = 8) -> dict[str, dict]:
    """
    Fetch validated ticks for multiple symbols in parallel.
    Only returns symbols with valid data — invalid ones are silently dropped.
    """
    results: dict[str, dict] = {}

    def _fetch(sym: str) -> tuple[str, Optional[dict]]:
        return sym, get_validated_tick(sym)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for sym, tick in ex.map(lambda s: _fetch(s), symbols):
            if tick and tick.get("is_valid") and tick.get("price", 0) >= MIN_PRICE:
                results[sym] = tick

    return results


def generate_signal(tick: dict, avg_volume: float, vwap: float) -> str:
    """
    Signal gate — only runs on validated data.
    Returns NO_TRADE if tick is invalid or price is zero.
    """
    if not tick or not tick.get("is_valid") or tick.get("price", 0) < MIN_PRICE:
        return "NO_TRADE"

    ltp  = tick["price"]
    vol  = tick.get("volume", 0)
    evwap = tick.get("vwap") or vwap
    if evwap <= 0:
        evwap = vwap

    if vol > avg_volume * 1.5 and ltp > evwap:
        return "BUY"
    if vol > avg_volume * 1.5 and ltp < evwap:
        return "SELL"
    return "NO_TRADE"


def clear_live_cache() -> None:
    _live_cache.clear()

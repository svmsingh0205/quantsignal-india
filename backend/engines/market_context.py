"""
Market Context Engine — 20 Indicators (10 Indian + 10 Global)
Fetches all macro/market indicators and produces:
  - market_bias_score  : 0–1 (1 = strong bullish, 0 = strong bearish)
  - market_bias_label  : STRONG BULL / BULL / NEUTRAL / BEAR / STRONG BEAR
  - position_size_mult : 0.25–1.0 (scale down in bearish conditions)
  - indicator_table    : list of dicts for UI display
  - intraday_filter    : bool — True = safe to trade intraday
  - swing_filter       : bool — True = safe for swing trades

Used by:
  - _scan_one() in live_trader.py to boost/penalise signal scores
  - Market Context tab in the UI
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ── IST offset ────────────────────────────────────────────────────────────────
_IST = timezone(timedelta(hours=5, minutes=30))
def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)

# ── Ticker map ────────────────────────────────────────────────────────────────
_TICKERS = {
    # Indian
    "NIFTY50":    "^NSEI",
    "BANKNIFTY":  "^NSEBANK",
    "INDIA_VIX":  "^INDIAVIX",
    "NIFTY_IT":   "^CNXIT",
    "NIFTY_PSU":  "^CNXPSUBANK",
    # Global
    "SP500":      "^GSPC",
    "NASDAQ":     "^IXIC",
    "VIX":        "^VIX",
    "CRUDE_OIL":  "CL=F",
    "GOLD":       "GC=F",
    "DXY":        "DX-Y.NYB",
    "USDINR":     "USDINR=X",
    "BITCOIN":    "BTC-USD",
    "NIKKEI":     "^N225",
    "HANGSENG":   "^HSI",
}

# ── In-process cache (15-min TTL for macro data) ──────────────────────────────
_ctx_cache: dict[str, tuple[datetime, dict]] = {}
CTX_CACHE_TTL = timedelta(minutes=15)


def _fetch_ticker(key: str, ticker: str, period: str = "5d") -> Optional[pd.DataFrame]:
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval="1d")
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
        return df[keep].dropna()
    except Exception as e:
        logger.debug(f"_fetch_ticker {key}/{ticker}: {e}")
        return None


def _pct_change(df: pd.DataFrame, n: int = 1) -> float:
    if df is None or len(df) < n + 1:
        return 0.0
    return float((df["Close"].iloc[-1] / df["Close"].iloc[-n] - 1) * 100)


def _latest(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return 0.0
    return float(df["Close"].iloc[-1])


def _trend(df: pd.DataFrame, short: int = 5, long: int = 20) -> str:
    """Simple trend: UP / DOWN / NEUTRAL based on short vs long MA."""
    if df is None or len(df) < long:
        return "NEUTRAL"
    s = float(df["Close"].tail(short).mean())
    l = float(df["Close"].tail(long).mean())
    if s > l * 1.005:
        return "UP"
    if s < l * 0.995:
        return "DOWN"
    return "NEUTRAL"


def _rsi(df: pd.DataFrame, period: int = 14) -> float:
    if df is None or len(df) < period + 1:
        return 50.0
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


# ── Individual indicator scorers ──────────────────────────────────────────────

def _score_nifty(df) -> tuple[float, str, str]:
    """NIFTY 50 trend — HH-HL structure proxy via 5/20 MA."""
    trend = _trend(df)
    chg1d = _pct_change(df, 1)
    chg5d = _pct_change(df, 5)
    rsi = _rsi(df)
    score = 0.5
    if trend == "UP":   score += 0.20
    if trend == "DOWN": score -= 0.20
    score += np.clip(chg5d / 10, -0.15, 0.15)
    if rsi < 40: score += 0.05
    if rsi > 70: score -= 0.05
    note = f"Trend: {trend} | 1D: {chg1d:+.2f}% | 5D: {chg5d:+.2f}% | RSI: {rsi:.0f}"
    return float(np.clip(score, 0, 1)), trend, note


def _score_banknifty(df) -> tuple[float, str, str]:
    trend = _trend(df)
    chg1d = _pct_change(df, 1)
    score = 0.5 + (0.20 if trend == "UP" else -0.20 if trend == "DOWN" else 0)
    note = f"Trend: {trend} | 1D: {chg1d:+.2f}%"
    return float(np.clip(score, 0, 1)), trend, note


def _score_india_vix(df) -> tuple[float, str, str]:
    """Low VIX = trending market = bullish. High VIX = choppy = bearish."""
    vix = _latest(df)
    chg1d = _pct_change(df, 1)
    if vix == 0:
        return 0.5, "NEUTRAL", "VIX data unavailable"
    if vix < 12:   score, label = 0.85, "VERY LOW (trending)"
    elif vix < 16: score, label = 0.70, "LOW (bullish)"
    elif vix < 20: score, label = 0.55, "MODERATE"
    elif vix < 25: score, label = 0.35, "ELEVATED (caution)"
    else:          score, label = 0.15, "HIGH (avoid longs)"
    note = f"VIX: {vix:.1f} ({label}) | 1D chg: {chg1d:+.2f}%"
    trend = "DOWN" if chg1d > 2 else ("UP" if chg1d < -2 else "NEUTRAL")  # VIX falling = bullish
    return float(score), trend, note


def _score_sp500(df) -> tuple[float, str, str]:
    trend = _trend(df)
    chg1d = _pct_change(df, 1)
    chg5d = _pct_change(df, 5)
    score = 0.5
    if trend == "UP":   score += 0.20
    if trend == "DOWN": score -= 0.20
    score += np.clip(chg5d / 10, -0.15, 0.15)
    note = f"Trend: {trend} | 1D: {chg1d:+.2f}% | 5D: {chg5d:+.2f}%"
    return float(np.clip(score, 0, 1)), trend, note


def _score_vix(df) -> tuple[float, str, str]:
    """US VIX — fear gauge."""
    vix = _latest(df)
    if vix == 0:
        return 0.5, "NEUTRAL", "US VIX unavailable"
    if vix < 15:   score, label = 0.80, "Risk-On"
    elif vix < 20: score, label = 0.60, "Calm"
    elif vix < 25: score, label = 0.40, "Caution"
    elif vix < 30: score, label = 0.25, "Fear"
    else:          score, label = 0.10, "Extreme Fear"
    note = f"US VIX: {vix:.1f} ({label})"
    trend = "NEUTRAL"
    return float(score), trend, note


def _score_crude(df) -> tuple[float, str, str]:
    """Crude oil — India is net importer. Rising crude = bearish India."""
    chg1d = _pct_change(df, 1)
    chg5d = _pct_change(df, 5)
    price = _latest(df)
    # Rising crude = bearish for India
    score = 0.5 - np.clip(chg5d / 10, -0.25, 0.25)
    trend = "UP" if chg5d > 2 else ("DOWN" if chg5d < -2 else "NEUTRAL")
    impact = "Bearish India" if chg5d > 3 else ("Bullish India" if chg5d < -3 else "Neutral")
    note = f"Crude: ${price:.1f} | 5D: {chg5d:+.2f}% → {impact}"
    return float(np.clip(score, 0, 1)), trend, note


def _score_gold(df) -> tuple[float, str, str]:
    """Gold rising = risk-off = mixed/bearish for equities."""
    chg5d = _pct_change(df, 5)
    price = _latest(df)
    # Gold up = fear = slightly bearish equities
    score = 0.5 - np.clip(chg5d / 15, -0.15, 0.15)
    trend = "UP" if chg5d > 1 else ("DOWN" if chg5d < -1 else "NEUTRAL")
    note = f"Gold: ${price:.0f} | 5D: {chg5d:+.2f}% | {'Risk-Off' if chg5d > 2 else 'Neutral'}"
    return float(np.clip(score, 0, 1)), trend, note


def _score_dxy(df) -> tuple[float, str, str]:
    """Strong DXY = FII outflows from India = bearish."""
    chg5d = _pct_change(df, 5)
    price = _latest(df)
    score = 0.5 - np.clip(chg5d / 5, -0.25, 0.25)
    trend = "UP" if chg5d > 0.5 else ("DOWN" if chg5d < -0.5 else "NEUTRAL")
    impact = "FII Outflow Risk" if chg5d > 1 else ("FII Inflow Positive" if chg5d < -1 else "Neutral")
    note = f"DXY: {price:.1f} | 5D: {chg5d:+.2f}% → {impact}"
    return float(np.clip(score, 0, 1)), trend, note


def _score_usdinr(df) -> tuple[float, str, str]:
    """Rupee depreciation (USD/INR rising) = bearish India."""
    chg5d = _pct_change(df, 5)
    price = _latest(df)
    score = 0.5 - np.clip(chg5d / 3, -0.20, 0.20)
    trend = "UP" if chg5d > 0.3 else ("DOWN" if chg5d < -0.3 else "NEUTRAL")
    note = f"USD/INR: {price:.2f} | 5D: {chg5d:+.2f}% | {'Rupee Weak' if chg5d > 0.5 else 'Rupee Stable'}"
    return float(np.clip(score, 0, 1)), trend, note


def _score_bitcoin(df) -> tuple[float, str, str]:
    """BTC rising = risk-on sentiment = bullish equities."""
    chg5d = _pct_change(df, 5)
    price = _latest(df)
    score = 0.5 + np.clip(chg5d / 20, -0.15, 0.15)
    trend = "UP" if chg5d > 3 else ("DOWN" if chg5d < -3 else "NEUTRAL")
    note = f"BTC: ${price:,.0f} | 5D: {chg5d:+.2f}% | {'Risk-On' if chg5d > 5 else 'Risk-Off' if chg5d < -5 else 'Neutral'}"
    return float(np.clip(score, 0, 1)), trend, note


def _score_nasdaq(df) -> tuple[float, str, str]:
    """NASDAQ trend impacts Indian IT stocks."""
    trend = _trend(df)
    chg5d = _pct_change(df, 5)
    score = 0.5 + (0.15 if trend == "UP" else -0.15 if trend == "DOWN" else 0)
    note = f"Trend: {trend} | 5D: {chg5d:+.2f}% | IT sector impact"
    return float(np.clip(score, 0, 1)), trend, note


# ── Indicator registry ────────────────────────────────────────────────────────
# (key, display_name, category, weight, scorer_fn, ticker_key, period)
_INDICATORS = [
    # Indian (weight sum = 0.55)
    ("nifty50",    "NIFTY 50",          "🇮🇳 Indian",  0.15, _score_nifty,      "NIFTY50",   "1mo"),
    ("banknifty",  "BANK NIFTY",        "🇮🇳 Indian",  0.10, _score_banknifty,  "BANKNIFTY", "1mo"),
    ("india_vix",  "INDIA VIX",         "🇮🇳 Indian",  0.12, _score_india_vix,  "INDIA_VIX", "1mo"),
    ("usdinr",     "USD/INR (Rupee)",   "🇮🇳 Indian",  0.08, _score_usdinr,     "USDINR",    "1mo"),
    ("nifty_it",   "NIFTY IT",          "🇮🇳 Indian",  0.05, _score_nifty,      "NIFTY_IT",  "1mo"),
    ("nifty_psu",  "NIFTY PSU BANK",    "🇮🇳 Indian",  0.05, _score_banknifty,  "NIFTY_PSU", "1mo"),
    # Global (weight sum = 0.45)
    ("sp500",      "S&P 500",           "🌍 Global",   0.12, _score_sp500,      "SP500",     "1mo"),
    ("us_vix",     "US VIX (Fear)",     "🌍 Global",   0.10, _score_vix,        "VIX",       "1mo"),
    ("crude",      "Crude Oil",         "🌍 Global",   0.08, _score_crude,      "CRUDE_OIL", "1mo"),
    ("gold",       "Gold",              "🌍 Global",   0.04, _score_gold,       "GOLD",      "1mo"),
    ("dxy",        "DXY (USD Index)",   "🌍 Global",   0.06, _score_dxy,        "DXY",       "1mo"),
    ("bitcoin",    "Bitcoin",           "🌍 Global",   0.03, _score_bitcoin,    "BITCOIN",   "1mo"),
    ("nasdaq",     "NASDAQ",            "🌍 Global",   0.02, _score_nasdaq,     "NASDAQ",    "1mo"),
]


# ── Main public function ──────────────────────────────────────────────────────

def get_market_context(force_refresh: bool = False) -> dict:
    """
    Fetch all 20 indicators and return composite market context.

    Returns:
        market_bias_score   : float 0–1
        market_bias_label   : str
        position_size_mult  : float 0.25–1.0
        intraday_filter     : bool
        swing_filter        : bool
        indicators          : list[dict] — one per indicator for UI
        indian_score        : float
        global_score        : float
        fetched_at          : str (IST)
    """
    cache_key = "market_context"
    if not force_refresh and cache_key in _ctx_cache:
        ts, data = _ctx_cache[cache_key]
        if (_now_utc() - ts) < CTX_CACHE_TTL:
            return data

    # Parallel fetch all tickers
    dfs: dict[str, Optional[pd.DataFrame]] = {}

    def _fetch(key: str, ticker: str, period: str):
        return key, _fetch_ticker(key, ticker, period)

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {
            ex.submit(_fetch, key, _TICKERS[ticker_key], period): key
            for key, _, _, _, _, ticker_key, period in _INDICATORS
            if ticker_key in _TICKERS
        }
        for fut in as_completed(futs):
            key, df = fut.result()
            dfs[key] = df

    # Score each indicator
    indicators = []
    weighted_sum = 0.0
    total_weight = 0.0
    indian_sum = indian_w = 0.0
    global_sum = global_w = 0.0

    for key, name, category, weight, scorer, ticker_key, _ in _INDICATORS:
        df = dfs.get(key)
        try:
            score, trend, note = scorer(df)
        except Exception:
            score, trend, note = 0.5, "NEUTRAL", "Data unavailable"

        signal = "BULLISH" if score >= 0.60 else ("BEARISH" if score <= 0.40 else "NEUTRAL")
        impact = "🟢 Positive" if score >= 0.60 else ("🔴 Negative" if score <= 0.40 else "🟡 Neutral")

        indicators.append({
            "key": key, "name": name, "category": category,
            "score": round(score, 3), "weight": weight,
            "trend": trend, "signal": signal, "impact": impact,
            "note": note,
        })

        weighted_sum += score * weight
        total_weight += weight

        if "Indian" in category:
            indian_sum += score * weight
            indian_w += weight
        else:
            global_sum += score * weight
            global_w += weight

    bias = weighted_sum / total_weight if total_weight > 0 else 0.5
    indian_score = indian_sum / indian_w if indian_w > 0 else 0.5
    global_score = global_sum / global_w if global_w > 0 else 0.5

    # Label
    if bias >= 0.72:   label = "STRONG BULL 🚀"
    elif bias >= 0.58: label = "BULL 📈"
    elif bias >= 0.42: label = "NEUTRAL ↔️"
    elif bias >= 0.28: label = "BEAR 📉"
    else:              label = "STRONG BEAR 🔻"

    # Position size multiplier — scale down in bearish conditions
    if bias >= 0.65:   psm = 1.00
    elif bias >= 0.55: psm = 0.85
    elif bias >= 0.45: psm = 0.65
    elif bias >= 0.35: psm = 0.40
    else:              psm = 0.25

    # India VIX check for intraday filter
    vix_ind = next((i for i in indicators if i["key"] == "india_vix"), None)
    vix_score = vix_ind["score"] if vix_ind else 0.5
    intraday_filter = bias >= 0.45 and vix_score >= 0.40   # avoid intraday in high VIX + bearish
    swing_filter    = bias >= 0.40 and indian_score >= 0.45

    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y %I:%M %p IST")

    result = {
        "market_bias_score":  round(bias, 4),
        "market_bias_label":  label,
        "position_size_mult": round(psm, 2),
        "intraday_filter":    intraday_filter,
        "swing_filter":       swing_filter,
        "indian_score":       round(indian_score, 4),
        "global_score":       round(global_score, 4),
        "indicators":         indicators,
        "fetched_at":         ist_now,
    }

    _ctx_cache[cache_key] = (_now_utc(), result)
    return result


def get_market_bias_score(cached_ok: bool = True) -> float:
    """Quick accessor — returns just the bias score (0–1)."""
    try:
        ctx = get_market_context(force_refresh=not cached_ok)
        return ctx["market_bias_score"]
    except Exception:
        return 0.5


def get_position_size_multiplier() -> float:
    """Returns position size multiplier based on current market context."""
    try:
        ctx = get_market_context()
        return ctx["position_size_mult"]
    except Exception:
        return 0.75

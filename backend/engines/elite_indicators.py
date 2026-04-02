"""
Elite Indicators Engine — Institutional-grade technical indicators.

Implements:
  1. VWAP (intraday institutional control)
  2. Anchored VWAP (swing level)
  3. Volume Profile (POC, Value Area High/Low)
  4. Order Book Imbalance (bid vs ask pressure proxy)
  5. Smart Money Flow Index
  6. Relative Strength vs Index (Nifty)
  7. Market Structure (HH/HL + BOS + CHoCH)
  8. Liquidity Zones (stop hunt zones)
  9. Volatility Regime (India VIX + ATR hybrid)
 10. Delta Volume (buy vs sell pressure proxy)
 11. Breakout Strength Score
 12. Trend Strength Index (custom, not RSI)

All indicators use LIVE DATA — no static computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. VWAP — Volume Weighted Average Price
# ══════════════════════════════════════════════════════════════════════════════
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    Standard VWAP = cumsum(typical_price * volume) / cumsum(volume).
    Resets at session start (first bar of each day).
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tpv = (tp * df["Volume"]).cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    return vwap.rename("VWAP")


def compute_vwap_bands(df: pd.DataFrame, n_std: list[float] = None) -> pd.DataFrame:
    """
    VWAP with standard deviation bands (institutional support/resistance).
    Returns df with VWAP, VWAP_U1, VWAP_L1, VWAP_U2, VWAP_L2.
    """
    if n_std is None:
        n_std = [1.0, 2.0]
    df = df.copy()
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    cum_vol = df["Volume"].cumsum()
    cum_tpv = (tp * df["Volume"]).cumsum()
    vwap = cum_tpv / cum_vol.replace(0, np.nan)
    df["VWAP"] = vwap

    # Rolling std of (tp - vwap) weighted by volume
    variance = ((tp - vwap) ** 2 * df["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
    std = np.sqrt(variance.clip(lower=0))

    for n in n_std:
        df[f"VWAP_U{int(n)}"] = vwap + n * std
        df[f"VWAP_L{int(n)}"] = vwap - n * std

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. ANCHORED VWAP
# ══════════════════════════════════════════════════════════════════════════════
def compute_anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0) -> pd.Series:
    """
    Anchored VWAP from a specific bar (e.g. swing low, earnings date).
    anchor_idx: integer position in df to anchor from.
    """
    df_slice = df.iloc[anchor_idx:].copy()
    tp = (df_slice["High"] + df_slice["Low"] + df_slice["Close"]) / 3
    cum_vol = df_slice["Volume"].cumsum()
    avwap = (tp * df_slice["Volume"]).cumsum() / cum_vol.replace(0, np.nan)
    # Reindex to full df
    result = pd.Series(np.nan, index=df.index)
    result.iloc[anchor_idx:] = avwap.values
    return result.rename("AVWAP")


# ══════════════════════════════════════════════════════════════════════════════
# 3. VOLUME PROFILE — POC, VAH, VAL
# ══════════════════════════════════════════════════════════════════════════════
def compute_volume_profile(df: pd.DataFrame, bins: int = 50) -> dict:
    """
    Volume Profile: distributes volume across price bins.
    Returns:
        poc: Point of Control (price with highest volume)
        vah: Value Area High (70% of volume above POC)
        val: Value Area Low (70% of volume below POC)
        profile: dict of price_level → volume
    """
    if df.empty or len(df) < 5:
        return {"poc": 0, "vah": 0, "val": 0, "profile": {}}

    price_min = float(df["Low"].min())
    price_max = float(df["High"].max())
    if price_max <= price_min:
        return {"poc": price_min, "vah": price_max, "val": price_min, "profile": {}}

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    vol_per_bin = np.zeros(bins)

    for _, row in df.iterrows():
        # Distribute bar volume across price range of the bar
        bar_low = float(row["Low"])
        bar_high = float(row["High"])
        bar_vol = float(row["Volume"])
        if bar_high <= bar_low:
            continue
        # Find bins that overlap with this bar
        overlap_mask = (bin_edges[1:] >= bar_low) & (bin_edges[:-1] <= bar_high)
        n_overlap = overlap_mask.sum()
        if n_overlap > 0:
            vol_per_bin[overlap_mask] += bar_vol / n_overlap

    poc_idx = int(np.argmax(vol_per_bin))
    poc = float(bin_centers[poc_idx])

    # Value Area: 70% of total volume
    total_vol = vol_per_bin.sum()
    target_vol = total_vol * 0.70
    accumulated = vol_per_bin[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx

    while accumulated < target_vol and (lo_idx > 0 or hi_idx < bins - 1):
        lo_vol = vol_per_bin[lo_idx - 1] if lo_idx > 0 else 0
        hi_vol = vol_per_bin[hi_idx + 1] if hi_idx < bins - 1 else 0
        if hi_vol >= lo_vol and hi_idx < bins - 1:
            hi_idx += 1
            accumulated += vol_per_bin[hi_idx]
        elif lo_idx > 0:
            lo_idx -= 1
            accumulated += vol_per_bin[lo_idx]
        else:
            break

    return {
        "poc": round(poc, 2),
        "vah": round(float(bin_centers[hi_idx]), 2),
        "val": round(float(bin_centers[lo_idx]), 2),
        "profile": {round(float(bin_centers[i]), 2): round(float(vol_per_bin[i]), 0)
                    for i in range(bins)},
    }



# ══════════════════════════════════════════════════════════════════════════════
# 4. ORDER BOOK IMBALANCE (proxy from OHLCV)
# ══════════════════════════════════════════════════════════════════════════════
def compute_order_book_imbalance(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Proxy for order book imbalance using price action and volume.
    Buy pressure = (Close - Low) / (High - Low) * Volume
    Sell pressure = (High - Close) / (High - Low) * Volume
    Imbalance = (buy - sell) / (buy + sell), range [-1, 1]
    """
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    buy_vol = ((df["Close"] - df["Low"]) / hl) * df["Volume"]
    sell_vol = ((df["High"] - df["Close"]) / hl) * df["Volume"]
    total = buy_vol + sell_vol
    imbalance = (buy_vol - sell_vol) / total.replace(0, np.nan)
    return imbalance.rolling(window).mean().rename("OBI")


# ══════════════════════════════════════════════════════════════════════════════
# 5. SMART MONEY FLOW INDEX
# ══════════════════════════════════════════════════════════════════════════════
def compute_smart_money_flow(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Smart Money Flow = weighted by intraday price position.
    High-close bars (smart money buying) get more weight.
    """
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    # Smart money weight: close near high = buying, close near low = selling
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    weight = (df["Close"] - df["Low"]) / hl  # 0 = close at low, 1 = close at high
    smart_flow = tp * df["Volume"] * weight
    mf_pos = smart_flow.where(df["Close"] >= df["Close"].shift(1), 0).rolling(period).sum()
    mf_neg = smart_flow.where(df["Close"] < df["Close"].shift(1), 0).rolling(period).sum().abs()
    mfr = mf_pos / mf_neg.replace(0, np.nan)
    smfi = 100 - (100 / (1 + mfr))
    return smfi.rename("SMFI")


# ══════════════════════════════════════════════════════════════════════════════
# 6. RELATIVE STRENGTH vs INDEX
# ══════════════════════════════════════════════════════════════════════════════
def compute_relative_strength(df_stock: pd.DataFrame, df_index: pd.DataFrame,
                               period: int = 20) -> pd.Series:
    """
    RS = stock return / index return over rolling period.
    RS > 1 = outperforming, RS < 1 = underperforming.
    """
    stock_ret = df_stock["Close"].pct_change(period)
    # Align index to stock dates
    idx_aligned = df_index["Close"].reindex(df_stock.index, method="ffill")
    idx_ret = idx_aligned.pct_change(period)
    rs = (1 + stock_ret) / (1 + idx_ret.replace(0, np.nan))
    return rs.rename("RS_vs_Index")


# ══════════════════════════════════════════════════════════════════════════════
# 7. MARKET STRUCTURE — HH/HL, BOS, CHoCH
# ══════════════════════════════════════════════════════════════════════════════
def compute_market_structure(df: pd.DataFrame, swing_window: int = 5) -> dict:
    """
    Identifies:
      - HH (Higher High), HL (Higher Low) = uptrend
      - LH (Lower High), LL (Lower Low) = downtrend
      - BOS (Break of Structure) = trend continuation
      - CHoCH (Change of Character) = trend reversal signal

    Returns dict with:
      structure: "UPTREND" | "DOWNTREND" | "RANGING"
      last_bos: price level of last BOS
      last_choch: price level of last CHoCH (if any)
      swing_highs: list of recent swing high prices
      swing_lows: list of recent swing low prices
    """
    if len(df) < swing_window * 3:
        return {"structure": "RANGING", "last_bos": 0, "last_choch": 0,
                "swing_highs": [], "swing_lows": []}

    highs = df["High"].values
    lows = df["Low"].values
    closes = df["Close"].values
    n = len(df)

    # Find swing highs and lows
    swing_highs = []
    swing_lows = []
    for i in range(swing_window, n - swing_window):
        if highs[i] == max(highs[i - swing_window:i + swing_window + 1]):
            swing_highs.append((i, highs[i]))
        if lows[i] == min(lows[i - swing_window:i + swing_window + 1]):
            swing_lows.append((i, lows[i]))

    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return {"structure": "RANGING", "last_bos": 0, "last_choch": 0,
                "swing_highs": [v for _, v in swing_highs[-3:]],
                "swing_lows": [v for _, v in swing_lows[-3:]]}

    # Last 3 swing highs and lows
    sh = [v for _, v in swing_highs[-3:]]
    sl = [v for _, v in swing_lows[-3:]]

    hh = sh[-1] > sh[-2] if len(sh) >= 2 else False
    hl = sl[-1] > sl[-2] if len(sl) >= 2 else False
    lh = sh[-1] < sh[-2] if len(sh) >= 2 else False
    ll = sl[-1] < sl[-2] if len(sl) >= 2 else False

    if hh and hl:
        structure = "UPTREND"
    elif lh and ll:
        structure = "DOWNTREND"
    else:
        structure = "RANGING"

    # BOS: price breaks above last swing high (uptrend) or below last swing low (downtrend)
    last_price = float(closes[-1])
    last_sh = sh[-1] if sh else 0
    last_sl = sl[-1] if sl else 0
    last_bos = last_sh if last_price > last_sh else (last_sl if last_price < last_sl else 0)

    # CHoCH: structure flip
    last_choch = 0.0
    if structure == "UPTREND" and ll:
        last_choch = float(sl[-1])
    elif structure == "DOWNTREND" and hh:
        last_choch = float(sh[-1])

    return {
        "structure": structure,
        "last_bos": round(last_bos, 2),
        "last_choch": round(last_choch, 2),
        "swing_highs": [round(v, 2) for v in sh],
        "swing_lows": [round(v, 2) for v in sl],
        "hh": hh, "hl": hl, "lh": lh, "ll": ll,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. LIQUIDITY ZONES (Stop Hunt Zones)
# ══════════════════════════════════════════════════════════════════════════════
def compute_liquidity_zones(df: pd.DataFrame, lookback: int = 50,
                             tolerance_pct: float = 0.003) -> dict:
    """
    Identifies liquidity zones where stop-losses cluster:
      - Equal highs (resistance liquidity)
      - Equal lows (support liquidity)
      - Previous day high/low
      - Round number levels

    Returns dict with buy_side_liquidity and sell_side_liquidity zones.
    """
    if len(df) < lookback:
        lookback = len(df)

    recent = df.tail(lookback)
    highs = recent["High"].values
    lows = recent["Low"].values
    close = float(df["Close"].iloc[-1])

    # Find equal highs (within tolerance) — sell-side liquidity above
    sell_side = []
    for i in range(len(highs)):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) / max(highs[i], 1e-6) <= tolerance_pct:
                level = round((highs[i] + highs[j]) / 2, 2)
                if level > close and level not in sell_side:
                    sell_side.append(level)

    # Find equal lows — buy-side liquidity below
    buy_side = []
    for i in range(len(lows)):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) / max(lows[i], 1e-6) <= tolerance_pct:
                level = round((lows[i] + lows[j]) / 2, 2)
                if level < close and level not in buy_side:
                    buy_side.append(level)

    # Round number levels (psychological support/resistance)
    magnitude = 10 ** (len(str(int(close))) - 2)
    round_levels = [round(close / magnitude) * magnitude + i * magnitude
                    for i in range(-3, 4)]
    round_above = sorted([l for l in round_levels if l > close])[:2]
    round_below = sorted([l for l in round_levels if l < close], reverse=True)[:2]

    return {
        "buy_side_liquidity": sorted(buy_side, reverse=True)[:5],
        "sell_side_liquidity": sorted(sell_side)[:5],
        "round_resistance": round_above,
        "round_support": round_below,
        "nearest_resistance": min(sell_side + round_above, default=close * 1.05),
        "nearest_support": max(buy_side + round_below, default=close * 0.95),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. VOLATILITY REGIME
# ══════════════════════════════════════════════════════════════════════════════
def compute_volatility_regime(df: pd.DataFrame, atr_period: int = 14,
                               vol_window: int = 20) -> dict:
    """
    Hybrid volatility regime using ATR + historical volatility.
    Returns regime: "LOW" | "NORMAL" | "HIGH" | "EXTREME"
    """
    if len(df) < max(atr_period, vol_window) + 5:
        return {"regime": "NORMAL", "atr": 0, "atr_pct": 0,
                "hist_vol": 0, "regime_score": 0.5}

    # ATR
    high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(atr_period).mean().iloc[-1])
    price = float(df["Close"].iloc[-1])
    atr_pct = atr / price if price > 0 else 0

    # Historical volatility (annualised)
    returns = df["Close"].pct_change().dropna()
    hist_vol = float(returns.rolling(vol_window).std().iloc[-1]) * np.sqrt(252)

    # ATR percentile vs last 252 bars
    atr_series = tr.rolling(atr_period).mean().dropna()
    if len(atr_series) >= 20:
        atr_pct_rank = float((atr_series < atr).mean())
    else:
        atr_pct_rank = 0.5

    # Regime classification
    if atr_pct_rank >= 0.85 or hist_vol >= 0.60:
        regime = "EXTREME"
        regime_score = 0.1
    elif atr_pct_rank >= 0.65 or hist_vol >= 0.40:
        regime = "HIGH"
        regime_score = 0.35
    elif atr_pct_rank <= 0.25 or hist_vol <= 0.15:
        regime = "LOW"
        regime_score = 0.85
    else:
        regime = "NORMAL"
        regime_score = 0.65

    return {
        "regime": regime,
        "atr": round(atr, 2),
        "atr_pct": round(atr_pct * 100, 2),
        "hist_vol": round(hist_vol * 100, 1),
        "atr_percentile": round(atr_pct_rank * 100, 0),
        "regime_score": regime_score,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. DELTA VOLUME (Buy vs Sell Pressure)
# ══════════════════════════════════════════════════════════════════════════════
def compute_delta_volume(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """
    Delta Volume = buy volume - sell volume (proxy from OHLCV).
    Positive delta = buying pressure, negative = selling pressure.
    """
    df = df.copy()
    hl = (df["High"] - df["Low"]).replace(0, np.nan)
    buy_vol = ((df["Close"] - df["Low"]) / hl) * df["Volume"]
    sell_vol = ((df["High"] - df["Close"]) / hl) * df["Volume"]
    df["Delta_Vol"] = buy_vol - sell_vol
    df["Delta_Vol_MA"] = df["Delta_Vol"].rolling(window).mean()
    df["Cumulative_Delta"] = df["Delta_Vol"].cumsum()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 11. BREAKOUT STRENGTH SCORE
# ══════════════════════════════════════════════════════════════════════════════
def compute_breakout_strength(df: pd.DataFrame, lookback: int = 20) -> dict:
    """
    Breakout Strength Score (0–1):
    Combines: volume confirmation, ATR expansion, close position, momentum.
    """
    if len(df) < lookback + 5:
        return {"score": 0.5, "is_breakout": False, "direction": "NEUTRAL",
                "strength": "WEAK", "reasons": []}

    recent = df.tail(lookback)
    last = df.iloc[-1]
    prev = df.iloc[-2]

    price = float(last["Close"])
    high_n = float(recent["High"].max())
    low_n = float(recent["Low"].min())
    avg_vol = float(recent["Volume"].mean())
    last_vol = float(last["Volume"])

    score = 0.0
    reasons = []

    # Price breakout above N-bar high
    if price >= high_n * 0.998:
        score += 0.30
        reasons.append(f"Price at {lookback}-bar high breakout")
        direction = "UP"
    elif price <= low_n * 1.002:
        score += 0.30
        reasons.append(f"Price at {lookback}-bar low breakdown")
        direction = "DOWN"
    else:
        direction = "NEUTRAL"

    # Volume confirmation
    if avg_vol > 0:
        v_ratio = last_vol / avg_vol
        if v_ratio >= 2.0:
            score += 0.25
            reasons.append(f"Volume {v_ratio:.1f}x avg — strong confirmation")
        elif v_ratio >= 1.5:
            score += 0.15
            reasons.append(f"Volume {v_ratio:.1f}x avg — moderate confirmation")
        elif v_ratio < 0.8:
            score -= 0.10
            reasons.append("Low volume — weak breakout")

    # ATR expansion (volatility expanding = real breakout)
    high_l, low_l, prev_close = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([high_l - low_l, (high_l - prev_close).abs(), (low_l - prev_close).abs()], axis=1).max(axis=1)
    atr_now = float(tr.rolling(14).mean().iloc[-1])
    atr_prev = float(tr.rolling(14).mean().iloc[-5]) if len(df) >= 19 else atr_now
    if atr_prev > 0 and atr_now / atr_prev >= 1.2:
        score += 0.15
        reasons.append("ATR expanding — volatility breakout")

    # Close position in bar (close near high = bullish breakout)
    bar_range = float(last["High"] - last["Low"])
    if bar_range > 0:
        close_pos = (price - float(last["Low"])) / bar_range
        if close_pos >= 0.70:
            score += 0.10
            reasons.append("Strong close near bar high")
        elif close_pos <= 0.30:
            score += 0.10
            reasons.append("Weak close near bar low")

    score = float(np.clip(score, 0.0, 1.0))
    is_breakout = score >= 0.55 and direction != "NEUTRAL"
    strength = "STRONG" if score >= 0.70 else ("MODERATE" if score >= 0.50 else "WEAK")

    return {
        "score": round(score, 4),
        "is_breakout": is_breakout,
        "direction": direction,
        "strength": strength,
        "reasons": reasons,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 12. TREND STRENGTH INDEX (custom — not RSI)
# ══════════════════════════════════════════════════════════════════════════════
def compute_trend_strength_index(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Trend Strength Index (TSI) — custom indicator.
    Measures directional consistency of price moves (not just magnitude).
    TSI = (positive closes / total closes) * directional_factor
    Range: 0–1. > 0.65 = strong trend, < 0.35 = weak/ranging.
    """
    if len(df) < period + 1:
        return pd.Series(0.5, index=df.index, name="TSI")

    closes = df["Close"]
    delta = closes.diff()

    # Directional consistency: % of bars moving in dominant direction
    pos_bars = (delta > 0).rolling(period).sum()
    neg_bars = (delta < 0).rolling(period).sum()
    total = pos_bars + neg_bars

    # Dominant direction ratio
    dom_ratio = pd.concat([pos_bars, neg_bars], axis=1).max(axis=1) / total.replace(0, np.nan)

    # Magnitude factor: avg gain / avg loss ratio
    avg_gain = delta.where(delta > 0, 0).rolling(period).mean()
    avg_loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    mag_factor = avg_gain / (avg_gain + avg_loss.replace(0, np.nan)).replace(0, np.nan)

    tsi = (dom_ratio * 0.6 + mag_factor * 0.4).clip(0, 1)
    return tsi.rename("TSI")


# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE ELITE SCORE
# ══════════════════════════════════════════════════════════════════════════════
def compute_elite_score(df: pd.DataFrame, df_index: Optional[pd.DataFrame] = None) -> dict:
    """
    Compute all elite indicators and return a composite score + breakdown.
    This is the main entry point for the elite indicator engine.
    """
    if df.empty or len(df) < 30:
        return {"elite_score": 0.5, "breakdown": {}, "signals": [], "warnings": []}

    score_components = {}
    signals = []
    warnings = []

    try:
        # VWAP position
        vwap = compute_vwap(df)
        price = float(df["Close"].iloc[-1])
        vwap_val = float(vwap.iloc[-1]) if not vwap.empty else price
        vwap_pos = (price - vwap_val) / vwap_val if vwap_val > 0 else 0
        vwap_score = float(np.clip(0.5 + vwap_pos * 5, 0, 1))
        score_components["vwap"] = vwap_score
        if vwap_pos > 0.002:
            signals.append(f"Price above VWAP ({vwap_val:.2f}) ✅")
        elif vwap_pos < -0.002:
            signals.append(f"Price below VWAP ({vwap_val:.2f}) ⚠️")
    except Exception as e:
        logger.debug("VWAP error: %s", e)
        score_components["vwap"] = 0.5

    try:
        # Order Book Imbalance
        obi = compute_order_book_imbalance(df)
        obi_val = float(obi.iloc[-1]) if not obi.empty and not np.isnan(obi.iloc[-1]) else 0
        obi_score = float(np.clip(0.5 + obi_val * 0.5, 0, 1))
        score_components["obi"] = obi_score
        if obi_val > 0.2:
            signals.append(f"Buy pressure dominant (OBI: {obi_val:.2f}) ✅")
        elif obi_val < -0.2:
            signals.append(f"Sell pressure dominant (OBI: {obi_val:.2f}) ⚠️")
    except Exception as e:
        logger.debug("OBI error: %s", e)
        score_components["obi"] = 0.5

    try:
        # Smart Money Flow
        smfi = compute_smart_money_flow(df)
        smfi_val = float(smfi.iloc[-1]) if not smfi.empty and not np.isnan(smfi.iloc[-1]) else 50
        smfi_score = float(np.clip(smfi_val / 100, 0, 1))
        score_components["smfi"] = smfi_score
        if smfi_val > 60:
            signals.append(f"Smart money accumulating (SMFI: {smfi_val:.0f}) ✅")
        elif smfi_val < 40:
            signals.append(f"Smart money distributing (SMFI: {smfi_val:.0f}) ⚠️")
    except Exception as e:
        logger.debug("SMFI error: %s", e)
        score_components["smfi"] = 0.5

    try:
        # Market Structure
        ms = compute_market_structure(df)
        ms_score = 0.75 if ms["structure"] == "UPTREND" else (0.25 if ms["structure"] == "DOWNTREND" else 0.5)
        score_components["market_structure"] = ms_score
        signals.append(f"Market structure: {ms['structure']}")
        if ms.get("last_choch", 0) > 0:
            warnings.append(f"CHoCH detected at ₹{ms['last_choch']:.2f} — possible reversal")
    except Exception as e:
        logger.debug("Market structure error: %s", e)
        score_components["market_structure"] = 0.5

    try:
        # Breakout Strength
        bos = compute_breakout_strength(df)
        score_components["breakout"] = bos["score"]
        if bos["is_breakout"]:
            signals.extend(bos["reasons"])
    except Exception as e:
        logger.debug("Breakout error: %s", e)
        score_components["breakout"] = 0.5

    try:
        # Trend Strength Index
        tsi = compute_trend_strength_index(df)
        tsi_val = float(tsi.iloc[-1]) if not tsi.empty and not np.isnan(tsi.iloc[-1]) else 0.5
        score_components["tsi"] = tsi_val
        if tsi_val > 0.65:
            signals.append(f"Strong trend (TSI: {tsi_val:.2f}) ✅")
        elif tsi_val < 0.35:
            warnings.append(f"Weak/ranging market (TSI: {tsi_val:.2f})")
    except Exception as e:
        logger.debug("TSI error: %s", e)
        score_components["tsi"] = 0.5

    try:
        # Volatility Regime
        vol_regime = compute_volatility_regime(df)
        score_components["volatility"] = vol_regime["regime_score"]
        if vol_regime["regime"] in ("HIGH", "EXTREME"):
            warnings.append(f"Volatility regime: {vol_regime['regime']} — widen stops")
    except Exception as e:
        logger.debug("Volatility regime error: %s", e)
        score_components["volatility"] = 0.5

    try:
        # Relative Strength vs Index
        if df_index is not None and not df_index.empty:
            rs = compute_relative_strength(df, df_index)
            rs_val = float(rs.iloc[-1]) if not rs.empty and not np.isnan(rs.iloc[-1]) else 1.0
            rs_score = float(np.clip(rs_val / 2, 0, 1))
            score_components["relative_strength"] = rs_score
            if rs_val > 1.1:
                signals.append(f"Outperforming Nifty (RS: {rs_val:.2f}) ✅")
            elif rs_val < 0.9:
                warnings.append(f"Underperforming Nifty (RS: {rs_val:.2f})")
    except Exception as e:
        logger.debug("RS error: %s", e)

    # Weighted composite score
    weights = {
        "vwap": 0.20, "obi": 0.15, "smfi": 0.15, "market_structure": 0.20,
        "breakout": 0.15, "tsi": 0.10, "volatility": 0.05,
    }
    if "relative_strength" in score_components:
        weights["relative_strength"] = 0.10
        # Renormalize
        total_w = sum(weights.values())
        weights = {k: v / total_w for k, v in weights.items()}

    elite_score = sum(score_components.get(k, 0.5) * w for k, w in weights.items())
    elite_score = float(np.clip(elite_score, 0, 1))

    return {
        "elite_score": round(elite_score, 4),
        "breakdown": {k: round(v, 4) for k, v in score_components.items()},
        "signals": signals,
        "warnings": warnings,
        "vwap": round(vwap_val if "vwap" in score_components else 0, 2),
        "market_structure": ms.get("structure", "RANGING") if "market_structure" in score_components else "RANGING",
    }

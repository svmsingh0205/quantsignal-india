"""
Stock Analysis Engine — Comprehensive single-stock deep-dive.
Orchestrates all existing engines into one unified AnalysisBundle.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional
import logging

from .data_service import DataService
from .feature_engine import FeatureEngine
from .intraday_engine import IntradayEngine
from .prediction_engine import PredictionEngine
from .risk_engine import RiskEngine
from .monte_carlo import MonteCarloEngine
from .multi_analyzer import MultiAnalyzer
from .stock_metadata import StockMetadata
from ..intraday_config import SECTOR_GROUPS

logger = logging.getLogger(__name__)

SYMBOL_TO_SECTOR: dict[str, str] = {}
for _sec, _syms in SECTOR_GROUPS.items():
    for _s in _syms:
        SYMBOL_TO_SECTOR[_s.replace(".NS", "")] = _sec


@dataclass
class TimeframeSignal:
    timeframe: str
    signal: str        # "BUY" | "SELL" | "NEUTRAL"
    score: float       # 0–1
    rsi: float
    macd_bullish: bool
    above_ema: bool
    supertrend: str
    reasons: list


@dataclass
class TradeSetup:
    entry: float
    target_1: float
    target_2: float
    stop_loss: float
    atr: float
    risk_per_share: float
    reward_per_share: float
    risk_reward: float
    qty: int
    invested: float
    max_profit: float
    max_loss: float
    signal: str
    confidence: float


@dataclass
class AnalysisBundle:
    symbol: str
    yf_symbol: str
    sector: str
    capital: float
    risk_pct: float
    df_daily: pd.DataFrame
    df_intra: pd.DataFrame
    df_weekly: pd.DataFrame
    df_features: pd.DataFrame
    current_price: float
    price_52w_high: float
    price_52w_low: float
    price_change_1d: float
    price_change_1d_abs: float
    intra_score: Optional[dict]
    pred_next_day: Optional[dict]
    pred_multi: Optional[dict]
    risk_metrics: Optional[dict]
    mc_result: Optional[dict]
    ma_result: Optional[dict]
    metadata_report: dict
    trade_setup: TradeSetup
    peer_data: dict = field(default_factory=dict)


def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    prev_close = df["Close"].shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(period).mean().iloc[-1])
    return atr if not np.isnan(atr) else float(df["Close"].iloc[-1] * 0.02)


def compute_trade_setup(
    price: float,
    df_daily: pd.DataFrame,
    capital: float,
    risk_pct: float,
    ma_result: dict,
) -> TradeSetup:
    atr = max(_compute_atr(df_daily), price * 0.005)
    entry = round(price, 2)
    stop_loss = round(max(entry - 1.5 * atr, entry * 0.93), 2)
    target_1 = round(entry + 2.0 * atr, 2)
    target_2 = round(entry + 3.0 * atr, 2)
    risk_per_share = entry - stop_loss
    reward_per_share = target_1 - entry
    if risk_per_share > 0:
        risk_reward = round(reward_per_share / risk_per_share, 2)
        qty = max(0, int(capital * risk_pct / risk_per_share))
    else:
        risk_reward = 0.0
        qty = 0
    invested = round(qty * entry, 2)
    max_profit = round(qty * reward_per_share, 2)
    max_loss = round(qty * risk_per_share, 2)
    confidence = ma_result.get("combined_score", 0.5)
    signal = "BUY" if confidence >= 0.55 else ("WATCH" if confidence >= 0.45 else "AVOID")
    return TradeSetup(
        entry=entry, target_1=target_1, target_2=target_2, stop_loss=stop_loss,
        atr=round(atr, 2), risk_per_share=round(risk_per_share, 2),
        reward_per_share=round(reward_per_share, 2), risk_reward=risk_reward,
        qty=qty, invested=invested, max_profit=max_profit, max_loss=max_loss,
        signal=signal, confidence=round(confidence, 4),
    )


def build_timeframe_signals(
    symbol: str,
    df_intra: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame,
) -> list:
    """Build TimeframeSignal objects for 5m, 1d, and 1w timeframes."""
    signals = []

    # 5m Intraday signal
    if not df_intra.empty and len(df_intra) >= 30:
        try:
            sc_5m = IntradayEngine.score_stock(df_intra)
            score = float(np.clip(sc_5m.get("score", 0.5), 0.0, 1.0))
            sig = "BUY" if score >= 0.55 else ("SELL" if score < 0.35 else "NEUTRAL")
            last_intra = df_intra.iloc[-1]
            signals.append(TimeframeSignal(
                timeframe="5m Intraday",
                signal=sig,
                score=score,
                rsi=float(sc_5m.get("rsi", 50)),
                macd_bullish=float(last_intra.get("MACD_Hist", 0)) > 0 if "MACD_Hist" in df_intra.columns else False,
                above_ema=float(last_intra.get("Close", 0)) > float(last_intra.get("EMA21", 0)) if "EMA21" in df_intra.columns else False,
                supertrend=sc_5m.get("supertrend", "NEUTRAL"),
                reasons=sc_5m.get("reasons", []),
            ))
        except Exception as e:
            logger.warning("build_timeframe_signals 5m error: %s", e)

    # 1d and 1w signals
    for tf_label, df_tf in [("1d Swing", df_daily), ("1w Positional", df_weekly)]:
        if df_tf is None or df_tf.empty or len(df_tf) < 30:
            continue
        try:
            feat = FeatureEngine.compute_all_features(df_tf)
            last = feat.iloc[-1]
            rsi = float(last.get("RSI", 50)) if hasattr(last, "get") else 50.0
            macd_hist = float(last.get("MACD_Hist", 0)) if hasattr(last, "get") else 0.0
            price = float(df_tf["Close"].iloc[-1])
            ema21 = float(last.get("MA20", price)) if hasattr(last, "get") else price

            s = 0.5
            if rsi < 45:
                s += 0.15
            elif rsi > 70:
                s -= 0.10
            else:
                s += 0.05
            s += 0.15 if macd_hist > 0 else -0.05
            s += 0.10 if price > ema21 else -0.05
            s = float(np.clip(s, 0.0, 1.0))

            sig = "BUY" if s >= 0.55 else ("SELL" if s < 0.35 else "NEUTRAL")
            signals.append(TimeframeSignal(
                timeframe=tf_label,
                signal=sig,
                score=s,
                rsi=rsi,
                macd_bullish=macd_hist > 0,
                above_ema=price > ema21,
                supertrend="N/A",
                reasons=[],
            ))
        except Exception as e:
            logger.warning("build_timeframe_signals %s error: %s", tf_label, e)

    return signals


def load_analysis_bundle(symbol: str, capital: float, risk_pct: float) -> AnalysisBundle:
    import streamlit as st

    cache_key = f"sap_{symbol}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    yf_sym = f"{symbol}.NS"
    sector = SYMBOL_TO_SECTOR.get(symbol, "Other")

    # Parallel data fetch
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_daily = ex.submit(DataService.fetch_ohlcv, yf_sym, "2y", "1d")
        f_intra = ex.submit(IntradayEngine.fetch_intraday, yf_sym, "5d", "5m")
        f_weekly = ex.submit(DataService.fetch_ohlcv, yf_sym, "5y", "1wk")
        df_daily = f_daily.result()
        df_intra = f_intra.result()
        df_weekly = f_weekly.result()

    if df_daily.empty or len(df_daily) < 60:
        raise ValueError(f"Insufficient daily data for {symbol}")

    df_intra = IntradayEngine.add_indicators(df_intra) if not df_intra.empty else df_intra
    df_features = FeatureEngine.compute_all_features(df_daily)

    price = float(df_daily["Close"].iloc[-1])
    prev_price = float(df_daily["Close"].iloc[-2]) if len(df_daily) >= 2 else price
    price_change_1d = round((price / prev_price - 1) * 100, 2)
    price_change_1d_abs = round(price - prev_price, 2)
    price_52w_high = float(df_daily["High"].tail(252).max())
    price_52w_low = float(df_daily["Low"].tail(252).min())

    # Parallel engine execution — failures return None via _safe()
    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            logger.warning("Engine error %s: %s", getattr(fn, "__name__", str(fn)), e)
            return None

    with ThreadPoolExecutor(max_workers=6) as ex:
        f_intra_score = ex.submit(_safe, IntradayEngine.score_stock, df_intra) if not df_intra.empty else None
        f_pred_nd = ex.submit(_safe, PredictionEngine.predict_next_day, df_daily)
        f_pred_multi = ex.submit(_safe, PredictionEngine.predict_multi_horizon, df_daily)
        f_risk = ex.submit(_safe, RiskEngine.compute_all, df_daily["Close"])
        f_mc = ex.submit(_safe, MonteCarloEngine.simulate, df_daily["Close"], 3000, 30)
        f_ma = ex.submit(_safe, MultiAnalyzer().analyze, df_daily, sector, capital, risk_pct)

        intra_score = (f_intra_score.result() if f_intra_score else None) or {
            "score": 0.5, "rsi": 50, "reasons": [], "atr": price * 0.01,
            "vwap": price, "vol_ratio": 1, "supertrend": "NEUTRAL"
        }
        pred_next_day = f_pred_nd.result()
        pred_multi = f_pred_multi.result()
        risk_metrics = f_risk.result()
        mc_result = f_mc.result()
        ma_result = f_ma.result() or {"combined_score": 0.5, "signal": "NEUTRAL", "reasoning": []}

    vol = (risk_metrics or {}).get("volatility_annual", 0.25)
    rsi = intra_score.get("rsi", 50)
    metadata_report = StockMetadata.generate_report(
        symbol=symbol, price=price, sector=sector,
        confidence=ma_result.get("combined_score", 0.5),
        direction=(pred_next_day or {}).get("direction", "NEUTRAL"),
        predicted_return=(pred_next_day or {}).get("predicted_return", 0),
        volatility=vol, rsi=rsi,
        entry=price, target=price * 1.05, stop_loss=price * 0.97,
        mode="swing",
    )

    trade_setup = compute_trade_setup(price, df_daily, capital, risk_pct, ma_result)

    bundle = AnalysisBundle(
        symbol=symbol, yf_symbol=yf_sym, sector=sector, capital=capital, risk_pct=risk_pct,
        df_daily=df_daily, df_intra=df_intra, df_weekly=df_weekly, df_features=df_features,
        current_price=price, price_52w_high=price_52w_high, price_52w_low=price_52w_low,
        price_change_1d=price_change_1d, price_change_1d_abs=price_change_1d_abs,
        intra_score=intra_score, pred_next_day=pred_next_day, pred_multi=pred_multi,
        risk_metrics=risk_metrics, mc_result=mc_result, ma_result=ma_result,
        metadata_report=metadata_report, trade_setup=trade_setup,
    )
    st.session_state[cache_key] = bundle
    return bundle


def load_peer_data(symbol: str, sector: str) -> list[dict]:
    sector_syms = SECTOR_GROUPS.get(sector, [])
    candidates = [s.replace(".NS", "") for s in sector_syms if s.replace(".NS", "") != symbol][:4]
    all_syms = [symbol] + candidates
    peer_dfs = DataService.fetch_multiple([f"{s}.NS" for s in all_syms], period="6mo")
    peers = []
    for sym in all_syms:
        df = peer_dfs.get(f"{sym}.NS", pd.DataFrame())
        if df.empty or len(df) < 20:
            continue
        try:
            feat = FeatureEngine.compute_all_features(df)
            last = feat.iloc[-1]
            price = float(df["Close"].iloc[-1])
            returns = df["Close"].pct_change().dropna()
            peers.append({
                "symbol": sym,
                "price": round(price, 2),
                "return_1m": round((df["Close"].iloc[-1] / df["Close"].iloc[-21] - 1) * 100, 2) if len(df) >= 21 else 0,
                "return_3m": round((df["Close"].iloc[-1] / df["Close"].iloc[-63] - 1) * 100, 2) if len(df) >= 63 else 0,
                "return_6m": round((df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100, 2),
                "rsi": round(float(last.get("RSI", 50)), 1),
                "volatility": round(float(returns.std() * np.sqrt(252) * 100), 1),
                "sharpe": round(RiskEngine.sharpe_ratio(returns), 2),
                "is_target": sym == symbol,
                "df": df,
            })
        except Exception:
            continue
    return peers

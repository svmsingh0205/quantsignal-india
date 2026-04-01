"""Signal Engine — combines quant models to produce actionable trade signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .data_service import DataService
from .feature_engine import FeatureEngine
from .ml_engine import MLEngine
from ..config import (
    ALL_SYMBOLS, MIN_CONFIDENCE, MIN_RISK_REWARD,
    MAX_SIGNALS, ML_WEIGHT, ENTRY_SCORE_WEIGHT, RISK_WEIGHT,
)
import logging

logger = logging.getLogger(__name__)


class SignalEngine:
    """Generates ranked trading signals across the full stock universe."""

    def __init__(self):
        self.ml = MLEngine()

    def generate_signals(
        self,
        symbols: Optional[list[str]] = None,
        min_confidence: float = MIN_CONFIDENCE,
        min_rr: float = MIN_RISK_REWARD,
        max_signals: int = MAX_SIGNALS,
        capital: float = 1_000_000,
        risk_pct: float = 0.02,
    ) -> list[dict]:
        symbols = symbols or ALL_SYMBOLS
        raw_signals = []
        for sym in symbols:
            try:
                sig = self._evaluate_stock(sym, capital, risk_pct)
                if sig and sig["confidence"] >= min_confidence and sig["risk_reward"] >= min_rr:
                    raw_signals.append(sig)
            except Exception as e:
                logger.warning("Error evaluating %s: %s", sym, e)
        raw_signals.sort(key=lambda x: x["confidence"], reverse=True)
        return raw_signals[:max_signals]

    def _evaluate_stock(self, symbol: str, capital: float, risk_pct: float) -> Optional[dict]:
        df = DataService.fetch_ohlcv(symbol, period="1y")
        if df.empty or len(df) < 200:
            return None
        features_df = FeatureEngine.compute_all_features(df)
        ml_features = FeatureEngine.get_ml_features(df)
        if ml_features.empty or len(ml_features) < 60:
            return None
        ml_result = self.ml.get_stock_prediction(ml_features, df)
        prob_up = ml_result["probability_up"]
        entry_score = FeatureEngine.compute_entry_score(df)
        last = features_df.iloc[-1]
        vol = last.get("Volatility", 0.25)
        risk_score = float(np.clip(1 - vol / 0.5, 0, 1))
        confidence = ML_WEIGHT * prob_up + ENTRY_SCORE_WEIGHT * entry_score + RISK_WEIGHT * risk_score
        confidence = float(np.clip(confidence, 0, 1))
        current_price = float(df["Close"].iloc[-1])
        atr = self._compute_atr(df)
        support = self._find_support(df)
        resistance = self._find_resistance(df)
        entry = round(current_price, 2)
        stop_loss = max(support, entry - 2 * atr)
        stop_loss = min(stop_loss, entry * 0.97)
        stop_loss = round(stop_loss, 2)
        target = min(resistance, entry + 3 * atr)
        target = max(target, entry * 1.04)
        target = round(target, 2)
        risk = entry - stop_loss
        reward = target - entry
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        risk_amount = capital * risk_pct
        position_size = int(risk_amount / risk) if risk > 0 else 0
        position_value = round(position_size * entry, 2)
        rsi = last.get("RSI", 50)
        momentum = last.get("Momentum", 0)
        rationale = self._build_rationale(prob_up, entry_score, rsi, momentum, vol, risk_reward)
        clean_sym = symbol.replace(".NS", "")
        return {
            "symbol": clean_sym,
            "entry": entry,
            "target": target,
            "stop_loss": stop_loss,
            "confidence": round(confidence, 4),
            "risk_reward": risk_reward,
            "direction": "LONG" if prob_up > 0.5 else "SHORT",
            "position_size": position_size,
            "position_value": position_value,
            "ml_probability": round(prob_up, 4),
            "entry_score": round(entry_score, 4),
            "risk_score": round(risk_score, 4),
            "rsi": round(float(rsi), 2),
            "momentum": round(float(momentum), 4),
            "volatility": round(float(vol), 4),
            "rationale": rationale,
        }

    @staticmethod
    def _compute_atr(df: pd.DataFrame, period: int = 14) -> float:
        high, low, close = df["High"], df["Low"], df["Close"].shift(1)
        tr = pd.concat([high - low, (high - close).abs(), (low - close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().iloc[-1]
        return float(atr) if not np.isnan(atr) else float(df["Close"].iloc[-1] * 0.02)

    @staticmethod
    def _find_support(df: pd.DataFrame, window: int = 20) -> float:
        return float(df["Low"].iloc[-window:].min())

    @staticmethod
    def _find_resistance(df: pd.DataFrame, window: int = 20) -> float:
        return float(df["High"].iloc[-window:].max())

    @staticmethod
    def _build_rationale(prob_up, entry_score, rsi, momentum, vol, rr) -> str:
        parts = []
        if prob_up > 0.65:
            parts.append(f"Strong ML bullish ({prob_up:.0%})")
        elif prob_up > 0.5:
            parts.append(f"Moderate ML bullish ({prob_up:.0%})")
        else:
            parts.append(f"ML bearish ({prob_up:.0%})")
        if entry_score > 0.6:
            parts.append("favourable market timing")
        if rsi < 35:
            parts.append("RSI oversold")
        elif rsi > 70:
            parts.append("RSI overbought — caution")
        if momentum > 0.05:
            parts.append("positive momentum")
        if vol < 0.2:
            parts.append("low volatility regime")
        parts.append(f"R:R {rr:.1f}x")
        return "; ".join(parts)

"""Signal Engine — combines quant models to produce actionable trade signals."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .data_service import DataService
from .feature_engine import FeatureEngine
from .ml_engine import MLEngine
from .multi_analyzer import MultiAnalyzer
from .elite_indicators import compute_elite_score
from .universe import MASTER_UNIVERSE, SYMBOL_TO_SECTOR as _UNIVERSE_SECTOR
from ..config import (
    ALL_SYMBOLS, MIN_CONFIDENCE, MIN_RISK_REWARD,
    MAX_SIGNALS, ML_WEIGHT, ENTRY_SCORE_WEIGHT, RISK_WEIGHT, SECTOR_MAP,
)
import logging

logger = logging.getLogger(__name__)

# Build reverse symbol → sector map (universe takes priority, config as fallback)
_SYMBOL_SECTOR: dict[str, str] = {}
for _sec, _syms in SECTOR_MAP.items():
    for _s in _syms:
        _SYMBOL_SECTOR[_s] = _sec
# Overlay universe sector map
_SYMBOL_SECTOR.update(_UNIVERSE_SECTOR)

# Extended symbol list: use full universe if available
_FULL_UNIVERSE = MASTER_UNIVERSE if MASTER_UNIVERSE else ALL_SYMBOLS


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
        enabled_analyzers: Optional[list[str]] = None,
    ) -> list[dict]:
        symbols = symbols or _FULL_UNIVERSE
        raw_signals = []
        for sym in symbols:
            try:
                sig = self._evaluate_stock(sym, capital, risk_pct, enabled_analyzers)
                if sig and sig["confidence"] >= min_confidence and sig["risk_reward"] >= min_rr:
                    raw_signals.append(sig)
            except Exception as e:
                logger.warning("Error evaluating %s: %s", sym, e)
        raw_signals.sort(key=lambda x: x["confidence"], reverse=True)
        return raw_signals[:max_signals]

    def _evaluate_stock(
        self,
        symbol: str,
        capital: float,
        risk_pct: float,
        enabled_analyzers: Optional[list[str]] = None,
    ) -> Optional[dict]:
        df = DataService.fetch_ohlcv(symbol, period="1y")
        if df.empty or len(df) < 200:
            return None
        features_df = FeatureEngine.compute_all_features(df)
        ml_features = FeatureEngine.get_ml_features(df)
        if ml_features.empty or len(ml_features) < 60:
            return None

        # ── ML score ──────────────────────────────────────────────────────────
        ml_result = self.ml.get_stock_prediction(ml_features, df)
        prob_up = ml_result["probability_up"]

        # ── Entry timing score ────────────────────────────────────────────────
        entry_score = FeatureEngine.compute_entry_score(df)

        # ── Risk score ────────────────────────────────────────────────────────
        last = features_df.iloc[-1]
        vol = last.get("Volatility", 0.25)
        risk_score = float(np.clip(1 - vol / 0.5, 0, 1))

        # ── Elite indicators score ────────────────────────────────────────────
        try:
            elite = compute_elite_score(df)
            elite_score = elite.get("elite_score", 0.5)
        except Exception:
            elite_score = 0.5

        # ── Multi-analyzer score ──────────────────────────────────────────────
        sector = _SYMBOL_SECTOR.get(symbol, _SYMBOL_SECTOR.get(symbol.replace(".NS", ""), ""))
        ma = MultiAnalyzer(enabled_analyzers)
        ma_result = ma.analyze(df, sector=sector, capital=capital, risk_pct=risk_pct)
        ma_score = ma_result.get("combined_score", 0.5)

        # ── Combined confidence (ML 30% + entry 15% + risk 10% + elite 20% + multi 25%) ──
        confidence = (
            0.30 * prob_up
            + 0.15 * entry_score
            + 0.10 * risk_score
            + 0.20 * elite_score
            + 0.25 * ma_score
        )
        confidence = float(np.clip(confidence, 0, 1))

        # ── Price levels ──────────────────────────────────────────────────────
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

        # ── Capital-based position sizing ─────────────────────────────────────
        risk_amount = capital * risk_pct
        position_size = int(risk_amount / risk) if risk > 0 else 0
        position_value = round(position_size * entry, 2)
        potential_profit = round(position_size * reward, 2)
        potential_loss = round(position_size * risk, 2)

        rsi = last.get("RSI", 50)
        momentum = last.get("Momentum", 0)

        # ── Reasoning: merge legacy + multi-analyzer ──────────────────────────
        legacy_rationale = self._build_rationale(prob_up, entry_score, rsi, momentum, vol, risk_reward)
        ma_reasons = ma_result.get("reasoning", [])
        full_rationale = legacy_rationale
        if ma_reasons:
            full_rationale += "; " + "; ".join(ma_reasons[:3])

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
            "potential_profit": potential_profit,
            "potential_loss": potential_loss,
            "risk_amount": round(risk_amount, 2),
            "ml_probability": round(prob_up, 4),
            "entry_score": round(entry_score, 4),
            "risk_score": round(risk_score, 4),
            "elite_score": round(elite_score, 4),
            "multi_analyzer_score": round(ma_score, 4),
            "analyzer_breakdown": ma_result.get("analyzers", {}),
            "rsi": round(float(rsi), 2),
            "momentum": round(float(momentum), 4),
            "volatility": round(float(vol), 4),
            "sector": sector,
            "rationale": full_rationale,
            "reasoning": ma_reasons,
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

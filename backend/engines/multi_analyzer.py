"""
Multi-Analyzer Engine — QuantSignal India
Combines Indian technical, global macro, momentum, volume, and sentiment
analyzers into a unified, capital-aware signal with per-analyzer scores.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── Global index proxies (yfinance tickers) ──────────────────────────────────
GLOBAL_INDICES = {
    "S&P 500":    "^GSPC",
    "NASDAQ":     "^IXIC",
    "Dow Jones":  "^DJI",
    "VIX":        "^VIX",
    "India VIX":  "^INDIAVIX",
    "Nikkei 225": "^N225",
    "Hang Seng":  "^HSI",
    "FTSE 100":   "^FTSE",
    "DAX":        "^GDAXI",
    "Crude Oil":  "CL=F",
    "Gold":       "GC=F",
    "USD/INR":    "USDINR=X",
    "DXY":        "DX-Y.NYB",
}

# ── Geopolitical / macro theme scores (static, updated periodically) ─────────
GEOPOLITICAL_THEME_SCORES: dict[str, float] = {
    "defence":   0.85,   # India-US 10yr deal, strong order book
    "psu_banks": 0.70,   # PSU bank outperformance, credit growth
    "infra":     0.80,   # ₹11L cr budget, infra supercycle
    "energy":    0.72,   # Energy security + renewables push
    "it":        0.68,   # India-US Mission 500 trade deal
    "pharma":    0.65,   # China+1 API sourcing, US generics
    "metals":    0.60,   # Infra demand, China recovery
    "auto":      0.63,   # PLI scheme, EV transition
    "fmcg":      0.55,   # Rural consumption recovery
    "finance":   0.67,   # Credit growth, rate cycle
    "chemicals": 0.62,   # China+1 supply chain shift
    "realty":    0.58,   # Infra supercycle, urbanisation
    "telecom":   0.60,   # 5G rollout, data growth
}


@dataclass
class AnalyzerResult:
    name: str
    score: float          # 0-1
    signal: str           # BULLISH / NEUTRAL / BEARISH
    weight: float         # contribution weight
    details: dict = field(default_factory=dict)

    @property
    def weighted_score(self) -> float:
        return self.score * self.weight


class TechnicalAnalyzer:
    """RSI, MACD, Moving Averages, Bollinger Bands, Supertrend."""

    NAME = "Technical"
    WEIGHT = 0.30

    @staticmethod
    def analyze(df: pd.DataFrame) -> AnalyzerResult:
        score = 0.5
        details: dict = {}
        try:
            c = df["Close"]
            # RSI
            delta = c.diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rsi = float((100 - (100 / (1 + gain / loss.replace(0, np.nan)))).iloc[-1])
            details["rsi"] = round(rsi, 1)

            # MACD
            ema12 = c.ewm(span=12).mean()
            ema26 = c.ewm(span=26).mean()
            macd = ema12 - ema26
            macd_sig = macd.ewm(span=9).mean()
            macd_hist = float((macd - macd_sig).iloc[-1])
            details["macd_hist"] = round(macd_hist, 4)

            # Moving averages
            price = float(c.iloc[-1])
            ma20 = float(c.rolling(20).mean().iloc[-1]) if len(c) >= 20 else price
            ma50 = float(c.rolling(50).mean().iloc[-1]) if len(c) >= 50 else price
            ma200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else price
            details["above_ma20"] = price > ma20
            details["above_ma50"] = price > ma50
            details["above_ma200"] = price > ma200

            # Bollinger position
            std20 = float(c.rolling(20).std().iloc[-1])
            bb_pos = (price - (ma20 - 2 * std20)) / (4 * std20) if std20 > 0 else 0.5
            details["bb_position"] = round(float(np.clip(bb_pos, 0, 1)), 3)

            # Score assembly
            s = 0.0
            # RSI: oversold = bullish, overbought = bearish
            if rsi < 30:
                s += 0.20
            elif rsi < 45:
                s += 0.15
            elif rsi < 60:
                s += 0.10
            elif rsi < 75:
                s += 0.05
            else:
                s -= 0.05

            # MACD histogram positive = bullish
            s += 0.15 if macd_hist > 0 else -0.05

            # MA alignment
            s += 0.10 if price > ma20 else -0.05
            s += 0.10 if price > ma50 else -0.05
            s += 0.10 if price > ma200 else -0.05

            # Bollinger: near lower band = oversold bounce potential
            if bb_pos < 0.2:
                s += 0.10
            elif bb_pos > 0.8:
                s -= 0.05

            score = float(np.clip(0.5 + s, 0.0, 1.0))
        except Exception as e:
            logger.debug("TechnicalAnalyzer error: %s", e)

        signal = "BULLISH" if score > 0.6 else ("BEARISH" if score < 0.4 else "NEUTRAL")
        return AnalyzerResult(TechnicalAnalyzer.NAME, score, signal,
                              TechnicalAnalyzer.WEIGHT, details)


class MomentumAnalyzer:
    """Price momentum, rate-of-change, trend strength."""

    NAME = "Momentum"
    WEIGHT = 0.20

    @staticmethod
    def analyze(df: pd.DataFrame) -> AnalyzerResult:
        score = 0.5
        details: dict = {}
        try:
            c = df["Close"]
            price = float(c.iloc[-1])

            # Multi-period returns
            for n in [5, 10, 20, 60]:
                if len(c) > n:
                    ret = float((c.iloc[-1] / c.iloc[-n]) - 1)
                    details[f"ret_{n}d"] = round(ret * 100, 2)

            # 200-day momentum
            if len(c) >= 200:
                ma200 = float(c.rolling(200).mean().iloc[-1])
                mom = (price / ma200) - 1
                details["momentum_200d"] = round(mom * 100, 2)
            else:
                mom = 0.0

            # ADX-like trend strength (simplified)
            if len(c) >= 14:
                high = df["High"]
                low = df["Low"]
                tr = pd.concat([
                    high - low,
                    (high - c.shift(1)).abs(),
                    (low - c.shift(1)).abs(),
                ], axis=1).max(axis=1)
                atr14 = float(tr.rolling(14).mean().iloc[-1])
                details["atr14"] = round(atr14, 2)

            # Score
            s = 0.0
            ret_20 = details.get("ret_20d", 0) / 100
            ret_60 = details.get("ret_60d", 0) / 100
            s += float(np.clip(ret_20 * 2, -0.2, 0.2))
            s += float(np.clip(ret_60 * 1, -0.15, 0.15))
            s += float(np.clip(mom * 1.5, -0.15, 0.15))

            score = float(np.clip(0.5 + s, 0.0, 1.0))
        except Exception as e:
            logger.debug("MomentumAnalyzer error: %s", e)

        signal = "BULLISH" if score > 0.6 else ("BEARISH" if score < 0.4 else "NEUTRAL")
        return AnalyzerResult(MomentumAnalyzer.NAME, score, signal,
                              MomentumAnalyzer.WEIGHT, details)


class VolumeAnalyzer:
    """Volume analysis — accumulation/distribution, OBV trend."""

    NAME = "Volume"
    WEIGHT = 0.15

    @staticmethod
    def analyze(df: pd.DataFrame) -> AnalyzerResult:
        score = 0.5
        details: dict = {}
        try:
            c = df["Close"]
            v = df["Volume"]

            vol_ma20 = v.rolling(20).mean()
            vol_ratio = float((v / vol_ma20.replace(0, np.nan)).iloc[-1])
            details["vol_ratio"] = round(vol_ratio, 2)

            # OBV trend (simplified)
            obv = (np.sign(c.diff()) * v).cumsum()
            obv_ma10 = obv.rolling(10).mean()
            obv_trend = float(obv.iloc[-1]) > float(obv_ma10.iloc[-1])
            details["obv_rising"] = obv_trend

            # Price-volume divergence
            price_up = float(c.iloc[-1]) > float(c.iloc[-5]) if len(c) >= 5 else True
            vol_up = float(v.iloc[-1]) > float(vol_ma20.iloc[-1])
            details["price_vol_confirm"] = price_up == vol_up

            # Score
            s = 0.0
            if vol_ratio > 2.0:
                s += 0.20
            elif vol_ratio > 1.5:
                s += 0.12
            elif vol_ratio > 1.2:
                s += 0.06
            elif vol_ratio < 0.5:
                s -= 0.10

            if obv_trend:
                s += 0.15
            if details["price_vol_confirm"]:
                s += 0.10

            score = float(np.clip(0.5 + s, 0.0, 1.0))
        except Exception as e:
            logger.debug("VolumeAnalyzer error: %s", e)

        signal = "BULLISH" if score > 0.6 else ("BEARISH" if score < 0.4 else "NEUTRAL")
        return AnalyzerResult(VolumeAnalyzer.NAME, score, signal,
                              VolumeAnalyzer.WEIGHT, details)


class GlobalMacroAnalyzer:
    """Global indices, USD/INR, crude oil, gold impact on Indian markets."""

    NAME = "Global Macro"
    WEIGHT = 0.20

    @staticmethod
    def analyze(sector: str = "") -> AnalyzerResult:
        """Uses cached global data or falls back to theme scores."""
        score = 0.5
        details: dict = {}
        try:
            from .data_service import DataService

            # Fetch key global proxies
            sp500 = DataService.fetch_ohlcv("^GSPC", period="1mo")
            vix = DataService.fetch_ohlcv("^VIX", period="5d")
            usdinr = DataService.fetch_ohlcv("USDINR=X", period="5d")
            crude = DataService.fetch_ohlcv("CL=F", period="5d")
            gold = DataService.fetch_ohlcv("GC=F", period="5d")

            s = 0.0

            # S&P 500 trend (positive = risk-on = good for India)
            if not sp500.empty and len(sp500) >= 10:
                sp_ret = float((sp500["Close"].iloc[-1] / sp500["Close"].iloc[-10]) - 1)
                details["sp500_10d"] = round(sp_ret * 100, 2)
                s += float(np.clip(sp_ret * 3, -0.15, 0.15))

            # VIX: high VIX = fear = bearish
            if not vix.empty:
                vix_val = float(vix["Close"].iloc[-1])
                details["vix"] = round(vix_val, 1)
                if vix_val < 15:
                    s += 0.10
                elif vix_val < 20:
                    s += 0.05
                elif vix_val > 30:
                    s -= 0.15
                elif vix_val > 25:
                    s -= 0.08

            # USD/INR: rupee weakening = bearish for importers
            if not usdinr.empty and len(usdinr) >= 5:
                inr_chg = float((usdinr["Close"].iloc[-1] / usdinr["Close"].iloc[-5]) - 1)
                details["usdinr_5d_chg"] = round(inr_chg * 100, 2)
                # Rupee depreciation (positive inr_chg) = bearish for India
                s -= float(np.clip(inr_chg * 5, -0.10, 0.10))

            # Crude oil: high crude = bad for India (net importer)
            if not crude.empty and len(crude) >= 5:
                crude_chg = float((crude["Close"].iloc[-1] / crude["Close"].iloc[-5]) - 1)
                details["crude_5d_chg"] = round(crude_chg * 100, 2)
                s -= float(np.clip(crude_chg * 2, -0.08, 0.08))

            # Gold: rising gold = risk-off = mixed for India
            if not gold.empty and len(gold) >= 5:
                gold_chg = float((gold["Close"].iloc[-1] / gold["Close"].iloc[-5]) - 1)
                details["gold_5d_chg"] = round(gold_chg * 100, 2)

            score = float(np.clip(0.5 + s, 0.0, 1.0))

        except Exception as e:
            logger.debug("GlobalMacroAnalyzer error: %s", e)
            # Fallback to static theme score
            score = GEOPOLITICAL_THEME_SCORES.get(sector, 0.55)

        signal = "BULLISH" if score > 0.6 else ("BEARISH" if score < 0.4 else "NEUTRAL")
        return AnalyzerResult(GlobalMacroAnalyzer.NAME, score, signal,
                              GlobalMacroAnalyzer.WEIGHT, details)


class GeopoliticalAnalyzer:
    """Sector-specific geopolitical and macro theme scoring."""

    NAME = "Geopolitical"
    WEIGHT = 0.15

    @staticmethod
    def analyze(sector: str = "") -> AnalyzerResult:
        score = GEOPOLITICAL_THEME_SCORES.get(sector, 0.55)
        details = {
            "sector": sector,
            "theme_score": score,
            "themes": GeopoliticalAnalyzer._get_themes(sector),
        }
        signal = "BULLISH" if score > 0.6 else ("BEARISH" if score < 0.4 else "NEUTRAL")
        return AnalyzerResult(GeopoliticalAnalyzer.NAME, score, signal,
                              GeopoliticalAnalyzer.WEIGHT, details)

    @staticmethod
    def _get_themes(sector: str) -> list[str]:
        themes = {
            "defence":   ["India-US 10yr defence deal", "Make in India", "HAL/BEL order book"],
            "psu_banks": ["PSU bank outperformance", "Credit growth cycle", "NPA resolution"],
            "infra":     ["₹11L cr budget allocation", "Infra supercycle", "RVNL/IRFC pipeline"],
            "energy":    ["Energy security push", "Renewables PLI", "Green hydrogen"],
            "it":        ["India-US Mission 500", "AI/cloud demand", "Rupee tailwind"],
            "pharma":    ["China+1 API sourcing", "US generics demand", "CDMO growth"],
            "metals":    ["Infra demand", "China recovery", "Steel cycle"],
            "auto":      ["EV PLI scheme", "Rural demand", "Export growth"],
            "fmcg":      ["Rural consumption", "Premiumisation", "Volume recovery"],
            "finance":   ["Credit growth", "Rate cycle turn", "Fintech expansion"],
            "chemicals": ["China+1 shift", "Specialty chemicals", "Fluorochemicals"],
            "realty":    ["Urbanisation", "Infra supercycle", "Affordable housing"],
            "telecom":   ["5G rollout", "Data growth", "ARPU improvement"],
        }
        return themes.get(sector, ["India growth story", "Domestic consumption"])


# ── Master multi-analyzer orchestrator ───────────────────────────────────────

class MultiAnalyzer:
    """
    Orchestrates all analyzers and produces a unified capital-aware signal.

    Analyzer weights (must sum to 1.0):
      Technical    0.30
      Momentum     0.20
      Volume       0.15
      Global Macro 0.20
      Geopolitical 0.15
    """

    ALL_ANALYZERS = ["Technical", "Momentum", "Volume", "Global Macro", "Geopolitical"]

    def __init__(self, enabled_analyzers: Optional[list[str]] = None):
        self.enabled = enabled_analyzers or self.ALL_ANALYZERS

    def analyze(
        self,
        df: pd.DataFrame,
        sector: str = "",
        capital: float = 1_000_000,
        risk_pct: float = 0.02,
    ) -> dict:
        """
        Run all enabled analyzers and return combined result.
        Returns dict with per-analyzer scores + combined confidence.
        """
        results: list[AnalyzerResult] = []

        if "Technical" in self.enabled:
            results.append(TechnicalAnalyzer.analyze(df))
        if "Momentum" in self.enabled:
            results.append(MomentumAnalyzer.analyze(df))
        if "Volume" in self.enabled:
            results.append(VolumeAnalyzer.analyze(df))
        if "Global Macro" in self.enabled:
            results.append(GlobalMacroAnalyzer.analyze(sector))
        if "Geopolitical" in self.enabled:
            results.append(GeopoliticalAnalyzer.analyze(sector))

        if not results:
            return {"error": "No analyzers enabled"}

        # Normalize weights among enabled analyzers
        total_weight = sum(r.weight for r in results)
        combined_score = sum(r.weighted_score for r in results) / total_weight if total_weight > 0 else 0.5
        combined_score = float(np.clip(combined_score, 0.0, 1.0))

        # Build per-analyzer breakdown
        breakdown = {
            r.name: {
                "score": round(r.score, 4),
                "signal": r.signal,
                "weight": round(r.weight, 2),
                "contribution": round(r.weighted_score / total_weight, 4),
                "details": r.details,
            }
            for r in results
        }

        # Capital-aware position sizing
        risk_amount = capital * risk_pct
        signal = "STRONG BUY" if combined_score >= 0.72 else (
            "BUY" if combined_score >= 0.60 else (
                "NEUTRAL" if combined_score >= 0.45 else (
                    "SELL" if combined_score >= 0.35 else "STRONG SELL"
                )
            )
        )

        # Build reasoning
        reasoning = MultiAnalyzer._build_reasoning(results, combined_score)

        return {
            "combined_score": round(combined_score, 4),
            "signal": signal,
            "reasoning": reasoning,
            "risk_amount": round(risk_amount, 2),
            "analyzers": breakdown,
            "enabled_count": len(results),
        }

    @staticmethod
    def _build_reasoning(results: list[AnalyzerResult], score: float) -> list[str]:
        reasons = []
        for r in results:
            if r.signal == "BULLISH":
                if r.name == "Technical":
                    d = r.details
                    if d.get("rsi", 50) < 45:
                        reasons.append(f"RSI oversold at {d.get('rsi', 0):.0f} — bounce potential")
                    if d.get("macd_hist", 0) > 0:
                        reasons.append("MACD histogram positive — bullish momentum")
                    if d.get("above_ma200"):
                        reasons.append("Price above 200-day MA — long-term uptrend intact")
                elif r.name == "Momentum":
                    d = r.details
                    ret20 = d.get("ret_20d", 0)
                    if ret20 > 3:
                        reasons.append(f"Strong 20-day return of {ret20:.1f}%")
                elif r.name == "Volume":
                    d = r.details
                    if d.get("vol_ratio", 1) > 1.5:
                        reasons.append(f"Volume spike {d.get('vol_ratio', 1):.1f}x avg — institutional interest")
                    if d.get("obv_rising"):
                        reasons.append("OBV rising — accumulation phase")
                elif r.name == "Global Macro":
                    d = r.details
                    if d.get("vix", 20) < 18:
                        reasons.append(f"Low VIX ({d.get('vix', 0):.0f}) — risk-on environment")
                    if d.get("sp500_10d", 0) > 1:
                        reasons.append(f"S&P 500 up {d.get('sp500_10d', 0):.1f}% — global risk appetite")
                elif r.name == "Geopolitical":
                    themes = r.details.get("themes", [])
                    if themes:
                        reasons.append(f"Geopolitical tailwind: {themes[0]}")
            elif r.signal == "BEARISH":
                if r.name == "Technical":
                    d = r.details
                    if d.get("rsi", 50) > 70:
                        reasons.append(f"RSI overbought at {d.get('rsi', 0):.0f} — caution")
                elif r.name == "Global Macro":
                    d = r.details
                    if d.get("vix", 20) > 25:
                        reasons.append(f"Elevated VIX ({d.get('vix', 0):.0f}) — risk-off")
                    if d.get("crude_5d_chg", 0) > 3:
                        reasons.append(f"Crude oil up {d.get('crude_5d_chg', 0):.1f}% — inflation risk")

        if not reasons:
            reasons.append(f"Combined analyzer score: {score:.0%}")
        return reasons

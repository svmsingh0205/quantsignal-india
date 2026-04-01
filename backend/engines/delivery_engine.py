"""
Delivery Engine — Swing/Positional trading with multi-horizon predictions.
Supports 10, 20, 30, 60-day and custom holding periods.
Incorporates seasonal trends, macro factors, sector rotation, and geopolitical themes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
import logging

from .data_service import DataService
from .prediction_engine import PredictionEngine
from .feature_engine import FeatureEngine
from .risk_engine import RiskEngine
from .stock_metadata import StockMetadata, GLOBAL_FACTORS
from ..intraday_config import INTRADAY_STOCKS, SECTOR_GROUPS

logger = logging.getLogger(__name__)

# ── Sector → clean label map ──────────────────────────────────────────────────
SYMBOL_TO_SECTOR: dict[str, str] = {}
for _sec, _syms in SECTOR_GROUPS.items():
    for _s in _syms:
        SYMBOL_TO_SECTOR[_s.replace(".NS", "")] = _sec

# ── Seasonal bias by month (1=Jan … 12=Dec) ──────────────────────────────────
# Based on historical NSE patterns
SEASONAL_BIAS: dict[int, dict] = {
    1:  {"bias": 0.03,  "note": "Jan effect — FII inflows, budget expectations"},
    2:  {"bias": 0.02,  "note": "Pre-budget rally, Q3 results season"},
    3:  {"bias": -0.01, "note": "FY-end selling, tax-loss harvesting"},
    4:  {"bias": 0.04,  "note": "New FY optimism, Q4 results, IPO season"},
    5:  {"bias": 0.01,  "note": "Sell in May caution, but India resilient"},
    6:  {"bias": -0.02, "note": "Monsoon uncertainty, global risk-off"},
    7:  {"bias": 0.02,  "note": "Monsoon progress, Q1 results"},
    8:  {"bias": 0.01,  "note": "Mid-year consolidation"},
    9:  {"bias": -0.02, "note": "Global sell-off risk, FII outflows"},
    10: {"bias": 0.03,  "note": "Festive season rally, Diwali effect"},
    11: {"bias": 0.02,  "note": "Post-Diwali momentum, Q2 results"},
    12: {"bias": 0.01,  "note": "Year-end window dressing"},
}

# ── Holding period → horizon days map ────────────────────────────────────────
HOLDING_PERIODS = {
    "10 Days":  10,
    "20 Days":  20,
    "30 Days":  30,
    "60 Days":  60,
    "Custom":   None,
}


class DeliveryEngine:
    """
    Generates delivery/swing/positional trade signals for a given holding period.
    Auto-selects top stocks with sector diversification.
    """

    def __init__(self, capital: float = 100_000, risk_pct: float = 0.02):
        self.capital = capital
        self.risk_pct = risk_pct

    # ── Public API ────────────────────────────────────────────────────────────

    def scan(
        self,
        holding_days: int = 30,
        top_n: int = 10,
        sector_filter: Optional[list[str]] = None,
        symbols: Optional[list[str]] = None,
        max_workers: int = 12,
    ) -> list[dict]:
        """
        Scan universe and return top_n delivery trade candidates.
        Ensures sector diversification (max 2 per sector).
        """
        universe = self._build_universe(sector_filter, symbols)
        results = []

        def _eval(sym: str) -> Optional[dict]:
            try:
                return self._evaluate(sym, holding_days)
            except Exception as e:
                logger.debug("DeliveryEngine._evaluate %s: %s", sym, e)
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_eval, sym): sym for sym in universe}
            for fut in as_completed(futs):
                res = fut.result()
                if res:
                    results.append(res)

        # Sort by composite score
        results.sort(key=lambda x: x["composite_score"], reverse=True)

        # Sector diversification: max 2 per sector
        diversified, sector_count = [], {}
        for r in results:
            sec = r["sector"]
            if sector_count.get(sec, 0) < 2:
                diversified.append(r)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(diversified) >= top_n:
                break

        return diversified

    # ── Internal ──────────────────────────────────────────────────────────────

    def _evaluate(self, symbol: str, holding_days: int) -> Optional[dict]:
        df = DataService.fetch_ohlcv(symbol, period="2y")
        if df.empty or len(df) < 120:
            return None

        clean = symbol.replace(".NS", "")
        sector = SYMBOL_TO_SECTOR.get(clean, "Other")
        price = float(df["Close"].iloc[-1])

        # ── ML prediction for the holding horizon ────────────────────────────
        pred = PredictionEngine._train_predict(df, horizon=holding_days)
        if not pred or pred.get("direction") == "DOWN":
            return None

        # ── Technical features ────────────────────────────────────────────────
        features = FeatureEngine.compute_all_features(df)
        last = features.iloc[-1]
        rsi = float(last.get("RSI", 50))
        momentum = float(last.get("Momentum", 0))
        vol = float(last.get("Volatility", 0.25))
        bb_pos = float(last.get("BB_Position", 0.5))
        macd_hist = float(last.get("MACD_Hist", 0))
        vol_ratio = float(last.get("Volume_Ratio", 1))

        # ── Seasonal adjustment ───────────────────────────────────────────────
        month = pd.Timestamp.now().month
        seasonal = SEASONAL_BIAS.get(month, {"bias": 0, "note": ""})
        seasonal_adj = seasonal["bias"]

        # ── Geopolitical / sector theme score ─────────────────────────────────
        geo_score = self._geo_score(sector)

        # ── Risk metrics ──────────────────────────────────────────────────────
        risk_metrics = RiskEngine.compute_all(df["Close"])
        sharpe = risk_metrics.get("sharpe_ratio", 0)
        max_dd = abs(risk_metrics.get("max_drawdown", {}).get("max_drawdown", 0.2))

        # ── Composite score ───────────────────────────────────────────────────
        ml_score = pred["confidence"]
        tech_score = self._tech_score(rsi, macd_hist, momentum, bb_pos, vol_ratio)
        risk_score = float(np.clip(1 - vol / 0.5, 0, 1))
        composite = (
            0.35 * ml_score
            + 0.25 * tech_score
            + 0.15 * geo_score
            + 0.15 * risk_score
            + 0.10 * float(np.clip((sharpe + 1) / 3, 0, 1))
        ) + seasonal_adj * 0.5
        composite = float(np.clip(composite, 0, 1))

        if composite < 0.45:
            return None

        # ── Price levels ──────────────────────────────────────────────────────
        atr = self._atr(df)
        entry = round(price, 2)
        stop_loss = round(max(entry - 2.5 * atr, entry * 0.93), 2)
        target = round(min(entry + 4 * atr, entry * (1 + pred["predicted_return"] / 100)), 2)
        target = max(target, round(entry * 1.04, 2))
        risk_per_share = entry - stop_loss
        reward_per_share = target - entry
        rr = round(reward_per_share / risk_per_share, 2) if risk_per_share > 0 else 0

        # ── Position sizing ───────────────────────────────────────────────────
        risk_amount = self.capital * self.risk_pct
        qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        position_value = round(qty * entry, 2)
        potential_profit = round(qty * reward_per_share, 2)
        potential_loss = round(qty * risk_per_share, 2)

        # ── Reasoning ─────────────────────────────────────────────────────────
        reasons = self._build_reasons(
            pred, rsi, macd_hist, momentum, vol_ratio, geo_score,
            seasonal, sharpe, max_dd, holding_days
        )

        return {
            "symbol": clean,
            "sector": sector,
            "entry": entry,
            "target": target,
            "stop_loss": stop_loss,
            "risk_reward": rr,
            "composite_score": round(composite, 4),
            "ml_confidence": round(ml_score, 4),
            "tech_score": round(tech_score, 4),
            "geo_score": round(geo_score, 4),
            "predicted_return": pred["predicted_return"],
            "predicted_price": pred["predicted_price"],
            "price_low": pred["price_low"],
            "price_high": pred["price_high"],
            "direction": pred["direction"],
            "holding_days": holding_days,
            "rsi": round(rsi, 1),
            "momentum": round(momentum * 100, 2),
            "volatility": round(vol * 100, 1),
            "vol_ratio": round(vol_ratio, 2),
            "sharpe": round(sharpe, 2),
            "max_drawdown": round(max_dd * 100, 1),
            "seasonal_note": seasonal["note"],
            "seasonal_bias": round(seasonal_adj * 100, 1),
            "qty": qty,
            "position_value": position_value,
            "potential_profit": potential_profit,
            "potential_loss": potential_loss,
            "risk_amount": round(risk_amount, 2),
            "reasons": reasons,
            "signal": "STRONG BUY" if composite >= 0.72 else ("BUY" if composite >= 0.55 else "WATCH"),
            "global_factors": StockMetadata.get_global_factors(sector),
        }

    @staticmethod
    def _tech_score(rsi, macd_hist, momentum, bb_pos, vol_ratio) -> float:
        s = 0.5
        if rsi < 35:
            s += 0.15
        elif rsi < 50:
            s += 0.08
        elif rsi > 70:
            s -= 0.10
        s += 0.10 if macd_hist > 0 else -0.05
        s += float(np.clip(momentum * 2, -0.10, 0.10))
        if bb_pos < 0.25:
            s += 0.08
        elif bb_pos > 0.80:
            s -= 0.05
        if vol_ratio > 1.5:
            s += 0.07
        return float(np.clip(s, 0, 1))

    @staticmethod
    def _geo_score(sector: str) -> float:
        from ..engines.multi_analyzer import GEOPOLITICAL_THEME_SCORES
        # Map display sector label to config key
        mapping = {
            "🛡️ Defence": "defence", "🏦 PSU Banks": "psu_banks",
            "🏗️ Infra/Rail": "infra", "⚡ Energy": "energy",
            "💻 IT/Tech": "it", "💊 Pharma": "pharma",
            "⚙️ Metals": "metals", "🚗 Auto/EV": "auto",
            "🛒 FMCG": "fmcg", "💰 Finance": "finance",
            "🧪 Chemicals": "chemicals", "🏠 Realty/Cement": "realty",
            "📡 Telecom": "telecom",
        }
        key = mapping.get(sector, sector.lower().split()[0] if sector else "")
        return GEOPOLITICAL_THEME_SCORES.get(key, 0.55)

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> float:
        high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = float(tr.rolling(period).mean().iloc[-1])
        return atr if not np.isnan(atr) else float(df["Close"].iloc[-1] * 0.02)

    @staticmethod
    def _build_reasons(pred, rsi, macd_hist, momentum, vol_ratio, geo_score,
                       seasonal, sharpe, max_dd, holding_days) -> list[str]:
        r = []
        ret = pred["predicted_return"]
        r.append(f"ML predicts +{ret:.1f}% over {holding_days} days (conf {pred['confidence']:.0%})")
        if rsi < 40:
            r.append(f"RSI oversold at {rsi:.0f} — good entry zone")
        elif rsi < 55:
            r.append(f"RSI neutral at {rsi:.0f} — room to run")
        if macd_hist > 0:
            r.append("MACD bullish — momentum building")
        if momentum > 0.03:
            r.append(f"Strong 200-day momentum +{momentum*100:.1f}%")
        if vol_ratio > 1.3:
            r.append(f"Volume {vol_ratio:.1f}x avg — institutional accumulation")
        if geo_score >= 0.70:
            r.append(f"Strong geopolitical tailwind (score {geo_score:.0%})")
        if seasonal["bias"] > 0:
            r.append(f"Seasonal tailwind: {seasonal['note']}")
        if sharpe > 1.0:
            r.append(f"Sharpe {sharpe:.1f} — good risk-adjusted returns")
        if max_dd < 0.15:
            r.append(f"Low max drawdown {max_dd*100:.0f}% — stable stock")
        return r

    @staticmethod
    def _build_universe(sector_filter, symbols) -> list[str]:
        if symbols:
            return [f"{s}.NS" if not s.endswith(".NS") else s for s in symbols]
        if sector_filter:
            out = []
            for s in sector_filter:
                out.extend(SECTOR_GROUPS.get(s, []))
            return list(dict.fromkeys(out))
        return INTRADAY_STOCKS[:]

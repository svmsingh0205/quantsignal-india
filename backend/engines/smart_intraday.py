"""
Smart Intraday Engine — Auto-selects top 10 stocks for a given trading date.
No manual filters required. Uses combined signals: technical + volume +
momentum + macro + geopolitical themes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Optional
import logging

from .intraday_engine import IntradayEngine
from .data_service import DataService
from .prediction_engine import PredictionEngine
from .feature_engine import FeatureEngine
from .stock_metadata import StockMetadata, PENNY_MAX_PRICE
from ..intraday_config import INTRADAY_STOCKS, SECTOR_GROUPS, MIN_CONFIDENCE

logger = logging.getLogger(__name__)

SYMBOL_TO_SECTOR: dict[str, str] = {}
for _sec, _syms in SECTOR_GROUPS.items():
    for _s in _syms:
        SYMBOL_TO_SECTOR[_s.replace(".NS", "")] = _sec

# ── Geopolitical theme scores (mirrors multi_analyzer) ───────────────────────
_GEO_SCORES = {
    "🛡️ Defence": 0.85, "🏦 PSU Banks": 0.70, "🏗️ Infra/Rail": 0.80,
    "⚡ Energy": 0.72, "💻 IT/Tech": 0.68, "💊 Pharma": 0.65,
    "⚙️ Metals": 0.60, "🚗 Auto/EV": 0.63, "🛒 FMCG": 0.55,
    "💰 Finance": 0.67, "🧪 Chemicals": 0.62, "🏠 Realty/Cement": 0.58,
    "📡 Telecom": 0.60, "📈 Small/Mid Cap": 0.55,
}


class SmartIntradayEngine:
    """
    Fully automated intraday stock picker.
    Scans ~500 stocks and returns top 10 with highest intraday profit probability.
    Ensures sector diversification (max 2 per sector in top 10).
    """

    def __init__(self, capital: float = 5000):
        self.capital = capital
        self._engine = IntradayEngine(capital=capital)

    def get_top10(
        self,
        trade_date: Optional[date] = None,
        max_workers: int = 20,
        universe: Optional[list[str]] = None,
    ) -> dict:
        """
        Returns top 10 intraday picks for the given date.
        trade_date: if None, uses today.
        """
        trade_date = trade_date or date.today()
        symbols = universe or INTRADAY_STOCKS[:]

        candidates = []
        done = 0
        total = len(symbols)

        def _eval(sym: str) -> Optional[dict]:
            try:
                return self._score_symbol(sym)
            except Exception as e:
                logger.debug("SmartIntraday._score_symbol %s: %s", sym, e)
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_eval, sym): sym for sym in symbols}
            for fut in as_completed(futs):
                done += 1
                res = fut.result()
                if res and res["composite"] >= MIN_CONFIDENCE:
                    candidates.append(res)

        # Sort by composite score
        candidates.sort(key=lambda x: x["composite"], reverse=True)

        # Sector diversification: max 2 per sector
        top10, sector_count = [], {}
        for c in candidates:
            sec = c["sector"]
            if sector_count.get(sec, 0) < 2:
                top10.append(c)
                sector_count[sec] = sector_count.get(sec, 0) + 1
            if len(top10) >= 10:
                break

        return {
            "trade_date": str(trade_date),
            "generated_at": datetime.now().strftime("%d %b %Y %I:%M %p"),
            "scanned": total,
            "candidates_found": len(candidates),
            "top10": top10,
            "sector_distribution": sector_count,
        }

    def _score_symbol(self, symbol: str) -> Optional[dict]:
        # ── Intraday data (5m candles) ────────────────────────────────────────
        df_intra = self._engine.fetch_intraday(symbol, period="5d", interval="5m")
        if df_intra.empty or len(df_intra) < 30:
            return None

        price = float(df_intra["Close"].iloc[-1])
        if price > self.capital * 0.95:
            return None

        df_intra = self._engine.add_indicators(df_intra)
        intra_score = self._engine.score_stock(df_intra)

        # ── Daily data for ML + features ─────────────────────────────────────
        df_daily = DataService.fetch_ohlcv(symbol, period="6mo")
        if df_daily.empty or len(df_daily) < 60:
            return None

        # ── Next-day ML prediction ────────────────────────────────────────────
        try:
            pred = PredictionEngine.predict_next_day(df_daily)
            ml_conf = pred.get("confidence", 0.5) if "error" not in pred else 0.5
            ml_dir = pred.get("direction", "NEUTRAL") if "error" not in pred else "NEUTRAL"
        except Exception:
            ml_conf, ml_dir = 0.5, "NEUTRAL"

        # ── Sector geo score ──────────────────────────────────────────────────
        clean = symbol.replace(".NS", "")
        sector = SYMBOL_TO_SECTOR.get(clean, "Other")
        geo = _GEO_SCORES.get(sector, 0.55)

        # ── Volume momentum ───────────────────────────────────────────────────
        vol_score = float(np.clip((intra_score["vol_ratio"] - 1) / 2, 0, 0.3))

        # ── Composite intraday score ──────────────────────────────────────────
        composite = (
            0.40 * intra_score["score"]   # TA: VWAP, EMA, RSI, Supertrend, MACD
            + 0.25 * ml_conf              # ML next-day direction
            + 0.20 * vol_score            # Volume momentum
            + 0.15 * geo                  # Geopolitical tailwind
        )
        composite = float(np.clip(composite, 0, 1))

        if composite < MIN_CONFIDENCE:
            return None

        # ── Price levels ──────────────────────────────────────────────────────
        atr = max(intra_score["atr"], price * 0.005)
        entry = round(price, 2)
        stop_loss = round(max(entry - 1.5 * atr, entry * 0.93), 2)
        target1 = round(entry + 2.0 * atr, 2)
        target2 = round(entry + 3.0 * atr, 2)
        risk = entry - stop_loss
        rr = round((target1 - entry) / risk, 2) if risk > 0 else 0
        qty = max(1, int(self.capital // price))

        # ── Reasoning ─────────────────────────────────────────────────────────
        reasons = list(intra_score["reasons"])
        if ml_dir == "UP":
            reasons.append(f"ML bullish ({ml_conf:.0%} confidence)")
        if geo >= 0.70:
            reasons.append(f"Strong sector tailwind ({sector})")

        return {
            "symbol": clean,
            "sector": sector,
            "price": entry,
            "target1": target1,
            "target2": target2,
            "stop_loss": stop_loss,
            "risk_reward": rr,
            "composite": round(composite, 4),
            "intra_score": round(intra_score["score"], 4),
            "ml_confidence": round(ml_conf, 4),
            "ml_direction": ml_dir,
            "geo_score": round(geo, 4),
            "rsi": intra_score["rsi"],
            "vwap": intra_score["vwap"],
            "vol_ratio": intra_score["vol_ratio"],
            "supertrend": intra_score["supertrend"],
            "qty": qty,
            "invested": round(qty * entry, 2),
            "potential_profit": round(qty * (target1 - entry), 2),
            "potential_loss": round(qty * risk, 2),
            "is_penny": price <= PENNY_MAX_PRICE,
            "reasons": reasons,
            "signal": "BUY" if composite >= 0.55 else "WATCH",
            "global_factors": StockMetadata.get_global_factors(sector),
        }

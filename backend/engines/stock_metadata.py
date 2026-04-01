"""
Stock Metadata Engine
Provides market cap classification, penny stock detection,
sector comparison, and global factor analysis.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Optional
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

# ── Market cap thresholds (INR crores) ───────────────────────────────────────
LARGE_CAP_MIN  = 20_000   # > ₹20,000 Cr
MID_CAP_MIN    = 5_000    # ₹5,000 – 20,000 Cr
SMALL_CAP_MIN  = 500      # ₹500 – 5,000 Cr
PENNY_MAX_PRICE = 50      # Price ≤ ₹50 = penny stock

# ── Global macro factors affecting Indian markets ────────────────────────────
GLOBAL_FACTORS = {
    "🛡️ Defence": {
        "positive": ["India-US 10yr defence deal", "Make in India push", "Rising defence budget (+9.5% FY26)", "India-Pakistan tensions → procurement surge", "HAL/BEL order books at record highs"],
        "negative": ["Import dependency on Russia/Israel", "Long gestation periods", "Govt capex delays"],
        "theme": "STRONG TAILWIND",
    },
    "🏦 PSU Banks": {
        "positive": ["Credit growth 14%+ YoY", "NPA cycle bottoming out", "Govt recapitalisation", "Rural credit expansion"],
        "negative": ["Interest rate sensitivity", "Competition from private banks", "Regulatory tightening"],
        "theme": "MODERATE TAILWIND",
    },
    "🏗️ Infra/Rail": {
        "positive": ["₹11L cr budget allocation", "PM Gati Shakti", "Railway modernisation", "Smart cities mission"],
        "negative": ["Land acquisition delays", "Commodity cost inflation", "Execution risk"],
        "theme": "STRONG TAILWIND",
    },
    "⚡ Energy": {
        "positive": ["500 GW renewable target by 2030", "Energy security post-Russia sanctions", "Green hydrogen push", "Solar PLI scheme"],
        "negative": ["Crude oil price volatility", "Regulatory uncertainty", "Grid integration challenges"],
        "theme": "STRONG TAILWIND",
    },
    "💻 IT/Tech": {
        "positive": ["India-US Mission 500 trade deal", "AI/cloud adoption", "Digital India", "Rupee depreciation boosts exports"],
        "negative": ["US recession risk", "H1B visa restrictions", "Client budget cuts"],
        "theme": "MODERATE TAILWIND",
    },
    "💊 Pharma": {
        "positive": ["China+1 API sourcing shift", "US generic drug approvals", "India-US pharma trade", "Biosimilars opportunity"],
        "negative": ["USFDA inspection risk", "Price erosion in US generics", "Raw material costs"],
        "theme": "MODERATE TAILWIND",
    },
    "⚙️ Metals": {
        "positive": ["Infrastructure demand", "China stimulus", "EV battery metals demand"],
        "negative": ["China slowdown risk", "Global recession", "Commodity cycle volatility"],
        "theme": "NEUTRAL",
    },
    "🚗 Auto/EV": {
        "positive": ["EV PLI scheme ₹26,000 Cr", "Rural demand recovery", "Export growth", "Premiumisation trend"],
        "negative": ["EV transition disruption", "Semiconductor shortage", "Rising input costs"],
        "theme": "MODERATE TAILWIND",
    },
    "🛒 FMCG": {
        "positive": ["Rural consumption recovery", "Premiumisation", "Modern trade expansion"],
        "negative": ["Input cost inflation", "Competition from D2C brands", "Slow rural wage growth"],
        "theme": "NEUTRAL",
    },
    "💰 Finance": {
        "positive": ["Financial inclusion push", "Digital lending growth", "Insurance penetration rising"],
        "negative": ["RBI rate sensitivity", "Asset quality concerns", "Regulatory changes"],
        "theme": "MODERATE TAILWIND",
    },
    "🧪 Chemicals": {
        "positive": ["China+1 supply chain shift", "Specialty chemicals demand", "Export growth"],
        "negative": ["China dumping risk", "Environmental regulations", "Feedstock volatility"],
        "theme": "MODERATE TAILWIND",
    },
    "🏠 Realty/Cement": {
        "positive": ["Housing demand surge", "Infra supercycle", "Smart cities"],
        "negative": ["Interest rate sensitivity", "Affordability concerns", "Regulatory delays"],
        "theme": "MODERATE TAILWIND",
    },
    "📡 Telecom": {
        "positive": ["5G rollout acceleration", "Data consumption growth", "Digital India"],
        "negative": ["High capex burden", "ARPU pressure", "Spectrum costs"],
        "theme": "NEUTRAL",
    },
    "📈 Small/Mid Cap": {
        "positive": ["Domestic consumption story", "Niche market leaders", "High growth potential"],
        "negative": ["Liquidity risk", "Governance concerns", "High volatility"],
        "theme": "HIGH RISK / HIGH REWARD",
    },
}


class StockMetadata:

    @staticmethod
    def classify_price(price: float) -> str:
        if price <= PENNY_MAX_PRICE:
            return "PENNY"
        elif price <= 500:
            return "SMALL"
        elif price <= 2000:
            return "MID"
        else:
            return "LARGE"

    @staticmethod
    def get_risk_level(price: float, volatility: float, rsi: float) -> str:
        score = 0
        if price <= PENNY_MAX_PRICE:
            score += 3
        elif price <= 200:
            score += 2
        if volatility > 0.40:
            score += 2
        elif volatility > 0.25:
            score += 1
        if rsi > 75 or rsi < 25:
            score += 1
        if score >= 4:
            return "🔴 VERY HIGH"
        elif score >= 3:
            return "🟠 HIGH"
        elif score >= 2:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"

    @staticmethod
    def get_holding_duration(mode: str, volatility: float, confidence: float) -> str:
        if mode == "intraday":
            return "Exit by 3:15 PM today"
        if confidence >= 0.75 and volatility < 0.25:
            return "3–6 months (strong conviction)"
        elif confidence >= 0.60:
            return "1–3 months"
        else:
            return "2–4 weeks (swing)"

    @staticmethod
    def get_global_factors(sector: str) -> dict:
        clean = sector.strip()
        for key in GLOBAL_FACTORS:
            if any(word in clean for word in key.split()):
                return GLOBAL_FACTORS[key]
        return {
            "positive": ["Domestic demand growth", "India growth story"],
            "negative": ["Global macro uncertainty", "Rupee volatility"],
            "theme": "NEUTRAL",
        }

    @staticmethod
    def generate_report(symbol: str, price: float, sector: str,
                        confidence: float, direction: str,
                        predicted_return: float, volatility: float,
                        rsi: float, entry: float, target: float,
                        stop_loss: float, mode: str = "intraday") -> dict:
        price_class = StockMetadata.classify_price(price)
        risk = StockMetadata.get_risk_level(price, volatility, rsi)
        holding = StockMetadata.get_holding_duration(mode, volatility, confidence)
        factors = StockMetadata.get_global_factors(sector)
        reward_pct = ((target - entry) / entry * 100) if entry > 0 else 0
        risk_pct = ((entry - stop_loss) / entry * 100) if entry > 0 else 0

        return {
            "symbol": symbol,
            "price_class": price_class,
            "is_penny": price_class == "PENNY",
            "risk_level": risk,
            "holding_duration": holding,
            "global_factors": factors,
            "entry": entry,
            "target": target,
            "stop_loss": stop_loss,
            "reward_pct": round(reward_pct, 2),
            "risk_pct": round(risk_pct, 2),
            "confidence": confidence,
            "direction": direction,
            "predicted_return": predicted_return,
            "volatility": round(volatility * 100, 1),
            "rsi": rsi,
            "sector": sector,
            "signal": "STRONG BUY" if confidence >= 0.75 else ("BUY" if confidence >= 0.55 else "WATCH"),
            "buy_sell": "BUY" if direction in ("UP", "NEUTRAL") and confidence >= 0.55 else "SELL/AVOID",
        }

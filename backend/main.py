"""FastAPI backend — QuantSignal India."""
from __future__ import annotations

import logging
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .config import (
    ALL_SYMBOLS, MIN_CONFIDENCE, MIN_RISK_REWARD, MAX_SIGNALS,
    RISK_FREE_RATE, MC_SIMULATIONS, MC_DAYS, SECTOR_MAP,
)
from .engines.data_service import DataService
from .engines.feature_engine import FeatureEngine
from .engines.ml_engine import MLEngine
from .engines.signal_engine import SignalEngine
from .engines.monte_carlo import MonteCarloEngine
from .engines.risk_engine import RiskEngine
from .engines.portfolio_engine import PortfolioEngine
from .engines.options_engine import OptionsEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QuantSignal India",
    description="Quant trading signals for NSE — ~200 stocks, 13 sectors",
    version="3.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

signal_engine = SignalEngine()


def _sym(s: str) -> str:
    s = s.strip().upper()
    return s if s.endswith(".NS") else s + ".NS"


@app.get("/")
def root():
    return {"status": "ok", "service": "QuantSignal India v3.0",
            "universe": len(ALL_SYMBOLS), "sectors": list(SECTOR_MAP.keys())}


@app.get("/api/signals")
def get_signals(
    capital: float = Query(1_000_000, ge=10_000),
    risk_pct: float = Query(0.02, ge=0.005, le=0.1),
    min_confidence: float = Query(MIN_CONFIDENCE, ge=0.0, le=1.0),
    min_rr: float = Query(MIN_RISK_REWARD, ge=0.5),
    max_signals: int = Query(MAX_SIGNALS, ge=1, le=50),
    sector: Optional[str] = Query(None, description="Sector name from SECTOR_MAP"),
):
    symbols = SECTOR_MAP.get(sector) if sector else None
    signals = signal_engine.generate_signals(
        symbols=symbols, min_confidence=min_confidence, min_rr=min_rr,
        max_signals=max_signals, capital=capital, risk_pct=risk_pct,
    )
    return {"count": len(signals), "capital": capital, "signals": signals}


@app.get("/api/market-score")
def get_market_score():
    df = DataService.fetch_index(period="2y")
    if df.empty:
        raise HTTPException(502, "Unable to fetch NIFTY data")
    entry_score = FeatureEngine.compute_entry_score(df)
    signal = "STRONG BUY" if entry_score >= 0.6 else ("MODERATE" if entry_score >= 0.3 else "DEFENSIVE")
    last = FeatureEngine.compute_all_features(df).iloc[-1]
    return {
        "entry_score": round(entry_score, 4), "signal": signal,
        "nifty_price": round(float(df["Close"].iloc[-1]), 2),
        "components": {
            "momentum": round(float(last.get("Momentum", 0)), 4),
            "volatility": round(float(last.get("Volatility", 0)), 4),
            "drawdown": round(float(last.get("Drawdown", 0)), 4),
            "rsi": round(float(last.get("RSI", 50)), 2),
        },
    }


@app.get("/api/prediction/{symbol}")
def get_prediction(symbol: str):
    sym = _sym(symbol)
    df = DataService.fetch_ohlcv(sym, period="2y")
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")
    ml_features = FeatureEngine.get_ml_features(df)
    if ml_features.empty or len(ml_features) < 60:
        raise HTTPException(422, f"Insufficient data for {symbol}")
    ml = MLEngine()
    result = ml.get_stock_prediction(ml_features, df)
    result["symbol"] = symbol.upper().replace(".NS", "")
    result["current_price"] = round(float(df["Close"].iloc[-1]), 2)
    return result


@app.get("/api/risk/{symbol}")
def get_risk(symbol: str):
    sym = _sym(symbol)
    df = DataService.fetch_ohlcv(sym, period="2y")
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")
    return {
        "symbol": symbol.upper().replace(".NS", ""),
        "metrics": RiskEngine.compute_all(df["Close"]),
        "stress_tests": RiskEngine.stress_test(df["Close"].pct_change().dropna()),
    }


@app.get("/api/monte-carlo/{symbol}")
def get_monte_carlo(
    symbol: str,
    simulations: int = Query(MC_SIMULATIONS, ge=100, le=50000),
    days: int = Query(MC_DAYS, ge=5, le=252),
):
    sym = _sym(symbol)
    df = DataService.fetch_ohlcv(sym, period="1y")
    if df.empty:
        raise HTTPException(404, f"No data for {symbol}")
    result = MonteCarloEngine.simulate(df["Close"], simulations, days)
    result["symbol"] = symbol.upper().replace(".NS", "")
    return result


@app.get("/api/portfolio")
def get_portfolio(
    symbols: str = Query("RELIANCE,TCS,HDFCBANK,INFY,HAL,BEL,RVNL,NTPC,SBIN,BHARTIARTL"),
    n_portfolios: int = Query(10000, ge=1000, le=50000),
):
    sym_list = [_sym(s) for s in symbols.split(",")]
    if len(sym_list) < 2:
        raise HTTPException(400, "Need at least 2 stocks")
    price_dict = DataService.fetch_multiple(sym_list, period="1y")
    if len(price_dict) < 2:
        raise HTTPException(502, "Could not fetch enough data")
    return PortfolioEngine.optimize(price_dict, n_portfolios)


@app.get("/api/options")
def get_options(
    spot: float = Query(..., gt=0),
    strike: float = Query(..., gt=0),
    expiry_days: int = Query(30, ge=1, le=365),
    rate: float = Query(RISK_FREE_RATE),
    volatility: float = Query(0.20, gt=0, le=2.0),
):
    return OptionsEngine.black_scholes(spot, strike, expiry_days / 365.0, rate, volatility)


@app.get("/api/sectors")
def get_sectors():
    return {"sectors": {k: len(v) for k, v in SECTOR_MAP.items()},
            "total_symbols": len(ALL_SYMBOLS)}


@app.delete("/api/cache")
def clear_cache():
    DataService.clear_cache()
    return {"status": "cache_cleared"}

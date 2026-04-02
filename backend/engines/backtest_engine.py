"""
Backtest Engine — QuantSignal India
====================================
BRD Fix: Issue 5 — Compare signal predictions vs actual historical outcomes.
Returns WIN / LOSS / HOLD with accuracy %, win rate, and PnL per stock.

Logic:
  - WIN  : actual HIGH >= target price (target was hit)
  - LOSS : actual LOW  <= stoploss     (stoploss was hit first)
  - HOLD : neither hit within lookforward window
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    symbol:       str
    entry_price:  float
    target:       float
    stoploss:     float
    outcome:      str    # "WIN" | "LOSS" | "HOLD"
    actual_high:  float
    actual_low:   float
    pnl_pct:      float
    days_checked: int


def run_backtest(
    symbol: str,
    entry_price: float,
    target: float,
    stoploss: float,
    lookforward_days: int = 10,
) -> Optional[BacktestResult]:
    """
    Backtest a single prediction against real historical data.

    Fetches the last N trading days and checks whether target or
    stoploss was hit. Whichever hit first determines the outcome.

    Returns None if data is unavailable (never returns fake results).
    """
    if entry_price <= 0 or target <= 0 or stoploss <= 0:
        logger.warning(f"{symbol}: invalid prices — entry={entry_price} target={target} sl={stoploss}")
        return None
    if target <= entry_price:
        logger.warning(f"{symbol}: target {target} must be above entry {entry_price}")
        return None
    if stoploss >= entry_price:
        logger.warning(f"{symbol}: stoploss {stoploss} must be below entry {entry_price}")
        return None

    try:
        yf_sym = symbol if symbol.endswith((".NS", ".BO")) else f"{symbol}.NS"
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period=f"{lookforward_days + 10}d", interval="1d")

        if hist is None or hist.empty or len(hist) < 2:
            logger.warning(f"{symbol}: no historical data returned")
            return None

        # Use last N rows as the "forward" window
        forward = hist.tail(lookforward_days)
        if forward.empty:
            return None

        actual_high = float(forward["High"].max())
        actual_low  = float(forward["Low"].min())
        days_checked = len(forward)

        # Determine day-by-day which hits first: target or stoploss
        outcome = "HOLD"
        pnl_pct = 0.0
        for _, row in forward.iterrows():
            day_high = float(row["High"])
            day_low  = float(row["Low"])
            # Check stoploss first (conservative — worst case)
            if day_low <= stoploss:
                outcome = "LOSS"
                pnl_pct = round(((stoploss - entry_price) / entry_price) * 100, 2)
                break
            if day_high >= target:
                outcome = "WIN"
                pnl_pct = round(((target - entry_price) / entry_price) * 100, 2)
                break

        if outcome == "HOLD":
            last_close = float(forward["Close"].iloc[-1])
            pnl_pct = round(((last_close - entry_price) / entry_price) * 100, 2)

        return BacktestResult(
            symbol=symbol,
            entry_price=round(entry_price, 2),
            target=round(target, 2),
            stoploss=round(stoploss, 2),
            outcome=outcome,
            actual_high=round(actual_high, 2),
            actual_low=round(actual_low, 2),
            pnl_pct=pnl_pct,
            days_checked=days_checked,
        )

    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}")
        return None


def run_batch_backtest(predictions: list[dict], lookforward_days: int = 10) -> dict:
    """
    Run backtest on a list of predictions.

    Each prediction dict must have:
        { symbol, entry_price, target, stoploss }

    Returns:
        {
            total, wins, losses, holds,
            win_rate_pct, accuracy_pct, total_pnl_pct,
            results: [list of BacktestResult dicts]
        }
    """
    if not predictions:
        return {"error": "No predictions provided", "total": 0}

    results: list[BacktestResult] = []
    for p in predictions:
        sym = p.get("symbol", "")
        ep  = float(p.get("entry_price", p.get("price", 0)))
        tgt = float(p.get("target", 0))
        sl  = float(p.get("stoploss", p.get("stop_loss", 0)))

        if not sym or ep <= 0 or tgt <= 0 or sl <= 0:
            logger.warning(f"Skipping invalid prediction: {p}")
            continue

        r = run_backtest(sym, ep, tgt, sl, lookforward_days)
        if r:
            results.append(r)

    if not results:
        return {"error": "No valid results — check symbol data availability", "total": 0}

    wins   = [r for r in results if r.outcome == "WIN"]
    losses = [r for r in results if r.outcome == "LOSS"]
    holds  = [r for r in results if r.outcome == "HOLD"]
    total  = len(results)
    total_pnl = sum(r.pnl_pct for r in results)

    return {
        "total":          total,
        "wins":           len(wins),
        "losses":         len(losses),
        "holds":          len(holds),
        "win_rate_pct":   round(len(wins) / total * 100, 1),
        "accuracy_pct":   round((len(wins) + len(holds)) / total * 100, 1),
        "total_pnl_pct":  round(total_pnl, 2),
        "avg_pnl_pct":    round(total_pnl / total, 2),
        "results":        [asdict(r) for r in results],
    }


def quick_backtest_10(symbols: list[str], entry_prices: dict[str, float]) -> dict:
    """
    Quick backtest on up to 10 random symbols using a simple
    5% target / 2% stoploss assumption. For BRD validation testing.
    """
    import random
    sample = random.sample(symbols, min(10, len(symbols)))
    predictions = []
    for sym in sample:
        price = entry_prices.get(sym, 100.0)
        predictions.append({
            "symbol":       sym,
            "entry_price":  price,
            "target":       round(price * 1.05, 2),   # 5% target
            "stoploss":     round(price * 0.98, 2),   # 2% stoploss
        })
    return run_batch_backtest(predictions)

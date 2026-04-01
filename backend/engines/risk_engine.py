"""Risk Engine — VaR, CVaR, Sharpe, Drawdown, Stress Testing."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ..config import RISK_FREE_RATE


class RiskEngine:
    @staticmethod
    def compute_all(prices: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> dict:
        if len(prices) < 30:
            return {"error": "Insufficient data"}
        returns = prices.pct_change().dropna()
        return {
            "var_historical": RiskEngine.var_historical(returns),
            "var_parametric": RiskEngine.var_parametric(returns),
            "var_monte_carlo": RiskEngine.var_monte_carlo(returns),
            "cvar": RiskEngine.cvar(returns),
            "sharpe_ratio": RiskEngine.sharpe_ratio(returns, risk_free_rate),
            "sortino_ratio": RiskEngine.sortino_ratio(returns, risk_free_rate),
            "max_drawdown": RiskEngine.max_drawdown(prices),
            "volatility_annual": round(float(returns.std() * np.sqrt(252)), 4),
            "total_return": round(float((prices.iloc[-1] / prices.iloc[0]) - 1), 4),
            "calmar_ratio": RiskEngine.calmar_ratio(prices, risk_free_rate),
        }

    @staticmethod
    def var_historical(returns: pd.Series, confidence: float = 0.95) -> dict:
        var = float(np.percentile(returns, (1 - confidence) * 100))
        return {"confidence": confidence, "var": round(var, 6), "method": "historical"}

    @staticmethod
    def var_parametric(returns: pd.Series, confidence: float = 0.95) -> dict:
        from scipy.stats import norm
        mu, sigma = float(returns.mean()), float(returns.std())
        var = mu + norm.ppf(1 - confidence) * sigma
        return {"confidence": confidence, "var": round(var, 6), "method": "parametric"}

    @staticmethod
    def var_monte_carlo(returns: pd.Series, confidence: float = 0.95, n_sims: int = 10000) -> dict:
        mu, sigma = float(returns.mean()), float(returns.std())
        simulated = np.random.normal(mu, sigma, n_sims)
        var = float(np.percentile(simulated, (1 - confidence) * 100))
        return {"confidence": confidence, "var": round(var, 6), "method": "monte_carlo"}

    @staticmethod
    def cvar(returns: pd.Series, confidence: float = 0.95) -> dict:
        var_val = float(np.percentile(returns, (1 - confidence) * 100))
        tail = returns[returns <= var_val]
        cvar_val = float(tail.mean()) if len(tail) > 0 else var_val
        return {"confidence": confidence, "cvar": round(cvar_val, 6)}

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
        excess = returns.mean() * 252 - risk_free_rate
        vol = returns.std() * np.sqrt(252)
        return round(float(excess / vol), 4) if vol > 0 else 0.0

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
        excess = returns.mean() * 252 - risk_free_rate
        downside = returns[returns < 0].std() * np.sqrt(252)
        return round(float(excess / downside), 4) if downside > 0 else 0.0

    @staticmethod
    def max_drawdown(prices: pd.Series) -> dict:
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        max_dd = float(drawdown.min())
        peak_idx = cummax[:drawdown.idxmin()].idxmax() if not drawdown.empty else None
        trough_idx = drawdown.idxmin() if not drawdown.empty else None
        return {
            "max_drawdown": round(max_dd, 4),
            "peak_date": str(peak_idx.date()) if peak_idx is not None else None,
            "trough_date": str(trough_idx.date()) if trough_idx is not None else None,
        }

    @staticmethod
    def calmar_ratio(prices: pd.Series, risk_free_rate: float = RISK_FREE_RATE) -> float:
        annual_return = float((prices.iloc[-1] / prices.iloc[0]) - 1)
        max_dd = abs(RiskEngine.max_drawdown(prices)["max_drawdown"])
        return round(float((annual_return - risk_free_rate) / max_dd), 4) if max_dd > 0 else 0.0

    @staticmethod
    def stress_test(returns: pd.Series) -> list[dict]:
        current_return = float(returns.mean())
        scenarios = [
            {"name": "Black Swan (-20%)", "shock": -0.20},
            {"name": "Market Crash (-10%)", "shock": -0.10},
            {"name": "Mild Correction (-5%)", "shock": -0.05},
            {"name": "Base Case", "shock": 0.0},
            {"name": "Bull Rally (+10%)", "shock": 0.10},
        ]
        return [
            {"scenario": s["name"], "shock": s["shock"],
             "annualized_impact": round((current_return + s["shock"]) * 252, 4)}
            for s in scenarios
        ]

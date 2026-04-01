"""Monte Carlo Simulation Engine — GBM-based price simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ..config import MC_SIMULATIONS, MC_DAYS


class MonteCarloEngine:
    @staticmethod
    def simulate(prices: pd.Series, n_simulations: int = MC_SIMULATIONS, n_days: int = MC_DAYS) -> dict:
        if len(prices) < 30:
            return {"error": "Insufficient price history"}
        log_returns = np.log(prices / prices.shift(1)).dropna()
        mu = float(log_returns.mean())
        sigma = float(log_returns.std())
        S0 = float(prices.iloc[-1])
        dt = 1 / 252
        Z = np.random.standard_normal((n_simulations, n_days))
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * Z
        paths = np.zeros((n_simulations, n_days + 1))
        paths[:, 0] = S0
        for t in range(1, n_days + 1):
            paths[:, t] = paths[:, t - 1] * np.exp(drift + diffusion[:, t - 1])
        final_prices = paths[:, -1]
        returns_final = (final_prices / S0) - 1
        var_95 = float(np.percentile(returns_final, 5))
        cvar_95 = float(np.mean(returns_final[returns_final <= var_95]))
        sample_idx = np.linspace(0, n_simulations - 1, 100, dtype=int)
        return {
            "current_price": round(S0, 2),
            "expected_price": round(float(np.mean(final_prices)), 2),
            "median_price": round(float(np.median(final_prices)), 2),
            "p5": round(float(np.percentile(final_prices, 5)), 2),
            "p25": round(float(np.percentile(final_prices, 25)), 2),
            "p75": round(float(np.percentile(final_prices, 75)), 2),
            "p95": round(float(np.percentile(final_prices, 95)), 2),
            "prob_profit": round(float(np.mean(final_prices > S0)), 4),
            "var_95": round(var_95, 4),
            "cvar_95": round(cvar_95, 4),
            "n_simulations": n_simulations,
            "n_days": n_days,
            "mu": round(mu, 6),
            "sigma": round(sigma, 6),
            "sample_paths": paths[sample_idx].tolist(),
            "final_distribution": final_prices.tolist(),
        }

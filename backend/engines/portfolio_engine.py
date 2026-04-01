"""Portfolio Optimization Engine — Efficient Frontier + Mean-Variance Optimization."""
from __future__ import annotations

import numpy as np
import pandas as pd
from ..config import PORTFOLIO_SIMULATIONS, RISK_FREE_RATE


class PortfolioEngine:
    @staticmethod
    def optimize(price_dict: dict[str, pd.DataFrame], n_portfolios: int = PORTFOLIO_SIMULATIONS,
                 risk_free_rate: float = RISK_FREE_RATE) -> dict:
        closes = pd.DataFrame({sym: df["Close"] for sym, df in price_dict.items() if not df.empty})
        closes = closes.dropna()
        if closes.shape[1] < 2 or len(closes) < 60:
            return {"error": "Need at least 2 stocks with 60+ days of data"}
        returns = closes.pct_change().dropna()
        symbols = list(returns.columns)
        n_assets = len(symbols)
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        results = np.zeros((n_portfolios, 3))
        weight_matrix = np.zeros((n_portfolios, n_assets))
        for i in range(n_portfolios):
            w = np.random.dirichlet(np.ones(n_assets))
            weight_matrix[i] = w
            port_return = float(np.dot(w, mean_returns))
            port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix.values, w))))
            results[i] = [port_return, port_vol,
                          (port_return - risk_free_rate) / port_vol if port_vol > 0 else 0]
        max_sharpe_idx = results[:, 2].argmax()
        min_vol_idx = results[:, 1].argmin()

        def _portfolio(idx):
            w = weight_matrix[idx]
            return {
                "return": round(float(results[idx, 0]), 4),
                "volatility": round(float(results[idx, 1]), 4),
                "sharpe": round(float(results[idx, 2]), 4),
                "weights": {sym.replace(".NS", ""): round(float(wt), 4)
                            for sym, wt in zip(symbols, w) if wt > 0.01},
            }

        sample_idx = np.linspace(0, n_portfolios - 1, min(2000, n_portfolios), dtype=int)
        return {
            "symbols": [s.replace(".NS", "") for s in symbols],
            "max_sharpe": _portfolio(max_sharpe_idx),
            "min_variance": _portfolio(min_vol_idx),
            "scatter": [{"return": round(float(results[i, 0]), 4),
                         "volatility": round(float(results[i, 1]), 4),
                         "sharpe": round(float(results[i, 2]), 4)} for i in sample_idx],
            "n_portfolios": n_portfolios,
            "individual_stats": {
                sym.replace(".NS", ""): {
                    "annual_return": round(float(mean_returns[sym]), 4),
                    "annual_volatility": round(float(np.sqrt(cov_matrix.loc[sym, sym])), 4),
                } for sym in symbols
            },
        }

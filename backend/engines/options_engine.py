"""Options Pricing Engine — Black-Scholes model with full Greeks."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


class OptionsEngine:
    @staticmethod
    def black_scholes(spot: float, strike: float, time_to_expiry: float,
                      rate: float, volatility: float) -> dict:
        if time_to_expiry <= 0 or volatility <= 0 or spot <= 0 or strike <= 0:
            return {"error": "Invalid inputs: all values must be positive"}
        S, K, T, r, sigma = spot, strike, time_to_expiry, rate, volatility
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta_call = float(norm.cdf(d1))
        gamma = float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))
        vega = float(S * norm.pdf(d1) * np.sqrt(T)) / 100
        theta_call = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
                           - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        theta_put = float((-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)))
                          + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho_call = float(K * T * np.exp(-r * T) * norm.cdf(d2)) / 100
        rho_put = float(-K * T * np.exp(-r * T) * norm.cdf(-d2)) / 100
        return {
            "inputs": {"spot": spot, "strike": strike, "time_to_expiry": time_to_expiry,
                       "rate": rate, "volatility": volatility},
            "call": {"price": round(float(call_price), 4), "delta": round(delta_call, 4),
                     "gamma": round(gamma, 6), "theta": round(theta_call, 4),
                     "vega": round(vega, 4), "rho": round(rho_call, 4)},
            "put": {"price": round(float(put_price), 4), "delta": round(delta_call - 1, 4),
                    "gamma": round(gamma, 6), "theta": round(theta_put, 4),
                    "vega": round(vega, 4), "rho": round(rho_put, 4)},
        }

    @staticmethod
    def option_chain(spot: float, strikes: list[float], time_to_expiry: float,
                     rate: float, volatility: float) -> list[dict]:
        chain = []
        for k in strikes:
            r = OptionsEngine.black_scholes(spot, k, time_to_expiry, rate, volatility)
            if "error" not in r:
                chain.append({"strike": k, "call_price": r["call"]["price"],
                               "put_price": r["put"]["price"], "call_delta": r["call"]["delta"],
                               "put_delta": r["put"]["delta"], "gamma": r["call"]["gamma"],
                               "vega": r["call"]["vega"]})
        return chain

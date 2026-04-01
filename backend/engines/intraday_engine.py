"""Live Intraday Engine — scans NSE stocks and generates real-time BUY signals."""
from __future__ import annotations

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time
from typing import Optional
import logging
import warnings

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


class IntradayEngine:
    """Live intraday stock screener and signal generator."""

    def __init__(self, capital: float = 1000):
        self.capital = capital

    @staticmethod
    def fetch_intraday(symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        """
        Fetch intraday OHLCV using Ticker.history() — avoids the yfinance
        MultiIndex / duplicate-column bug that causes identical prices across
        different stocks when yf.download() is used in a threaded context.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return pd.DataFrame()
            # Ticker.history() never returns MultiIndex, but guard anyway
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return df
        except Exception as e:
            logger.warning(f"fetch_intraday error {symbol}: {e}")
            return pd.DataFrame()

    @staticmethod
    def fetch_daily(symbol: str, period: str = "3mo") -> pd.DataFrame:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1d")
            if df.empty:
                return pd.DataFrame()
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index)
            df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
            return df
        except Exception:
            return pd.DataFrame()

    @staticmethod
    def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c = df["Close"]
        df["EMA9"] = c.ewm(span=9).mean()
        df["EMA21"] = c.ewm(span=21).mean()
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        tp = (df["High"] + df["Low"] + df["Close"]) / 3
        df["VWAP"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()
        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        df["BB_Upper"] = ma20 + 2 * std20
        df["BB_Lower"] = ma20 - 2 * std20
        high, low, prev_close = df["High"], df["Low"], c.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        df["ATR"] = tr.rolling(14).mean()
        df["Vol_MA20"] = df["Volume"].rolling(20).mean()
        df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"].replace(0, np.nan)
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        df = IntradayEngine._supertrend(df, period=10, multiplier=3)
        df["Change_Pct"] = c.pct_change() * 100
        return df

    @staticmethod
    def _supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3) -> pd.DataFrame:
        hl2 = (df["High"] + df["Low"]) / 2
        atr = df["ATR"] if "ATR" in df.columns else (df["High"] - df["Low"]).rolling(period).mean()
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        supertrend.iloc[0] = upper.iloc[0]
        direction.iloc[0] = -1
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["Close"].iloc[i] < lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = (
                    max(lower.iloc[i], supertrend.iloc[i - 1])
                    if direction.iloc[i - 1] == 1 else lower.iloc[i]
                )
            else:
                supertrend.iloc[i] = (
                    min(upper.iloc[i], supertrend.iloc[i - 1])
                    if direction.iloc[i - 1] == -1 else upper.iloc[i]
                )
        df["Supertrend"] = supertrend
        df["ST_Direction"] = direction
        return df

    @staticmethod
    def score_stock(df: pd.DataFrame) -> dict:
        if len(df) < 30:
            return {"score": 0, "reasons": ["Insufficient data"], "atr": 0,
                    "rsi": 50, "vwap": 0, "vol_ratio": 1, "ema9": 0, "ema21": 0, "supertrend": "SELL"}
        last = df.iloc[-1]
        prev = df.iloc[-2]
        score = 0.0
        reasons = []
        price = float(last["Close"])
        rsi = float(last.get("RSI", 50))
        vwap = float(last.get("VWAP", price))
        ema9 = float(last.get("EMA9", price))
        ema21 = float(last.get("EMA21", price))
        vol_ratio = float(last.get("Vol_Ratio", 1))
        macd_hist = float(last.get("MACD_Hist", 0))
        st_dir = float(last.get("ST_Direction", 0))
        atr = float(last.get("ATR", price * 0.01))
        bb_lower = float(last.get("BB_Lower", price * 0.98))

        if price > vwap:
            score += 0.15
            reasons.append("Price > VWAP")
        elif abs(price - vwap) / vwap < 0.003:
            score += 0.10
            reasons.append("Price at VWAP support")

        if ema9 > ema21:
            score += 0.15
            reasons.append("EMA9 > EMA21 (bullish)")
            if float(prev.get("EMA9", 0)) <= float(prev.get("EMA21", 0)):
                score += 0.10
                reasons.append("Fresh EMA crossover!")

        if 30 <= rsi <= 45:
            score += 0.15
            reasons.append(f"RSI oversold bounce ({rsi:.0f})")
        elif 45 < rsi <= 65:
            score += 0.10
            reasons.append(f"RSI momentum zone ({rsi:.0f})")
        elif rsi > 75:
            score -= 0.10
            reasons.append(f"RSI overbought ({rsi:.0f})")

        if vol_ratio > 2.0:
            score += 0.15
            reasons.append(f"High volume ({vol_ratio:.1f}x avg)")
        elif vol_ratio > 1.3:
            score += 0.10
            reasons.append(f"Volume above avg ({vol_ratio:.1f}x)")
        elif vol_ratio < 0.5:
            score -= 0.05
            reasons.append("Low volume")

        if macd_hist > 0:
            score += 0.10
            reasons.append("MACD bullish")
            if float(prev.get("MACD_Hist", 0)) <= 0:
                score += 0.05
                reasons.append("Fresh MACD crossover")

        if st_dir == 1:
            score += 0.15
            reasons.append("Supertrend BUY")

        if price < bb_lower * 1.01:
            score += 0.05
            reasons.append("Near BB lower band (support)")

        score = max(0, min(1, score))
        return {
            "score": round(score, 4),
            "reasons": reasons,
            "rsi": round(rsi, 1),
            "vwap": round(vwap, 2),
            "vol_ratio": round(vol_ratio, 2),
            "atr": round(atr, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "supertrend": "BUY" if st_dir == 1 else "SELL",
        }

    def scan_for_trades(
        self,
        symbols: list[str],
        min_confidence: float = 0.55,
        max_price: float = None,
    ) -> list[dict]:
        if max_price is None:
            max_price = self.capital * 0.95
        results = []
        for sym in symbols:
            try:
                df = self.fetch_intraday(sym, period="5d", interval="5m")
                if df.empty or len(df) < 30:
                    continue
                price = float(df["Close"].iloc[-1])
                if price > max_price:
                    continue
                df = self.add_indicators(df)
                scoring = self.score_stock(df)
                if scoring["score"] < min_confidence:
                    continue
                atr = scoring["atr"]
                entry = round(price, 2)
                stop_loss = round(price - 1.5 * atr, 2)
                target1 = round(price + 2.0 * atr, 2)
                target2 = round(price + 3.0 * atr, 2)
                risk = entry - stop_loss
                reward = target1 - entry
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                qty = max(1, int(self.capital // price))
                invested = round(qty * price, 2)
                results.append({
                    "symbol": sym.replace(".NS", ""),
                    "price": entry,
                    "qty": qty,
                    "invested": invested,
                    "target_1": target1,
                    "target_2": target2,
                    "stop_loss": stop_loss,
                    "confidence": scoring["score"],
                    "risk_reward": rr_ratio,
                    "potential_profit": round(qty * (target1 - entry), 2),
                    "potential_loss": round(qty * (entry - stop_loss), 2),
                    "rsi": scoring["rsi"],
                    "vwap": scoring["vwap"],
                    "vol_ratio": scoring["vol_ratio"],
                    "supertrend": scoring["supertrend"],
                    "reasons": scoring["reasons"],
                    "signal": "BUY" if scoring["score"] >= 0.55 else "WATCH",
                })
            except Exception as e:
                logger.warning(f"Error scanning {sym}: {e}")
        results.sort(key=lambda x: (x["confidence"], x["risk_reward"]), reverse=True)
        return results

    def get_best_trade(self, symbols: list[str]) -> Optional[dict]:
        trades = self.scan_for_trades(symbols)
        return trades[0] if trades else None


def is_market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return time(9, 15) <= now.time() <= time(15, 30)


def get_market_status() -> str:
    now = datetime.now()
    if now.weekday() >= 5:
        return "CLOSED (Weekend)"
    t = now.time()
    if t < time(9, 0):
        return "PRE-MARKET (opens 9:15)"
    if t < time(9, 15):
        return "OPENING SOON"
    if t <= time(15, 30):
        return "MARKET OPEN"
    return "CLOSED (after hours)"

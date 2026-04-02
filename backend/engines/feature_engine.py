"""Feature Engineering — computes technical indicators and ML features."""
from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngine:
    """Computes technical indicators and features for ML and signal engines."""

    @staticmethod
    def add_moving_averages(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
        if windows is None:
            windows = [20, 50, 200]
        df = df.copy()
        for w in windows:
            if len(df) >= w:
                df[f"MA{w}"] = df["Close"].rolling(w).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        df = df.copy()
        delta = df["Close"].diff()
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["RSI"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        df = df.copy()
        df["Returns"] = df["Close"].pct_change()
        df["Volatility"] = df["Returns"].rolling(window).std() * np.sqrt(252)
        return df

    @staticmethod
    def add_momentum(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
        df = df.copy()
        ma = df["Close"].rolling(window).mean()
        df["Momentum"] = (df["Close"] / ma) - 1
        return df

    @staticmethod
    def add_drawdown(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
        df = df.copy()
        rolling_max = df["Close"].rolling(window, min_periods=1).max()
        df["Drawdown"] = (df["Close"] - rolling_max) / rolling_max
        return df

    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for d in [5, 10, 20, 30, 60]:
            if len(df) > d:
                df[f"Return_{d}d"] = df["Close"].pct_change(d)
        return df

    @staticmethod
    def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Volume_MA20"] = df["Volume"].rolling(20).mean()
        df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"].replace(0, np.nan)
        return df

    @staticmethod
    def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        df = df.copy()
        ma = df["Close"].rolling(window).mean()
        std = df["Close"].rolling(window).std()
        df["BB_Upper"] = ma + 2 * std
        df["BB_Lower"] = ma - 2 * std
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / ma
        df["BB_Position"] = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)
        return df

    @staticmethod
    def add_macd(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        ema12 = df["Close"].ewm(span=12).mean()
        ema26 = df["Close"].ewm(span=26).mean()
        df["MACD"] = ema12 - ema26
        df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()
        df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, spans: list[int] = None) -> pd.DataFrame:
        """Add EMA columns: EMA9, EMA21, EMA50, EMA200."""
        if spans is None:
            spans = [9, 21, 50, 200]
        df = df.copy()
        for span in spans:
            if len(df) >= span:
                df[f"EMA{span}"] = df["Close"].ewm(span=span, adjust=False).mean()
        return df

    @staticmethod
    def add_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Add Supertrend indicator."""
        df = df.copy()
        hl2 = (df["High"] + df["Low"]) / 2
        high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
        tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        upper = hl2 + multiplier * atr
        lower = hl2 - multiplier * atr
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=float)
        if len(df) < 2:
            return df
        supertrend.iloc[0] = upper.iloc[0]
        direction.iloc[0] = -1
        for i in range(1, len(df)):
            if df["Close"].iloc[i] > upper.iloc[i - 1]:
                direction.iloc[i] = 1
            elif df["Close"].iloc[i] < lower.iloc[i - 1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i - 1]
            supertrend.iloc[i] = (
                max(lower.iloc[i], supertrend.iloc[i - 1]) if direction.iloc[i] == 1
                else min(upper.iloc[i], supertrend.iloc[i - 1])
            )
        df["Supertrend"] = supertrend
        df["ST_Direction"] = direction
        return df

    @classmethod
    def compute_all_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        df = cls.add_moving_averages(df)
        df = cls.add_ema(df)
        df = cls.add_rsi(df)
        df = cls.add_volatility(df)
        df = cls.add_momentum(df)
        df = cls.add_drawdown(df)
        df = cls.add_returns(df)
        df = cls.add_volume_features(df)
        df = cls.add_bollinger_bands(df)
        df = cls.add_macd(df)
        if len(df) >= 15:
            df = cls.add_supertrend(df)
        return df

    @classmethod
    def get_ml_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature matrix ready for ML (no NaN, normalized)."""
        df = cls.compute_all_features(df)
        feature_cols = [
            "RSI", "Volatility", "Momentum", "Drawdown",
            "Return_10d", "Return_30d", "Volume_Ratio",
            "BB_Position", "MACD_Hist",
        ]
        existing = [c for c in feature_cols if c in df.columns]
        features = df[existing].copy()
        features = features.dropna()
        return features

    @classmethod
    def compute_entry_score(cls, df: pd.DataFrame) -> float:
        """Market timing entry score (0-1)."""
        if df.empty or len(df) < 200:
            return 0.5
        df = cls.compute_all_features(df)
        last = df.iloc[-1]
        momentum = last.get("Momentum", 0)
        momentum_score = np.clip((momentum + 0.2) / 0.4, 0, 1)
        vol = last.get("Volatility", 0.2)
        vol_score = np.clip(1 - (vol / 0.5), 0, 1)
        dd = last.get("Drawdown", 0)
        dd_score = np.clip(1 + dd, 0, 1)
        entry_score = 0.4 * momentum_score + 0.35 * vol_score + 0.25 * dd_score
        return float(np.clip(entry_score, 0, 1))

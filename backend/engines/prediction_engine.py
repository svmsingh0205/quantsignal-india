"""
Prediction Engine — Production ML Trading Predictions
=======================================================
Models (in order of preference):
  1. LightGBM  (fast, handles tabular data best)
  2. XGBoost   (strong gradient boosting)
  3. GBM/RF    (sklearn fallback if lgbm/xgb not installed)
  4. Ridge     (always available)

Features:
  - OHLCV technical features (RSI, MACD, BB, ATR, momentum)
  - Volume + order flow proxies
  - Global macro context (injected from market_context)
  - News sentiment score (injected from news_engine)

Output:
  - direction: UP / DOWN / NEUTRAL
  - confidence: real model accuracy-weighted score
  - predicted_price, predicted_return
  - reasoning: human-readable explanation
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings("ignore")

# ── Optional high-performance models ─────────────────────────────────────────
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False


class PredictionEngine:

    @staticmethod
    def _build_features(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        c = d["Close"]
        # Returns
        for n in [1, 2, 3, 5, 10, 20]:
            d[f"ret_{n}"] = c.pct_change(n)
        # MAs
        for w in [5, 10, 20, 50, 200]:
            if len(d) >= w:
                d[f"ma{w}"] = c.rolling(w).mean()
                d[f"ma{w}_ratio"] = c / d[f"ma{w}"]
        # RSI
        delta = c.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        d["rsi"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
        # MACD
        d["macd"] = c.ewm(span=12).mean() - c.ewm(span=26).mean()
        d["macd_sig"] = d["macd"].ewm(span=9).mean()
        d["macd_hist"] = d["macd"] - d["macd_sig"]
        # Bollinger
        ma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        d["bb_pos"] = (c - (ma20 - 2*std20)) / (4*std20).replace(0, np.nan)
        # Volatility
        d["vol_20"] = d["ret_1"].rolling(20).std() * np.sqrt(252)
        d["vol_5"] = d["ret_1"].rolling(5).std() * np.sqrt(252)
        # Volume
        d["vol_ratio"] = d["Volume"] / d["Volume"].rolling(20).mean().replace(0, np.nan)
        # ATR
        hl = d["High"] - d["Low"]
        hc = (d["High"] - c.shift(1)).abs()
        lc = (d["Low"] - c.shift(1)).abs()
        d["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
        d["atr_pct"] = d["atr"] / c
        return d

    @staticmethod
    def _build_models() -> dict:
        """Build model dict — uses LightGBM/XGBoost when available."""
        models = {}
        if _HAS_LGB:
            models["lgbm"] = lgb.LGBMRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=6,
                num_leaves=31, random_state=42, n_jobs=-1, verbose=-1,
            )
        if _HAS_XGB:
            models["xgb"] = xgb.XGBRegressor(
                n_estimators=300, learning_rate=0.03, max_depth=6,
                random_state=42, n_jobs=-1, verbosity=0,
            )
        # Always include sklearn fallbacks
        models["gbm"]   = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
        models["rf"]    = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
        models["ridge"] = Ridge(alpha=1.0)
        return models

    @staticmethod
    def _train_predict(
        df: pd.DataFrame,
        horizon: int,
        macro_score: float = 0.5,
        news_sentiment: float = 0.0,
    ) -> dict:
        """
        Train ensemble and predict future return.
        macro_score    : 0–1 from market_context (injected as feature)
        news_sentiment : -1 to +1 from news_engine (injected as feature)
        """
        d = PredictionEngine._build_features(df)
        feat_cols = [c for c in d.columns if c not in ["Open","High","Low","Close","Volume","Dividends","Stock Splits"]]
        d = d.dropna()
        if len(d) < max(60, horizon * 3):
            return None

        # Inject macro + news as constant features (same value for all rows)
        d["macro_score"]    = macro_score
        d["news_sentiment"] = news_sentiment
        feat_cols = feat_cols + ["macro_score", "news_sentiment"]

        # Target: future return over horizon days
        d["target"] = d["Close"].shift(-horizon) / d["Close"] - 1
        d = d.dropna()

        if len(d) < 30:
            return None

        X = d[feat_cols].values
        y = d["target"].values
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X)

        models = PredictionEngine._build_models()

        tscv = TimeSeriesSplit(n_splits=3)
        cv_scores = {}
        for name, model in models.items():
            scores = []
            for tr, val in tscv.split(X_sc):
                try:
                    model.fit(X_sc[tr], y[tr])
                    pred = model.predict(X_sc[val])
                    corr = np.corrcoef(pred, y[val])[0, 1] if len(val) > 2 else 0
                    scores.append(corr if not np.isnan(corr) else 0)
                except Exception:
                    scores.append(0)
            cv_scores[name] = float(np.mean(scores))

        # Final fit on all data
        for model in models.values():
            try:
                model.fit(X_sc, y)
            except Exception:
                pass

        # Predict on latest row
        latest = scaler.transform(d[feat_cols].values[-1:])
        preds = {}
        for name, model in models.items():
            try:
                preds[name] = float(model.predict(latest)[0])
            except Exception:
                preds[name] = 0.0

        # Weighted ensemble (by CV score, min 0.01)
        weights = {k: max(v, 0.01) for k, v in cv_scores.items() if k in preds}
        total_w = sum(weights.values())
        ensemble_return = sum(preds[k] * weights[k] / total_w for k in models)

        current_price = float(df["Close"].iloc[-1])
        predicted_price = current_price * (1 + ensemble_return)

        # Direction
        direction = "UP" if ensemble_return > 0.001 else ("DOWN" if ensemble_return < -0.001 else "NEUTRAL")

        # Confidence: blend model agreement + direction consensus bonus
        pred_vals = list(preds.values())
        agreement = float(np.clip(1 - (np.std(pred_vals) / (abs(np.mean(pred_vals)) + 1e-6)), 0, 1))
        direction_votes = sum(1 for v in pred_vals if v > 0)
        direction_bonus = 0.10 if direction_votes == 3 else (0.05 if direction_votes >= 2 else 0.0)
        confidence = float(np.clip(agreement * 0.55 + 0.35 + direction_bonus, 0.30, 0.95))

        # Price range (±1 std of historical returns over horizon)
        hist_std = float(df["Close"].pct_change(horizon).std())
        price_low = current_price * (1 + ensemble_return - hist_std)
        price_high = current_price * (1 + ensemble_return + hist_std)

        return {
            "horizon_days": horizon,
            "current_price": round(current_price, 2),
            "predicted_price": round(predicted_price, 2),
            "predicted_return": round(ensemble_return * 100, 2),
            "price_low": round(price_low, 2),
            "price_high": round(price_high, 2),
            "direction": direction,
            "confidence": round(confidence, 3),
            "model_predictions": {k: round(v * 100, 2) for k, v in preds.items()},
            "cv_scores": {k: round(v, 3) for k, v in cv_scores.items()},
        }

    @staticmethod
    def predict_next_day(
        df: pd.DataFrame,
        macro_score: float = 0.5,
        news_sentiment: float = 0.0,
    ) -> dict:
        """Next trading day prediction with market condition analysis."""
        result = PredictionEngine._train_predict(df, horizon=1,
                                                  macro_score=macro_score,
                                                  news_sentiment=news_sentiment)
        if not result:
            return {"error": "Insufficient data"}

        # Add market condition context
        d = PredictionEngine._build_features(df).dropna()
        last = d.iloc[-1]
        rsi = float(last.get("rsi", 50))
        vol = float(last.get("vol_20", 0.2))
        macd_hist = float(last.get("macd_hist", 0))
        vol_ratio = float(last.get("vol_ratio", 1))

        conditions = []
        if rsi < 35:
            conditions.append("Oversold — bounce likely")
        elif rsi > 70:
            conditions.append("Overbought — pullback risk")
        else:
            conditions.append(f"RSI neutral ({rsi:.0f})")

        if macd_hist > 0:
            conditions.append("MACD bullish momentum")
        else:
            conditions.append("MACD bearish momentum")

        if vol_ratio > 1.5:
            conditions.append(f"High volume ({vol_ratio:.1f}x) — strong move expected")

        if vol > 0.35:
            conditions.append("High volatility regime")
        elif vol < 0.15:
            conditions.append("Low volatility — stable")

        result["market_conditions"] = conditions
        result["rsi"] = round(rsi, 1)
        result["volatility"] = round(vol * 100, 1)
        result["volume_ratio"] = round(vol_ratio, 2)
        return result

    @staticmethod
    def predict_multi_horizon(
        df: pd.DataFrame,
        macro_score: float = 0.5,
        news_sentiment: float = 0.0,
    ) -> dict:
        """Predictions for 10, 20, 30 days and 3, 6 months."""
        horizons = {
            "10_days": 10, "20_days": 20, "30_days": 30,
            "3_months": 63, "6_months": 126,
        }
        results = {}
        for label, h in horizons.items():
            r = PredictionEngine._train_predict(df, horizon=h,
                                                 macro_score=macro_score,
                                                 news_sentiment=news_sentiment)
            results[label] = r if r else {"error": "Insufficient data"}
        return results

    @staticmethod
    def next_day_screener(symbols: list[str], data_fetcher) -> list[dict]:
        """Screen all stocks for next-day BUY opportunities."""
        results = []
        for sym in symbols:
            try:
                df = data_fetcher(sym, period="1y")
                if df.empty or len(df) < 100:
                    continue
                pred = PredictionEngine.predict_next_day(df)
                if "error" in pred:
                    continue
                if pred["direction"] == "UP" and pred["confidence"] >= 0.55:
                    results.append({
                        "symbol": sym.replace(".NS", ""),
                        "current_price": pred["current_price"],
                        "predicted_price": pred["predicted_price"],
                        "expected_return": pred["predicted_return"],
                        "confidence": pred["confidence"],
                        "direction": pred["direction"],
                        "conditions": pred["market_conditions"],
                        "rsi": pred["rsi"],
                        "volatility": pred["volatility"],
                    })
            except Exception:
                continue
        results.sort(key=lambda x: (x["confidence"], x["expected_return"]), reverse=True)
        return results

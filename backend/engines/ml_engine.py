"""ML Engine — trains ensemble models and predicts UP/DOWN probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)


class MLEngine:
    """Ensemble ML engine for market direction prediction."""

    def __init__(self):
        self.models = {
            "logistic": LogisticRegression(max_iter=1000, random_state=42),
            "random_forest": RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            ),
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
        self._accuracy: dict[str, float] = {}

    def _prepare_labels(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        future_return = df["Close"].shift(-horizon) / df["Close"] - 1
        return (future_return > 0).astype(int)

    def train(self, features: pd.DataFrame, prices: pd.DataFrame, horizon: int = 5):
        labels = self._prepare_labels(prices, horizon)
        common_idx = features.index.intersection(labels.dropna().index)
        X = features.loc[common_idx].values
        y = labels.loc[common_idx].values
        if len(X) < 60:
            logger.warning("Insufficient data for training (%d rows)", len(X))
            return
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=3)
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                model.fit(X_scaled[train_idx], y[train_idx])
                pred = model.predict(X_scaled[val_idx])
                scores.append(accuracy_score(y[val_idx], pred))
            self._accuracy[name] = float(np.mean(scores))
        for model in self.models.values():
            model.fit(X_scaled, y)
        self.is_fitted = True

    def predict_proba(self, features: pd.DataFrame) -> dict:
        if not self.is_fitted:
            return {"probability_up": 0.5, "probability_down": 0.5, "model_scores": {}}
        X = self.scaler.transform(
            features.values.reshape(1, -1) if features.ndim == 1 else features.values[-1:]
        )
        probas = {}
        for name, model in self.models.items():
            p = model.predict_proba(X)[0]
            probas[name] = float(p[1])
        avg_up = float(np.mean(list(probas.values())))
        return {
            "probability_up": round(avg_up, 4),
            "probability_down": round(1 - avg_up, 4),
            "model_scores": {k: round(v, 4) for k, v in probas.items()},
            "model_accuracies": {k: round(v, 4) for k, v in self._accuracy.items()},
        }

    def get_stock_prediction(self, features: pd.DataFrame, prices: pd.DataFrame) -> dict:
        self.train(features, prices)
        if not self.is_fitted:
            return {"probability_up": 0.5, "probability_down": 0.5, "model_scores": {}}
        return self.predict_proba(features)

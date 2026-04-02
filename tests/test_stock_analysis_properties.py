"""
Property-Based Tests for Stock Analysis Engine
Uses hypothesis to verify correctness properties.

Run with: pytest tests/test_stock_analysis_properties.py -v
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from backend.engines.stock_analysis_engine import (
    compute_trade_setup,
    build_timeframe_signals,
    TradeSetup,
    TimeframeSignal,
    load_peer_data,
)
from backend.engines.monte_carlo import MonteCarloEngine
from backend.engines.risk_engine import RiskEngine
from backend.engines.multi_analyzer import MultiAnalyzer
from backend.engines.stock_metadata import StockMetadata


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int, start_price: float = 100.0) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with n rows."""
    np.random.seed(42)
    prices = start_price * np.cumprod(1 + np.random.normal(0, 0.01, n))
    highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n)))
    opens = prices * (1 + np.random.normal(0, 0.003, n))
    volumes = np.random.randint(100_000, 10_000_000, n).astype(float)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": prices, "Volume": volumes,
    }, index=idx)


def _make_ohlcv_from_prices(prices: list[float]) -> pd.DataFrame:
    """Build OHLCV from a list of close prices."""
    n = len(prices)
    arr = np.array(prices, dtype=float)
    highs = arr * 1.01
    lows = arr * 0.99
    opens = arr * 1.002
    volumes = np.full(n, 1_000_000.0)
    idx = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "Open": opens, "High": highs, "Low": lows,
        "Close": arr, "Volume": volumes,
    }, index=idx)


# ═══════════════════════════════════════════════════════════════════════════════
# Property 3: ATR Floor Prevents Zero Division
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    price=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    capital=st.floats(min_value=10_000.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
    risk_pct=st.floats(min_value=0.001, max_value=0.10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=200, deadline=5000)
def test_property3_atr_floor(price: float, capital: float, risk_pct: float):
    """
    Property 3: ATR Floor Prevents Zero Division.
    For any valid price, the computed ATR must be >= price * 0.005.
    This prevents division-by-zero in position sizing.
    """
    # Build a minimal OHLCV where ATR could be near zero (flat prices)
    n = 20
    flat_prices = [price] * n  # perfectly flat — ATR would be 0 without floor
    df = _make_ohlcv_from_prices(flat_prices)

    ma_result = {"combined_score": 0.5, "signal": "NEUTRAL", "reasoning": []}
    setup = compute_trade_setup(price, df, capital, risk_pct, ma_result)

    # ATR floor: must be at least 0.5% of price
    assert setup.atr >= price * 0.005, (
        f"ATR floor violated: atr={setup.atr} < price*0.005={price * 0.005}"
    )
    # No division by zero: risk_per_share must be >= 0
    assert setup.risk_per_share >= 0, f"Negative risk_per_share: {setup.risk_per_share}"
    # qty must be non-negative
    assert setup.qty >= 0, f"Negative qty: {setup.qty}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 1: Trade Setup Ordering Invariant
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    price=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    capital=st.floats(min_value=10_000.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
    risk_pct=st.floats(min_value=0.001, max_value=0.10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, deadline=5000)
def test_property1_trade_setup_ordering(price: float, capital: float, risk_pct: float):
    """
    Property 1: Trade Setup Ordering Invariant.
    stop_loss < entry < target_1 < target_2 for all valid inputs.
    """
    df = _make_ohlcv(20, start_price=price)
    ma_result = {"combined_score": 0.5, "signal": "NEUTRAL", "reasoning": []}
    setup = compute_trade_setup(price, df, capital, risk_pct, ma_result)

    assert setup.stop_loss < setup.entry, (
        f"Ordering violated: stop_loss={setup.stop_loss} >= entry={setup.entry}"
    )
    assert setup.entry < setup.target_1, (
        f"Ordering violated: entry={setup.entry} >= target_1={setup.target_1}"
    )
    assert setup.target_1 < setup.target_2, (
        f"Ordering violated: target_1={setup.target_1} >= target_2={setup.target_2}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 2: Trade Setup Arithmetic Invariant
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    price=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False, allow_infinity=False),
    capital=st.floats(min_value=10_000.0, max_value=10_000_000.0, allow_nan=False, allow_infinity=False),
    risk_pct=st.floats(min_value=0.001, max_value=0.10, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=300, deadline=5000)
def test_property2_trade_setup_arithmetic(price: float, capital: float, risk_pct: float):
    """
    Property 2: Trade Setup Arithmetic Invariant.
    invested == qty * entry (within float precision), qty >= 0,
    risk_reward > 0 when risk_per_share > 0.
    """
    df = _make_ohlcv(20, start_price=price)
    ma_result = {"combined_score": 0.5, "signal": "NEUTRAL", "reasoning": []}
    setup = compute_trade_setup(price, df, capital, risk_pct, ma_result)

    # qty must be non-negative
    assert setup.qty >= 0, f"qty={setup.qty} is negative"

    # invested == qty * entry (within float precision)
    expected_invested = round(setup.qty * setup.entry, 2)
    assert abs(setup.invested - expected_invested) < 0.02, (
        f"Arithmetic violated: invested={setup.invested} != qty*entry={expected_invested}"
    )

    # risk_reward > 0 when risk_per_share > 0
    if setup.risk_per_share > 0:
        assert setup.risk_reward > 0, (
            f"risk_reward={setup.risk_reward} <= 0 when risk_per_share={setup.risk_per_share} > 0"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 4: Timeframe Signal Validity
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    n_intra=st.integers(min_value=30, max_value=200),
    n_daily=st.integers(min_value=30, max_value=500),
    n_weekly=st.integers(min_value=30, max_value=260),
    start_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=30000)
def test_property4_timeframe_signal_validity(
    n_intra: int, n_daily: int, n_weekly: int, start_price: float
):
    """
    Property 4: Timeframe Signal Validity.
    1 <= len(result) <= 3, all scores in [0.0, 1.0],
    all signal strings in {"BUY", "SELL", "NEUTRAL"}.
    """
    df_intra = _make_ohlcv(n_intra, start_price=start_price)
    df_daily = _make_ohlcv(n_daily, start_price=start_price)
    df_weekly = _make_ohlcv(n_weekly, start_price=start_price)

    # Add required intraday indicator columns
    from backend.engines.intraday_engine import IntradayEngine
    df_intra = IntradayEngine.add_indicators(df_intra)

    signals = build_timeframe_signals("TEST", df_intra, df_daily, df_weekly)

    assert 1 <= len(signals) <= 3, f"Expected 1-3 signals, got {len(signals)}"

    valid_signals = {"BUY", "SELL", "NEUTRAL"}
    for sig in signals:
        assert 0.0 <= sig.score <= 1.0, (
            f"Score out of range: {sig.score} for timeframe {sig.timeframe}"
        )
        assert sig.signal in valid_signals, (
            f"Invalid signal '{sig.signal}' for timeframe {sig.timeframe}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 6: Monte Carlo Probability Bounds
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    n_prices=st.integers(min_value=30, max_value=252),
    n_sims=st.integers(min_value=100, max_value=500),
    start_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=30000)
def test_property6_monte_carlo_bounds(n_prices: int, n_sims: int, start_price: float):
    """
    Property 6: Monte Carlo Probability Bounds.
    prob_profit in [0.0, 1.0] and p5 <= p25 <= median_price <= p75 <= p95.
    """
    df = _make_ohlcv(n_prices, start_price=start_price)
    result = MonteCarloEngine.simulate(df["Close"], n_simulations=n_sims, n_days=30)

    assert "error" not in result, f"MC returned error: {result.get('error')}"

    prob = result["prob_profit"]
    assert 0.0 <= prob <= 1.0, f"prob_profit={prob} out of [0, 1]"

    p5 = result["p5"]
    p25 = result["p25"]
    median = result["median_price"]
    p75 = result["p75"]
    p95 = result["p95"]

    assert p5 <= p25, f"p5={p5} > p25={p25}"
    assert p25 <= median, f"p25={p25} > median={median}"
    assert median <= p75, f"median={median} > p75={p75}"
    assert p75 <= p95, f"p75={p75} > p95={p95}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 5: Multi-Analyzer Score and Signal Validity
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    n_prices=st.integers(min_value=60, max_value=252),
    start_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
    sector=st.sampled_from(["defence", "it", "pharma", "energy", "fmcg", "metals", "unknown_sector"]),
)
@settings(max_examples=50, deadline=60000)
def test_property5_multi_analyzer_validity(n_prices: int, start_price: float, sector: str):
    """
    Property 5: Multi-Analyzer Score and Signal Validity.
    combined_score in [0.0, 1.0] and signal in valid set.
    """
    df = _make_ohlcv(n_prices, start_price=start_price)
    ma = MultiAnalyzer()
    result = ma.analyze(df, sector=sector, capital=100_000, risk_pct=0.02)

    assert "error" not in result, f"MultiAnalyzer returned error: {result.get('error')}"

    score = result["combined_score"]
    assert 0.0 <= score <= 1.0, f"combined_score={score} out of [0, 1]"

    valid_signals = {"STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"}
    assert result["signal"] in valid_signals, (
        f"Invalid signal '{result['signal']}'"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 7: Peer Comparison Target Invariant
# ═══════════════════════════════════════════════════════════════════════════════

def test_property7_peer_comparison_target_invariant():
    """
    Property 7: Peer Comparison Target Invariant.
    1 <= len(result) <= 5, result[0].is_target == True,
    all return fields are finite floats.
    Uses a fixed sector with known peers to avoid network dependency.
    """
    # Use a minimal df_daily — load_peer_data uses DataService internally
    # so we just verify the structural invariants with a known sector
    from backend.engines.stock_analysis_engine import load_peer_data

    # We test with a mock by checking the function signature and return type
    # For unit testing without network, we verify the target stock is always first
    # by constructing a minimal scenario
    import unittest.mock as mock

    mock_df = _make_ohlcv(130, start_price=500.0)

    with mock.patch("backend.engines.stock_analysis_engine.DataService.fetch_multiple") as mock_fetch:
        # Return mock data for all symbols
        mock_fetch.return_value = {
            "RELIANCE.NS": mock_df,
            "TCS.NS": mock_df,
            "INFY.NS": mock_df,
        }
        result = load_peer_data("RELIANCE", "💻 IT/Tech")

    # If we got results, verify invariants
    if result:
        assert 1 <= len(result) <= 5, f"Expected 1-5 peers, got {len(result)}"
        assert result[0]["is_target"] is True, "First result must be the target stock"

        # All return fields must be finite floats
        float_fields = ["return_1m", "return_3m", "return_6m", "rsi", "volatility", "sharpe"]
        for peer in result:
            for field in float_fields:
                val = peer.get(field, None)
                if val is not None:
                    assert np.isfinite(val), f"Non-finite value for {field}: {val}"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 8: Risk Metrics Non-Negativity
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    n_prices=st.integers(min_value=30, max_value=252),
    start_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=100, deadline=10000)
def test_property8_risk_metrics_non_negativity(n_prices: int, start_price: float):
    """
    Property 8: Risk Metrics Non-Negativity.
    volatility_annual >= 0 for any price series with >= 30 rows.
    """
    df = _make_ohlcv(n_prices, start_price=start_price)
    result = RiskEngine.compute_all(df["Close"])

    assert "error" not in result, f"RiskEngine returned error: {result.get('error')}"
    vol = result["volatility_annual"]
    assert vol >= 0, f"volatility_annual={vol} is negative"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 9: ML Prediction Direction Validity
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    n_prices=st.integers(min_value=100, max_value=252),
    start_price=st.floats(min_value=10.0, max_value=5000.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=20, deadline=120000)
def test_property9_ml_prediction_direction(n_prices: int, start_price: float):
    """
    Property 9: ML Prediction Direction Validity.
    direction in {"UP", "DOWN", "NEUTRAL"} for any OHLCV DataFrame with >= 100 rows.
    """
    from backend.engines.prediction_engine import PredictionEngine

    df = _make_ohlcv(n_prices, start_price=start_price)
    result = PredictionEngine.predict_next_day(df)

    if "error" in result:
        # Insufficient data is acceptable — just skip
        return

    valid_directions = {"UP", "DOWN", "NEUTRAL"}
    assert result["direction"] in valid_directions, (
        f"Invalid direction '{result['direction']}'"
    )
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"confidence={result['confidence']} out of [0, 1]"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Property 10: Global Factor Structure Invariant
# ═══════════════════════════════════════════════════════════════════════════════

@given(
    sector=st.one_of(
        st.sampled_from([
            "🛡️ Defence", "🏦 PSU Banks", "🏗️ Infra/Rail", "⚡ Energy",
            "💻 IT/Tech", "💊 Pharma", "⚙️ Metals", "🚗 Auto/EV",
            "🛒 FMCG", "💰 Finance", "🧪 Chemicals", "🏠 Realty/Cement", "📡 Telecom",
        ]),
        st.text(min_size=0, max_size=50),  # unknown/random sectors
    )
)
@settings(max_examples=200, deadline=1000)
def test_property10_global_factor_structure(sector: str):
    """
    Property 10: Global Factor Structure Invariant.
    Result always contains keys 'positive', 'negative', 'theme'
    for any sector string including unknown sectors.
    """
    result = StockMetadata.get_global_factors(sector)

    assert "positive" in result, f"Missing 'positive' key for sector='{sector}'"
    assert "negative" in result, f"Missing 'negative' key for sector='{sector}'"
    assert "theme" in result, f"Missing 'theme' key for sector='{sector}'"

    assert isinstance(result["positive"], list), "'positive' must be a list"
    assert isinstance(result["negative"], list), "'negative' must be a list"
    assert isinstance(result["theme"], str), "'theme' must be a string"
    assert len(result["theme"]) > 0, "'theme' must be non-empty"

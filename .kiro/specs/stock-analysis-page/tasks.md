# Implementation Plan: Stock Analysis Page (🔍 Deep Dive Tab)

## Overview

Add a "🔍 Deep Dive" tab to `live_trader.py` that aggregates every available backend engine into a single scrollable Streamlit view. The implementation is split into: extending the `AnalysisBundle` dataclass and `load_analysis_bundle()` in `backend/engines/stock_analysis_engine.py`, adding the ten panel renderer functions, wiring the tab into `live_trader.py`, and writing property-based tests with `hypothesis`.

---

## Tasks

- [-] 1. Extend AnalysisBundle dataclass and load_analysis_bundle() in stock_analysis_engine.py
  - Add `df_weekly`, `price_change_1d_abs`, and `yf_symbol` fields to the `AnalysisBundle` dataclass in `backend/engines/stock_analysis_engine.py`
  - Update `load_analysis_bundle()` to fetch `df_weekly` (5y weekly OHLCV) in the parallel data-fetch block alongside `df_daily` and `df_intra`
  - Wrap each of the six engine calls in `_safe()` so that a failing engine stores `None` in the bundle instead of raising
  - Store `price_change_1d_abs` (absolute ₹ change) alongside the existing `price_change_1d` (% change)
  - Cache the completed bundle in `st.session_state["sap_{symbol}"]`; return cached value on repeat calls without re-fetching
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 13.2_

  - [-] 1.1 Write property test for ATR floor (Property 3)
    - **Property 3: ATR Floor Prevents Zero Division**
    - Use `hypothesis` strategies to generate price series where ATR may be zero or NaN; assert `atr >= price * 0.005` always holds after `compute_trade_setup`
    - **Validates: Requirements 11.2, 13.4**

- [~] 2. Implement compute_trade_setup() and build_timeframe_signals() pure functions
  - Verify `compute_trade_setup()` already in `stock_analysis_engine.py` satisfies the ATR floor, ordering invariant, and arithmetic invariant; patch any gaps
  - Implement `build_timeframe_signals(symbol, df_intra, df_daily, df_weekly) -> list[TimeframeSignal]` in `stock_analysis_engine.py` following the pseudocode in the design: 5m signal from `IntradayEngine.score_stock`, 1d/1w signals from `FeatureEngine` features with the score formula `s = 0.5 ± adjustments`; skip timeframes with fewer than 30 rows
  - Add `TimeframeSignal` dataclass to `stock_analysis_engine.py` if not already present
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 11.1, 11.2, 11.7, 11.8, 11.9, 11.10_

  - [~] 2.1 Write property test for Trade Setup Ordering Invariant (Property 1)
    - **Property 1: Trade Setup Ordering Invariant**
    - Use `hypothesis` to generate `(price, capital, risk_pct)` combos; assert `stop_loss < entry < target_1 < target_2` for all valid inputs
    - **Validates: Requirements 11.1, 11.7**

  - [~] 2.2 Write property test for Trade Setup Arithmetic Invariant (Property 2)
    - **Property 2: Trade Setup Arithmetic Invariant**
    - Assert `invested == qty * entry` (within float precision), `qty >= 0`, and `risk_reward > 0` when `risk_per_share > 0`
    - **Validates: Requirements 11.3, 11.8, 11.9, 11.10**

  - [~] 2.3 Write property test for Timeframe Signal Validity (Property 4)
    - **Property 4: Timeframe Signal Validity**
    - Use `hypothesis` DataFrames strategy; assert `1 <= len(result) <= 3`, all scores in `[0.0, 1.0]`, all signal strings in `{"BUY", "SELL", "NEUTRAL"}`
    - **Validates: Requirements 5.2, 5.3, 5.4**

- [~] 3. Checkpoint — core data layer complete
  - Ensure all tests pass, ask the user if questions arise.

- [~] 4. Implement render_price_overview() panel
  - Create `render_price_overview(bundle: AnalysisBundle) -> None` in `stock_analysis_engine.py` (or a new `frontend/deep_dive.py`)
  - Display current price, 1d absolute change, 1d % change, 52w high/low using `st.metric` and custom HTML badges
  - Call `StockMetadata.classify_price(price)` for cap badge and `StockMetadata.get_risk_level(price, vol, rsi)` for risk badge
  - Display sector badge from `SYMBOL_TO_SECTOR`
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [~] 5. Implement render_technical_chart() panel
  - Create `render_technical_chart(df_daily, df_features, df_intra, timeframe, trade_setup) -> None`
  - Build a 4-row Plotly `make_subplots`: row 1 candlestick + overlays (EMA9, EMA21, EMA50, EMA200, BB_Upper, BB_Lower, VWAP for intraday), row 2 RSI, row 3 MACD histogram, row 4 volume bars
  - Add a radio toggle `["Daily", "Intraday 5m"]` that switches the data source without re-fetching
  - Draw horizontal `hline` for `entry`, `target_1`, `stop_loss` when `trade_setup` is not None
  - Set chart height to 600px and `xaxis_rangeslider_visible = False`
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_

- [~] 6. Implement render_multi_timeframe() panel
  - Create `render_multi_timeframe(symbol, df_intra, df_daily, df_weekly) -> None`
  - Call `build_timeframe_signals()` and render one `st.column` card per returned `TimeframeSignal`
  - Each card shows: timeframe label, signal badge (BUY/SELL/NEUTRAL coloured), score %, RSI, MACD bullish flag, above-EMA flag, and reasons list
  - Omit cards for timeframes with insufficient data (< 30 rows) without raising
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

- [~] 7. Implement render_ml_predictions() panel
  - Create `render_ml_predictions(df_daily, current_price) -> None`
  - Call `PredictionEngine.predict_next_day(df_daily)` and `PredictionEngine.predict_multi_horizon(df_daily)`; wrap both in try/except
  - Render three prediction cards (next-day, 10-day, 30-day): direction badge, `st.progress` confidence bar, predicted price, price range (low–high)
  - Display market conditions list from `predict_next_day` result
  - On exception, display `st.warning("ML predictions unavailable — insufficient history")` and return
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

  - [~] 7.1 Write property test for ML Prediction Direction Validity (Property 9)
    - **Property 9: ML Prediction Direction Validity**
    - Assert `direction in {"UP", "DOWN", "NEUTRAL"}` for any OHLCV DataFrame with >= 100 rows
    - **Validates: Requirements 6.4**

- [~] 8. Implement render_risk_metrics() panel
  - Create `render_risk_metrics(df_daily, risk_free_rate=0.065) -> None`
  - Call `RiskEngine.compute_all(df_daily["Close"])`; if result contains `"error"` key, show `st.warning` and return
  - Display VaR 95%, CVaR, Sharpe, Sortino, Max Drawdown, Calmar, Annual Volatility, Total Return using `st.metric`
  - Colour-code Sharpe: green > 1, yellow 0–1, red < 0 (use inline HTML or `st.markdown`)
  - Call `RiskEngine.stress_test(returns)` and render the stress scenario table via `st.dataframe`
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_

  - [~] 8.1 Write property test for Risk Metrics Non-Negativity (Property 8)
    - **Property 8: Risk Metrics Non-Negativity**
    - Assert `volatility_annual >= 0` for any price series with >= 30 rows
    - **Validates: Requirements 7.4**

- [~] 9. Implement render_monte_carlo() panel
  - Create `render_monte_carlo(df_daily, n_simulations=5000, n_days=30) -> None`
  - Call `MonteCarloEngine.simulate(df_daily["Close"], n_simulations, n_days)`; if result contains `"error"` key, show `st.warning` and return
  - Build a Plotly fan chart: plot 50 sample paths as thin grey lines, then filled bands for p5–p95, p25–p75, and a bold median line
  - Display `prob_profit`, `expected_price`, `p5`, `p95` as `st.metric` below the chart
  - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [~] 9.1 Write property test for Monte Carlo Probability Bounds (Property 6)
    - **Property 6: Monte Carlo Probability Bounds**
    - Assert `prob_profit in [0.0, 1.0]` and `p5 <= p25 <= median_price <= p75 <= p95` for any price series with >= 30 rows and n_simulations >= 100
    - **Validates: Requirements 8.3, 8.4**

- [~] 10. Implement render_multi_analyzer_scorecard() panel
  - Create `render_multi_analyzer_scorecard(df_daily, sector, capital, risk_pct) -> None`
  - Call `MultiAnalyzer().analyze(df_daily, sector, capital, risk_pct)`
  - Build a Plotly radar chart with axes: Technical, Momentum, Volume, Global Macro, Geopolitical
  - Render per-analyzer score cards (one per column) each with signal badge and a `st.expander` for detail reasoning
  - Display combined signal badge (STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL) prominently above the radar chart
  - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5_

  - [~] 10.1 Write property test for Multi-Analyzer Score and Signal Validity (Property 5)
    - **Property 5: Multi-Analyzer Score and Signal Validity**
    - Assert `combined_score in [0.0, 1.0]` and `signal in {"STRONG BUY", "BUY", "NEUTRAL", "SELL", "STRONG SELL"}` for any valid OHLCV input
    - **Validates: Requirements 9.4, 9.5**

- [~] 11. Implement render_global_factors() panel
  - Create `render_global_factors(sector) -> None`
  - Call `StockMetadata.get_global_factors(sector)`; handle missing sector gracefully (default factors, no exception)
  - Render positive factors as green `.factor-pos` chips and negative factors as red `.factor-neg` chips using existing CSS classes in `live_trader.py`
  - Display theme badge (STRONG TAILWIND / MODERATE TAILWIND / NEUTRAL)
  - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [~] 11.1 Write property test for Global Factor Structure Invariant (Property 10)
    - **Property 10: Global Factor Structure Invariant**
    - Assert result always contains keys `"positive"`, `"negative"`, `"theme"` for any sector string including unknown sectors
    - **Validates: Requirements 10.3, 10.4**

- [~] 12. Implement render_trade_setup() panel
  - Create `render_trade_setup(bundle: AnalysisBundle, capital: float, risk_pct: float) -> None`
  - Display entry, target_1, target_2, stop_loss, qty, invested, max_profit, max_loss, risk_reward using `st.metric`
  - Add a capital `st.slider` that calls `compute_trade_setup()` with the new capital value and re-renders without re-fetching data
  - Add a copy-to-clipboard button using `st.code` + `st.button` pattern (write trade details as a formatted string into a `st.code` block)
  - Show `st.info` when ATR floor was applied (stock potentially illiquid)
  - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6, 13.4_

- [~] 13. Implement load_peer_data() and render_peer_comparison() panel
  - Verify `load_peer_data(symbol, sector, df_daily)` in `stock_analysis_engine.py` returns target stock first with `is_target=True`, up to 4 peers, all return fields as finite floats
  - Create `render_peer_comparison(symbol, sector, df_daily, peer_symbols) -> None`
  - Render a normalised price chart (rebased to 100) for all peers using Plotly line chart
  - Render a metrics table (symbol, price, 1m%, 3m%, 6m%, RSI, vol%, Sharpe) via `st.dataframe`; highlight the target row
  - When all peers return empty DataFrames, render only the target row and display `st.info("No peer data available for this sector")`
  - _Requirements: 12.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8, 13.5_

  - [~] 13.1 Write property test for Peer Comparison Target Invariant (Property 7)
    - **Property 7: Peer Comparison Target Invariant**
    - Assert `1 <= len(result) <= 5`, `result[0].is_target == True`, and all return fields are finite floats
    - **Validates: Requirements 12.5, 12.6, 12.7**

- [~] 14. Checkpoint — all panels implemented
  - Ensure all tests pass, ask the user if questions arise.

- [~] 15. Implement render_stock_analysis_tab() orchestrator and wire into live_trader.py
  - Create `render_stock_analysis_tab(capital, risk_pct, symbol_list) -> None` in `stock_analysis_engine.py`
  - Add a searchable `st.selectbox` over `symbol_list`; validate input against `ALL_SYMBOLS_CLEAN` and show `st.error` for unknown symbols without triggering data loading
  - Add capital input and risk % slider pre-filled from sidebar values
  - On "🔍 Analyse" button click: check `st.session_state["sap_{symbol}"]` for cached bundle; if miss, call `load_analysis_bundle()` wrapped in try/except — show `st.error` on `ValueError`, show `st.warning` on any other exception
  - Call all ten panel renderers in order, each wrapped in its own try/except that shows `st.warning` and continues
  - In `live_trader.py`, extend the `st.tabs(...)` call to include `"🔍 Deep Dive"` as `tab7`, and add `with tab7: render_stock_analysis_tab(capital=capital, risk_pct=0.02, symbol_list=ALL_SYMBOLS_CLEAN)`
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 2.4, 2.5, 13.1, 13.3_

- [~] 16. Final checkpoint — full integration
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Property tests use the `hypothesis` library (`pip install hypothesis`)
- All panel renderers should be co-located in `backend/engines/stock_analysis_engine.py` or a new `frontend/deep_dive.py`; either is acceptable as long as the import in `live_trader.py` resolves
- The existing CSS classes in `live_trader.py` (`.factor-pos`, `.factor-neg`, `.badge-buy`, `.badge-sell`, `.badge-watch`, `.stat-card`) should be reused by all new panels
- Session state key format: `"sap_{symbol}"` (e.g. `"sap_RELIANCE"`)
- Monte Carlo default: `n_simulations=5000`, `n_days=30` per performance considerations in the design

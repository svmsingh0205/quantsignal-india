# Requirements Document

## Introduction

This document specifies the requirements for the **Stock Analysis Page** feature of QuantSignal India. The feature adds a "🔍 Deep Dive" tab to `live_trader.py` that aggregates every available backend engine into a single, scrollable Streamlit view. A user selects any NSE symbol and immediately sees ten analysis panels — price overview, technical chart, multi-timeframe signals, ML predictions, risk metrics, Monte Carlo simulation, multi-analyzer scorecard, global factor impact, trade setup, and peer comparison — all without leaving the page.

---

## Glossary

- **StockAnalysisPage**: The top-level Streamlit component that owns the symbol selector, triggers data loading, and delegates rendering to each panel function. Implemented as the "🔍 Deep Dive" tab in `live_trader.py`.
- **DataLoader**: The `load_analysis_bundle()` function responsible for fetching all raw data and running all backend engines in parallel.
- **AnalysisBundle**: A typed dataclass that holds all raw DataFrames and engine output dicts for a single symbol analysis session.
- **TradeSetup**: A typed dataclass holding ATR-based entry, target, stop-loss, position size, and risk/reward metrics.
- **TimeframeSignal**: A typed dataclass holding the signal classification (BUY/SELL/NEUTRAL), score, and key indicators for a single timeframe (5m, 1d, 1wk).
- **PeerMetrics**: A typed dataclass holding price and return metrics for a single stock in the peer comparison panel.
- **DataService**: The existing `backend/engines/data_service.py` class that fetches and caches OHLCV data from yfinance.
- **IntradayEngine**: The existing `backend/engines/intraday_engine.py` class that fetches 5-minute intraday data and computes intraday indicators.
- **FeatureEngine**: The existing `backend/engines/feature_engine.py` class that computes technical indicators and ML features on daily OHLCV data.
- **PredictionEngine**: The existing `backend/engines/prediction_engine.py` class that produces next-day and multi-horizon ML price forecasts.
- **RiskEngine**: The existing `backend/engines/risk_engine.py` class that computes VaR, CVaR, Sharpe, Sortino, drawdown, and stress-test metrics.
- **MonteCarloEngine**: The existing `backend/engines/monte_carlo.py` class that runs GBM-based price simulations.
- **MultiAnalyzer**: The existing `backend/engines/multi_analyzer.py` class that combines Technical, Momentum, Volume, Global Macro, and Geopolitical analyzer scores.
- **StockMetadata**: The existing `backend/engines/stock_metadata.py` class that provides cap classification, risk level, holding duration, and global factor data.
- **ALL_SYMBOLS_CLEAN**: The deduplicated list of clean NSE tickers (without `.NS` suffix) derived from `INTRADAY_STOCKS` in `backend/intraday_config.py`.
- **SECTOR_GROUPS**: The sector-to-symbol mapping dict defined in `backend/intraday_config.py`.
- **ATR**: Average True Range — a 14-period volatility measure used to compute trade setup levels.
- **session_state**: Streamlit's `st.session_state` dict used to cache `AnalysisBundle` objects between reruns.

---

## Requirements

### Requirement 1: Deep Dive Tab Integration

**User Story:** As a trader, I want a dedicated Deep Dive tab in the existing QuantSignal India app, so that I can perform a comprehensive single-stock analysis without leaving the platform.

#### Acceptance Criteria

1. THE StockAnalysisPage SHALL be rendered as a tab labelled "🔍 Deep Dive" appended to the existing tab list in `live_trader.py`.
2. THE StockAnalysisPage SHALL accept `capital`, `risk_pct`, and `symbol_list` as inputs pre-populated from the existing sidebar values.
3. WHEN the StockAnalysisPage renders, THE StockAnalysisPage SHALL display a searchable symbol selectbox populated from ALL_SYMBOLS_CLEAN.
4. WHEN a symbol is submitted that is not present in ALL_SYMBOLS_CLEAN, THE StockAnalysisPage SHALL reject the input and display an error message without triggering data loading.
5. IF any engine raises an exception during analysis, THEN THE StockAnalysisPage SHALL display an `st.warning` message for the affected panel and continue rendering all other panels normally.
6. THE StockAnalysisPage SHALL not propagate any engine exception to the top-level Streamlit error handler.

---

### Requirement 2: Data Loading and Session State Caching

**User Story:** As a trader, I want analysis results to load quickly and be cached, so that switching between panels does not re-fetch data on every interaction.

#### Acceptance Criteria

1. WHEN the user clicks the Analyse button, THE DataLoader SHALL fetch `df_daily` (2-year daily OHLCV), `df_intra` (5-day 5-minute OHLCV), and `df_weekly` (5-year weekly OHLCV) in parallel using `ThreadPoolExecutor`.
2. WHEN `DataService.fetch_ohlcv` returns a DataFrame with fewer than 60 rows, THE DataLoader SHALL raise a `ValueError` and display `st.error` with a message identifying the symbol.
3. WHEN a network timeout causes `DataService.fetch_ohlcv` to return an empty DataFrame, THE DataLoader SHALL raise a `ValueError` and leave `session_state` unchanged.
4. WHEN the DataLoader completes successfully, THE DataLoader SHALL store the `AnalysisBundle` in `st.session_state` under the key `"sap_{symbol}"`.
5. WHEN the user clicks Analyse for a symbol already present in `session_state`, THE StockAnalysisPage SHALL return the cached `AnalysisBundle` without re-fetching data from yfinance.
6. THE DataLoader SHALL run all six engine calls (IntradayEngine.score_stock, PredictionEngine.predict_next_day, PredictionEngine.predict_multi_horizon, RiskEngine.compute_all, MonteCarloEngine.simulate, MultiAnalyzer.analyze) in parallel using `ThreadPoolExecutor`.

---

### Requirement 3: Price Overview Panel

**User Story:** As a trader, I want to see a concise price summary at the top of the analysis page, so that I can immediately assess the stock's current state and classification.

#### Acceptance Criteria

1. THE PriceOverviewPanel SHALL display the current price, 1-day absolute change, and 1-day percentage change.
2. THE PriceOverviewPanel SHALL display the 52-week high and 52-week low.
3. THE PriceOverviewPanel SHALL display the cap classification badge produced by `StockMetadata.classify_price(price)`.
4. THE PriceOverviewPanel SHALL display the risk level badge produced by `StockMetadata.get_risk_level(price, volatility, rsi)`.
5. THE PriceOverviewPanel SHALL display the sector badge derived from `SYMBOL_TO_SECTOR`.

---

### Requirement 4: Technical Chart Panel

**User Story:** As a trader, I want a full candlestick chart with technical overlays, so that I can visually assess price action and indicator alignment.

#### Acceptance Criteria

1. THE TechnicalChartPanel SHALL render a 4-row Plotly subplot containing: candlestick with overlays (row 1), RSI (row 2), MACD histogram (row 3), and volume bars (row 4).
2. THE TechnicalChartPanel SHALL overlay EMA9, EMA21, EMA50, EMA200, Bollinger Bands upper and lower, and VWAP (intraday only) on the candlestick row.
3. THE TechnicalChartPanel SHALL provide a timeframe toggle allowing the user to switch between "Daily" (df_daily) and "Intraday 5m" (df_intra) data.
4. WHEN a trade setup exists in the AnalysisBundle, THE TechnicalChartPanel SHALL render horizontal lines for entry, target_1, and stop_loss.
5. THE TechnicalChartPanel SHALL render the chart at a height of 600 pixels with `xaxis_rangeslider_visible = False`.

---

### Requirement 5: Multi-Timeframe Signals Panel

**User Story:** As a trader, I want to see signal summaries across multiple timeframes, so that I can confirm trade direction alignment before entering a position.

#### Acceptance Criteria

1. THE MultiTimeframePanel SHALL display signal cards for three timeframes: "5m Intraday", "1d Swing", and "1w Positional".
2. WHEN `build_timeframe_signals` is called with at least one non-empty DataFrame, THE MultiTimeframePanel SHALL return between 1 and 3 `TimeframeSignal` objects.
3. FOR ALL `TimeframeSignal` objects returned by `build_timeframe_signals`, THE score SHALL be in the range [0.0, 1.0].
4. FOR ALL `TimeframeSignal` objects returned by `build_timeframe_signals`, THE signal field SHALL be one of "BUY", "SELL", or "NEUTRAL".
5. THE MultiTimeframePanel SHALL derive the 5m signal from `IntradayEngine.score_stock(df_intra)`, the 1d signal from `FeatureEngine` features on `df_daily`, and the 1w signal from `FeatureEngine` features on `df_weekly`.
6. WHEN a timeframe DataFrame is empty or has fewer than 30 rows, THE MultiTimeframePanel SHALL omit that timeframe's signal card without raising an exception.

---

### Requirement 6: ML Predictions Panel

**User Story:** As a trader, I want to see ML-based price forecasts for multiple horizons, so that I can gauge the expected direction and magnitude of price movement.

#### Acceptance Criteria

1. THE MLPredictionPanel SHALL display prediction cards for next-day, 10-day, and 30-day horizons using `PredictionEngine.predict_next_day` and `PredictionEngine.predict_multi_horizon`.
2. THE MLPredictionPanel SHALL display direction badge, confidence bar, predicted price, and price range (low–high) for each horizon.
3. THE MLPredictionPanel SHALL display the market conditions list from `PredictionEngine.predict_next_day`.
4. FOR ALL valid OHLCV inputs with at least 100 rows, THE PredictionEngine.predict_next_day result SHALL have a direction field with value "UP", "DOWN", or "NEUTRAL".
5. WHEN `PredictionEngine` raises an exception, THE MLPredictionPanel SHALL display `st.warning("ML predictions unavailable — insufficient history")` and continue rendering other panels.

---

### Requirement 7: Risk Metrics Panel

**User Story:** As a trader, I want to see quantitative risk metrics with colour-coded thresholds, so that I can assess the risk-adjusted return profile of the stock.

#### Acceptance Criteria

1. THE RiskMetricsPanel SHALL display VaR 95%, CVaR, Sharpe ratio, Sortino ratio, Max Drawdown, Calmar ratio, Annual Volatility, and Total Return from `RiskEngine.compute_all`.
2. THE RiskMetricsPanel SHALL colour-code the Sharpe ratio: green for Sharpe > 1, yellow for 0–1, red for < 0.
3. THE RiskMetricsPanel SHALL display the stress scenario table from `RiskEngine.stress_test(returns)`.
4. FOR ALL price series with at least 30 rows, THE RiskEngine.compute_all result SHALL have a `volatility_annual` value that is greater than or equal to 0.
5. WHEN `RiskEngine.compute_all` returns an `"error"` key, THE RiskMetricsPanel SHALL display `st.warning` and skip rendering the metrics table.

---

### Requirement 8: Monte Carlo Simulation Panel

**User Story:** As a trader, I want to see a 30-day Monte Carlo simulation with probability statistics, so that I can understand the distribution of possible future prices.

#### Acceptance Criteria

1. THE MonteCarloPanel SHALL call `MonteCarloEngine.simulate(df_daily["Close"], n_simulations, n_days)` and render a fan chart showing p5, p25, median, p75, p95 bands plus 50 sample paths.
2. THE MonteCarloPanel SHALL display prob_profit, expected_price, p5, and p95 as summary metrics below the chart.
3. FOR ALL price series with at least 30 rows and n_simulations >= 100, THE MonteCarloEngine.simulate result SHALL have `prob_profit` in the range [0.0, 1.0].
4. FOR ALL price series with at least 30 rows, THE MonteCarloEngine.simulate result SHALL satisfy `p5 <= p25 <= median_price <= p75 <= p95`.
5. WHEN `MonteCarloEngine.simulate` returns an `"error"` key, THE MonteCarloPanel SHALL display `st.warning` and skip rendering the chart.

---

### Requirement 9: Multi-Analyzer Scorecard Panel

**User Story:** As a trader, I want to see a radar chart and per-analyzer breakdown, so that I can understand which factors are driving the combined signal.

#### Acceptance Criteria

1. THE MultiAnalyzerScorecardPanel SHALL call `MultiAnalyzer().analyze(df_daily, sector, capital, risk_pct)` and render a radar chart with axes for Technical, Momentum, Volume, Global Macro, and Geopolitical scores.
2. THE MultiAnalyzerScorecardPanel SHALL display per-analyzer score cards with signal badge and a detail expander.
3. THE MultiAnalyzerScorecardPanel SHALL display the combined signal badge (STRONG BUY / BUY / NEUTRAL / SELL / STRONG SELL).
4. FOR ALL valid OHLCV inputs, THE MultiAnalyzer.analyze result SHALL have `combined_score` in the range [0.0, 1.0].
5. FOR ALL valid OHLCV inputs, THE MultiAnalyzer.analyze result SHALL have `signal` equal to one of "STRONG BUY", "BUY", "NEUTRAL", "SELL", or "STRONG SELL".

---

### Requirement 10: Global Factor Impact Panel

**User Story:** As a trader, I want to see sector-specific geopolitical and macro themes, so that I can understand the macro tailwinds and headwinds affecting the stock.

#### Acceptance Criteria

1. THE GlobalFactorPanel SHALL call `StockMetadata.get_global_factors(sector)` and display positive factors as green chips and negative factors as red chips.
2. THE GlobalFactorPanel SHALL display the theme badge (e.g. "STRONG TAILWIND", "MODERATE TAILWIND", "NEUTRAL").
3. FOR ALL sector strings passed to `StockMetadata.get_global_factors`, THE result SHALL contain the keys "positive", "negative", and "theme".
4. WHEN the sector is not found in `GLOBAL_FACTORS`, THE GlobalFactorPanel SHALL display default factors without raising an exception.

---

### Requirement 11: Trade Setup Generator Panel

**User Story:** As a trader, I want an auto-generated ATR-based trade setup with position sizing, so that I can immediately act on the analysis with a concrete entry, target, and stop-loss plan.

#### Acceptance Criteria

1. THE TradeSetupPanel SHALL compute entry, target_1, target_2, and stop_loss using the ATR formula: `stop_loss = max(entry - 1.5 * ATR, entry * 0.93)`, `target_1 = entry + 2.0 * ATR`, `target_2 = entry + 3.0 * ATR`.
2. WHEN ATR is zero or NaN, THE compute_trade_setup function SHALL apply a floor of `price * 0.005` before computing trade levels.
3. THE TradeSetupPanel SHALL compute position quantity as `floor(capital * risk_pct / risk_per_share)`.
4. THE TradeSetupPanel SHALL display entry, target_1, target_2, stop_loss, quantity, invested amount, max profit, max loss, and risk-reward ratio.
5. THE TradeSetupPanel SHALL provide a capital slider that recalculates position sizing without re-fetching data.
6. THE TradeSetupPanel SHALL provide a copy-to-clipboard button for the trade details.
7. FOR ALL valid inputs where `price > 0` and `len(df_daily) >= 14` and `capital > 0` and `0 < risk_pct <= 0.10`, THE TradeSetup SHALL satisfy `stop_loss < entry < target_1 < target_2`.
8. FOR ALL valid inputs where `risk_per_share > 0`, THE TradeSetup SHALL satisfy `risk_reward > 0`.
9. THE TradeSetup SHALL satisfy `invested == qty * entry` within floating-point precision.
10. THE TradeSetup SHALL satisfy `qty >= 0` for all valid inputs.

---

### Requirement 12: Peer Comparison Panel

**User Story:** As a trader, I want to compare the analysed stock against its sector peers, so that I can assess relative performance and identify the strongest stock in the sector.

#### Acceptance Criteria

1. THE PeerComparisonPanel SHALL auto-select up to 4 peer symbols from `SECTOR_GROUPS[sector]`, excluding the target symbol.
2. THE PeerComparisonPanel SHALL fetch peer data via `DataService.fetch_multiple(peers, "6mo")`.
3. THE PeerComparisonPanel SHALL display a metrics table with columns: symbol, current price, 1-month return, 3-month return, 6-month return, RSI, annualised volatility, and Sharpe ratio.
4. THE PeerComparisonPanel SHALL display a normalised price chart rebased to 100 for visual comparison.
5. WHEN `load_peer_data` is called, THE result SHALL contain the target stock as the first element with `is_target == True`.
6. WHEN `load_peer_data` is called, THE result SHALL contain between 1 and 5 `PeerMetrics` objects.
7. FOR ALL `PeerMetrics` objects, THE `return_1m`, `return_3m`, and `return_6m` fields SHALL be finite float values.
8. WHEN all peer symbols return empty DataFrames, THE PeerComparisonPanel SHALL render with only the target stock's metrics and display a note indicating no peer data is available.

---

### Requirement 13: Error Handling

**User Story:** As a trader, I want the analysis page to handle data and engine failures gracefully, so that a single failure does not prevent me from seeing the rest of the analysis.

#### Acceptance Criteria

1. WHEN `DataService.fetch_ohlcv` returns an empty DataFrame, THE DataLoader SHALL raise `ValueError` and THE StockAnalysisPage SHALL display `st.error` with a retry prompt.
2. WHEN any individual engine call raises an exception, THE DataLoader SHALL catch the exception, store `None` for that engine's output in the AnalysisBundle, and continue executing all other engine calls.
3. WHEN an engine output in the AnalysisBundle is `None`, THE corresponding panel renderer SHALL display `st.warning` and skip rendering that panel's content.
4. WHEN `compute_trade_setup` encounters an ATR of zero or NaN, THE function SHALL apply the floor `atr = max(atr, price * 0.005)` and display `st.info` flagging the stock as potentially illiquid.
5. WHEN all peer symbols return empty DataFrames, THE PeerComparisonPanel SHALL render with only the target stock's row and display a note: "No peer data available for this sector".

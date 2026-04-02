"""
Deep Dive Panel Renderers — QuantSignal India
All 10 panel functions for the 🔍 Deep Dive tab.
Reuses CSS classes from live_trader.py: .badge-buy, .badge-sell, .badge-watch,
.factor-pos, .factor-neg, .stat-card
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Optional

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engines.stock_analysis_engine import (
    AnalysisBundle, TradeSetup, TimeframeSignal,
    compute_trade_setup, build_timeframe_signals, load_peer_data,
)
from backend.engines.prediction_engine import PredictionEngine
from backend.engines.risk_engine import RiskEngine
from backend.engines.monte_carlo import MonteCarloEngine
from backend.engines.multi_analyzer import MultiAnalyzer
from backend.engines.stock_metadata import StockMetadata


# ── Shared helpers ────────────────────────────────────────────────────────────

def _badge(sig: str) -> str:
    s = sig.upper()
    if "BUY" in s:
        return f'<span class="badge-buy">{sig}</span>'
    if "SELL" in s or "AVOID" in s:
        return f'<span class="badge-sell">{sig}</span>'
    return f'<span class="badge-watch">{sig}</span>'


def _section(title: str) -> None:
    st.markdown(
        f'<div style="font-size:1rem;font-weight:800;color:#e2e8f0;'
        f'border-left:3px solid #2563eb;padding-left:10px;margin:18px 0 10px;">'
        f'{title}</div>',
        unsafe_allow_html=True,
    )


def _plotly_layout(title: str = "", height: int = 450) -> dict:
    return dict(
        template="plotly_dark", paper_bgcolor="#04080f", plot_bgcolor="#080f1e",
        height=height, margin=dict(t=35 if title else 10, b=10, l=50, r=20),
        font=dict(family="Inter", size=11, color="#94a3b8"),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor="#0a1628", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#0a1628", showgrid=True, zeroline=False),
        title=dict(text=title, font=dict(size=13, color="#e2e8f0")) if title else None,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 1: Price Overview
# ═══════════════════════════════════════════════════════════════════════════════

def render_price_overview(bundle: AnalysisBundle) -> None:
    """Section 1 — current price, 1d change, 52w range, cap/risk/sector badges."""
    _section("📊 Price Overview")

    price = bundle.current_price
    chg_pct = bundle.price_change_1d
    chg_abs = bundle.price_change_1d_abs
    high_52 = bundle.price_52w_high
    low_52 = bundle.price_52w_low

    vol = (bundle.risk_metrics or {}).get("volatility_annual", 0.25)
    rsi = (bundle.intra_score or {}).get("rsi", 50)

    cap_class = StockMetadata.classify_price(price)
    risk_level = StockMetadata.get_risk_level(price, vol, rsi)
    sector = bundle.sector

    # Cap badge colours
    cap_colors = {"PENNY": "#7c3aed", "SMALL": "#0891b2", "MID": "#059669", "LARGE": "#1d4ed8"}
    cap_color = cap_colors.get(cap_class, "#475569")

    chg_color = "#10b981" if chg_pct >= 0 else "#ef4444"
    chg_arrow = "▲" if chg_pct >= 0 else "▼"

    # 52w range bar
    range_span = high_52 - low_52
    pos_pct = int(((price - low_52) / range_span * 100)) if range_span > 0 else 50

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0a1628,#080f1e);border:1px solid #0f2a4a;
                border-radius:16px;padding:20px 24px;margin-bottom:12px;">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:12px;">
        <span style="font-size:1.8rem;font-weight:900;color:#f1f5f9;">{bundle.symbol}</span>
        <span style="font-size:1.5rem;font-weight:800;color:#f1f5f9;">₹{price:,.2f}</span>
        <span style="color:{chg_color};font-size:1rem;font-weight:700;">
          {chg_arrow} ₹{abs(chg_abs):,.2f} ({chg_pct:+.2f}%)
        </span>
      </div>
      <div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px;">
        <span style="background:{cap_color}22;color:{cap_color};padding:3px 12px;border-radius:6px;
                     font-size:.75rem;font-weight:700;border:1px solid {cap_color}44;">
          🏢 {cap_class} CAP
        </span>
        <span style="background:rgba(99,102,241,.12);color:#a5b4fc;padding:3px 12px;border-radius:6px;
                     font-size:.75rem;font-weight:700;border:1px solid rgba(99,102,241,.25);">
          {sector}
        </span>
        <span style="background:rgba(15,32,64,.6);color:#94a3b8;padding:3px 12px;border-radius:6px;
                     font-size:.75rem;font-weight:700;">
          {risk_level}
        </span>
      </div>
      <div style="margin-top:8px;">
        <div style="display:flex;justify-content:space-between;color:#475569;font-size:.7rem;margin-bottom:4px;">
          <span>52W Low ₹{low_52:,.0f}</span>
          <span>52W High ₹{high_52:,.0f}</span>
        </div>
        <div style="background:#0f2040;border-radius:4px;height:6px;position:relative;">
          <div style="background:linear-gradient(90deg,#1e40af,#2563eb);width:{pos_pct}%;
                      height:100%;border-radius:4px;"></div>
        </div>
        <div style="color:#475569;font-size:.68rem;margin-top:3px;">
          Current price at {pos_pct}% of 52-week range
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 2: Technical Chart
# ═══════════════════════════════════════════════════════════════════════════════

def render_technical_chart(
    df_daily: pd.DataFrame,
    df_features: pd.DataFrame,
    df_intra: pd.DataFrame,
    trade_setup: Optional[TradeSetup] = None,
) -> None:
    """Section 2 — 4-row Plotly chart: candles+overlays / RSI / MACD / Volume."""
    _section("📈 Technical Chart")

    timeframe = st.radio(
        "Timeframe", ["Daily", "Intraday 5m"],
        horizontal=True, key="dd_tf_radio",
        label_visibility="collapsed",
    )

    use_intra = timeframe == "Intraday 5m" and not df_intra.empty
    df = df_intra.tail(200) if use_intra else df_daily.tail(252)
    feat = df_features if not use_intra else df  # intraday already has indicators

    fig = make_subplots(
        rows=4, cols=1,
        row_heights=[0.55, 0.15, 0.15, 0.15],
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # Row 1: Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
        decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
        name="Price", showlegend=False,
    ), row=1, col=1)

    # Overlays
    overlay_map = [
        ("EMA9",    "#6366f1", "solid",  1.0),
        ("EMA21",   "#ec4899", "solid",  1.0),
        ("EMA50",   "#f59e0b", "dash",   1.0),
        ("EMA200",  "#06b6d4", "dot",    1.2),
        ("BB_Upper","#94a3b8", "dot",    0.8),
        ("BB_Lower","#94a3b8", "dot",    0.8),
        ("MA20",    "#f59e0b", "dash",   1.0),
        ("MA50",    "#f59e0b", "dash",   1.0),
    ]
    for col_name, color, dash, width in overlay_map:
        src = df if col_name in df.columns else feat
        if col_name in src.columns:
            fig.add_trace(go.Scatter(
                x=src.index, y=src[col_name], mode="lines",
                line=dict(color=color, width=width, dash=dash),
                name=col_name, showlegend=True,
            ), row=1, col=1)

    # VWAP (intraday only)
    if use_intra and "VWAP" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["VWAP"], mode="lines",
            line=dict(color="#fbbf24", width=1.5, dash="dot"),
            name="VWAP",
        ), row=1, col=1)

    # Trade setup lines
    if trade_setup:
        for val, color, label in [
            (trade_setup.entry,    "#6366f1", "Entry"),
            (trade_setup.target_1, "#10b981", "T1"),
            (trade_setup.stop_loss,"#ef4444", "SL"),
        ]:
            fig.add_hline(y=val, line_color=color, line_dash="dash",
                          line_width=1.2, annotation_text=label,
                          annotation_font_color=color, row=1, col=1)

    # Row 2: RSI
    rsi_col = "RSI" if "RSI" in df.columns else ("RSI" if "RSI" in feat.columns else None)
    rsi_src = df if (rsi_col and rsi_col in df.columns) else feat
    if rsi_col and rsi_col in rsi_src.columns:
        fig.add_trace(go.Scatter(
            x=rsi_src.index, y=rsi_src[rsi_col], mode="lines",
            line=dict(color="#a78bfa", width=1.5), name="RSI",
        ), row=2, col=1)
        fig.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=0.8, row=2, col=1)
        fig.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=0.8, row=2, col=1)

    # Row 3: MACD histogram
    macd_src = df if "MACD_Hist" in df.columns else feat
    if "MACD_Hist" in macd_src.columns:
        hist = macd_src["MACD_Hist"]
        fig.add_trace(go.Bar(
            x=macd_src.index, y=hist,
            marker_color=["#10b981" if v >= 0 else "#ef4444" for v in hist],
            name="MACD Hist", showlegend=False,
        ), row=3, col=1)

    # Row 4: Volume
    vcols = ["#10b981" if c >= o else "#ef4444"
             for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df.index, y=df["Volume"],
        marker_color=vcols, name="Volume", showlegend=False, opacity=0.7,
    ), row=4, col=1)

    layout = _plotly_layout(height=600)
    layout["xaxis_rangeslider_visible"] = False
    layout["yaxis2"] = dict(title="RSI", range=[0, 100], gridcolor="#0a1628")
    layout["yaxis3"] = dict(title="MACD", gridcolor="#0a1628")
    layout["yaxis4"] = dict(title="Vol", gridcolor="#0a1628")
    fig.update_layout(**layout)
    fig.update_xaxes(rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 3: Multi-Timeframe Signals
# ═══════════════════════════════════════════════════════════════════════════════

def render_multi_timeframe(
    symbol: str,
    df_intra: pd.DataFrame,
    df_daily: pd.DataFrame,
    df_weekly: pd.DataFrame,
) -> None:
    """Section 3 — signal cards for 5m / 1d / 1w timeframes."""
    _section("⏱️ Multi-Timeframe Signals")

    signals = build_timeframe_signals(symbol, df_intra, df_daily, df_weekly)

    if not signals:
        st.info("Insufficient data for timeframe signals.")
        return

    cols = st.columns(len(signals))
    for col, sig in zip(cols, signals):
        sig_color = "#10b981" if sig.signal == "BUY" else ("#ef4444" if sig.signal == "SELL" else "#f59e0b")
        macd_icon = "✅" if sig.macd_bullish else "❌"
        ema_icon = "✅" if sig.above_ema else "❌"
        with col:
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #0f2040;border-radius:12px;padding:14px;text-align:center;">
              <div style="color:#94a3b8;font-size:.72rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:6px;">{sig.timeframe}</div>
              <div style="background:{sig_color}22;color:{sig_color};padding:4px 14px;border-radius:7px;
                          font-weight:800;font-size:.9rem;display:inline-block;margin-bottom:8px;">
                {sig.signal} {int(sig.score * 100)}%
              </div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;margin-top:8px;text-align:left;">
                <div style="color:#475569;font-size:.7rem;">RSI</div>
                <div style="color:#a78bfa;font-size:.7rem;font-weight:700;">{sig.rsi:.0f}</div>
                <div style="color:#475569;font-size:.7rem;">MACD</div>
                <div style="font-size:.7rem;">{macd_icon}</div>
                <div style="color:#475569;font-size:.7rem;">Above EMA</div>
                <div style="font-size:.7rem;">{ema_icon}</div>
              </div>
              {"".join(f'<div style="background:rgba(6,182,212,.08);color:#67e8f9;padding:2px 8px;border-radius:5px;margin:2px 0;font-size:.65rem;border:1px solid rgba(6,182,212,.2);">{r}</div>' for r in sig.reasons[:3]) if sig.reasons else ""}
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 4: ML Predictions
# ═══════════════════════════════════════════════════════════════════════════════

def render_ml_predictions(df_daily: pd.DataFrame, current_price: float) -> None:
    """Section 4 — next-day, 10-day, 30-day ML prediction cards."""
    _section("🤖 ML Predictions")

    try:
        pred_nd = PredictionEngine.predict_next_day(df_daily)
        pred_multi = PredictionEngine.predict_multi_horizon(df_daily)
    except Exception:
        st.warning("ML predictions unavailable — insufficient history")
        return

    if "error" in pred_nd:
        st.warning("ML predictions unavailable — insufficient history")
        return

    # Build 3 cards: next-day, 10-day, 30-day
    horizons = [
        ("Next Day",  pred_nd),
        ("10 Days",   pred_multi.get("10_days", {})),
        ("30 Days",   pred_multi.get("30_days", {})),
    ]

    cols = st.columns(3)
    for col, (label, pred) in zip(cols, horizons):
        if not pred or "error" in pred:
            with col:
                st.markdown(f"""
                <div style="background:#0a1628;border:1px solid #0f2040;border-radius:12px;padding:14px;text-align:center;">
                  <div style="color:#94a3b8;font-size:.72rem;font-weight:700;">{label}</div>
                  <div style="color:#475569;font-size:.8rem;margin-top:8px;">Unavailable</div>
                </div>""", unsafe_allow_html=True)
            continue

        direction = pred.get("direction", "NEUTRAL")
        confidence = pred.get("confidence", 0.5)
        pred_price = pred.get("predicted_price", current_price)
        price_low = pred.get("price_low", current_price * 0.95)
        price_high = pred.get("price_high", current_price * 1.05)

        dir_color = "#10b981" if direction == "UP" else ("#ef4444" if direction == "DOWN" else "#f59e0b")
        dir_arrow = "▲" if direction == "UP" else ("▼" if direction == "DOWN" else "↔")

        with col:
            st.markdown(f"""
            <div style="background:#0a1628;border:1px solid #0f2040;border-radius:12px;padding:14px;text-align:center;">
              <div style="color:#94a3b8;font-size:.72rem;font-weight:700;text-transform:uppercase;
                          letter-spacing:.06em;margin-bottom:6px;">{label}</div>
              <div style="color:{dir_color};font-size:1.1rem;font-weight:800;margin-bottom:4px;">
                {dir_arrow} {direction}
              </div>
              <div style="color:#f1f5f9;font-size:1rem;font-weight:700;">₹{pred_price:,.2f}</div>
              <div style="color:#475569;font-size:.7rem;margin:4px 0;">
                ₹{price_low:,.0f} – ₹{price_high:,.0f}
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence, text=f"Confidence {confidence:.0%}")

    # Market conditions
    conditions = pred_nd.get("market_conditions", [])
    if conditions:
        st.markdown(
            '<div style="color:#475569;font-size:.72rem;margin-top:8px;">Market conditions: '
            + " · ".join(f'<span style="color:#94a3b8;">{c}</span>' for c in conditions)
            + "</div>",
            unsafe_allow_html=True,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 5: Risk Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def render_risk_metrics(df_daily: pd.DataFrame, risk_free_rate: float = 0.065) -> None:
    """Section 5 — VaR, CVaR, Sharpe, Sortino, drawdown, stress test."""
    _section("🛡️ Risk Metrics")

    result = RiskEngine.compute_all(df_daily["Close"])
    if not result or "error" in result:
        st.warning("Risk metrics unavailable — insufficient data.")
        return

    sharpe = result.get("sharpe_ratio", 0)
    sharpe_color = "#10b981" if sharpe > 1 else ("#f59e0b" if sharpe >= 0 else "#ef4444")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c2.metric("Sortino Ratio", f"{result.get('sortino_ratio', 0):.2f}")
    c3.metric("Annual Volatility", f"{result.get('volatility_annual', 0) * 100:.1f}%")
    c4.metric("Total Return", f"{result.get('total_return', 0) * 100:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    var_val = result.get("var_historical", {})
    var_val = var_val.get("var", 0) if isinstance(var_val, dict) else var_val
    cvar_val = result.get("cvar", {})
    cvar_val = cvar_val.get("cvar", 0) if isinstance(cvar_val, dict) else cvar_val
    dd_val = result.get("max_drawdown", {})
    dd_val = dd_val.get("max_drawdown", 0) if isinstance(dd_val, dict) else dd_val
    c5.metric("VaR 95%", f"{float(var_val) * 100:.2f}%")
    c6.metric("CVaR 95%", f"{float(cvar_val) * 100:.2f}%")
    c7.metric("Max Drawdown", f"{float(dd_val) * 100:.1f}%")
    c8.metric("Calmar Ratio", f"{result.get('calmar_ratio', 0):.2f}")

    st.markdown(
        f'<div style="margin:6px 0 10px;font-size:.75rem;">Sharpe: '
        f'<span style="color:{sharpe_color};font-weight:700;">{sharpe:.2f}</span> '
        f'{"(Excellent)" if sharpe > 1 else ("(Acceptable)" if sharpe >= 0 else "(Poor)")}</div>',
        unsafe_allow_html=True,
    )

    # Stress test table
    try:
        returns = df_daily["Close"].pct_change().dropna()
        stress = RiskEngine.stress_test(returns)
        if stress:
            st.markdown("**Stress Test Scenarios**")
            st.dataframe(pd.DataFrame(stress), use_container_width=True, hide_index=True)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 6: Monte Carlo Simulation
# ═══════════════════════════════════════════════════════════════════════════════

def render_monte_carlo(
    df_daily: pd.DataFrame,
    n_simulations: int = 5000,
    n_days: int = 30,
) -> None:
    """Section 6 — fan chart + probability stats."""
    _section("🎲 Monte Carlo Simulation (30-day)")

    result = MonteCarloEngine.simulate(df_daily["Close"], n_simulations=n_simulations, n_days=n_days)
    if not result or "error" in result:
        st.warning("Monte Carlo unavailable — insufficient data.")
        return

    price = float(df_daily["Close"].iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prob Profit", f"{result.get('prob_profit', 0):.0%}")
    c2.metric("Expected Price", f"₹{result.get('expected_price', price):,.2f}")
    c3.metric("p5 (Worst 5%)", f"₹{result.get('p5', price):,.2f}")
    c4.metric("p95 (Best 5%)", f"₹{result.get('p95', price):,.2f}")

    paths = result.get("sample_paths", [])
    if not paths:
        return

    fig = go.Figure()
    days_x = list(range(len(paths[0])))

    # 50 sample paths (thin grey)
    for path in paths[:50]:
        fig.add_trace(go.Scatter(
            x=days_x, y=path, mode="lines",
            line=dict(color="rgba(148,163,184,0.10)", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Percentile bands
    import numpy as _np
    arr = _np.array(paths)
    bands = [
        (5,  95, "rgba(239,68,68,0.12)",   "#ef4444", "p5–p95"),
        (25, 75, "rgba(245,158,11,0.18)",  "#f59e0b", "p25–p75"),
    ]
    for lo, hi, fill, line_c, name in bands:
        lo_vals = _np.percentile(arr, lo, axis=0).tolist()
        hi_vals = _np.percentile(arr, hi, axis=0).tolist()
        fig.add_trace(go.Scatter(
            x=days_x + days_x[::-1],
            y=hi_vals + lo_vals[::-1],
            fill="toself", fillcolor=fill,
            line=dict(color="rgba(0,0,0,0)"),
            name=name, showlegend=True,
        ))

    # Median line
    median = _np.percentile(arr, 50, axis=0)
    fig.add_trace(go.Scatter(
        x=days_x, y=median, mode="lines",
        line=dict(color="#10b981", width=2.5),
        name="Median",
    ))

    fig.add_hline(y=price, line_color="#6366f1", line_dash="dot",
                  annotation_text="Current Price", annotation_font_color="#6366f1")

    layout = _plotly_layout("Monte Carlo — 30-Day Price Distribution", height=420)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 7: Multi-Analyzer Scorecard
# ═══════════════════════════════════════════════════════════════════════════════

def render_multi_analyzer_scorecard(
    df_daily: pd.DataFrame,
    sector: str,
    capital: float,
    risk_pct: float,
) -> None:
    """Section 7 — radar chart + per-analyzer breakdown."""
    _section("🔬 Multi-Analyzer Scorecard")

    try:
        ma = MultiAnalyzer()
        result = ma.analyze(df_daily, sector=sector, capital=capital, risk_pct=risk_pct)
    except Exception as e:
        st.warning(f"Multi-analyzer unavailable: {e}")
        return

    combined = result.get("combined_score", 0.5)
    signal = result.get("signal", "NEUTRAL")
    sig_color = "#10b981" if "BUY" in signal else ("#ef4444" if "SELL" in signal else "#f59e0b")

    st.markdown(
        f'<div style="margin-bottom:12px;font-size:1rem;">'
        f'Combined Score: <b style="color:#f1f5f9;font-size:1.3rem;">{combined:.0%}</b>'
        f'&nbsp;&nbsp;{_badge(signal)}</div>',
        unsafe_allow_html=True,
    )

    breakdown = result.get("analyzers", {})
    if breakdown:
        an_cols = st.columns(len(breakdown))
        for col, (name, data) in zip(an_cols, breakdown.items()):
            sc = data.get("score", 0.5)
            sig = data.get("signal", "NEUTRAL")
            sig_icon = "🟢" if sig == "BULLISH" else ("🔴" if sig == "BEARISH" else "🟡")
            with col:
                with st.expander(f"{sig_icon} {name} — {sc:.0%}", expanded=False):
                    st.markdown(f"**Score:** {sc:.0%}")
                    st.markdown(f"**Signal:** {sig}")
                    for r in data.get("reasons", [])[:5]:
                        st.markdown(f"• {r}")

        # Radar chart
        names_r = list(breakdown.keys())
        scores_r = [breakdown[n].get("score", 0.5) for n in names_r]
        fig_radar = go.Figure(go.Scatterpolar(
            r=scores_r + [scores_r[0]],
            theta=names_r + [names_r[0]],
            fill="toself",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color="#6366f1", width=2),
            name="Scores",
        ))
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                                gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                angularaxis=dict(gridcolor="#1e3a5f"),
                bgcolor="#080f1e",
            ),
            template="plotly_dark", paper_bgcolor="#04080f",
            height=380, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    reasons = result.get("reasoning", [])
    if reasons:
        st.markdown("**Signal Reasoning**")
        for r in reasons[:8]:
            icon = "✅" if any(w in r.lower() for w in ["bullish", "rising", "positive", "strong", "above"]) else "⚠️"
            st.markdown(f"{icon} {r}")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 8: Global Factor Impact
# ═══════════════════════════════════════════════════════════════════════════════

def render_global_factors(sector: str) -> None:
    """Section 8 — sector macro/geopolitical factors."""
    _section("🌍 Global Factor Impact")

    factors = StockMetadata.get_global_factors(sector)
    theme = factors.get("theme", "NEUTRAL")
    theme_color = "#10b981" if "STRONG" in theme else ("#f59e0b" if "MODERATE" in theme else "#94a3b8")

    st.markdown(
        f'<div style="margin-bottom:10px;">Theme: '
        f'<span style="color:{theme_color};font-weight:800;font-size:1rem;">{theme}</span></div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="color:#6ee7b7;font-size:.78rem;font-weight:700;margin-bottom:6px;">✅ Positive Factors</div>', unsafe_allow_html=True)
        for f in factors.get("positive", []):
            st.markdown(f'<span class="factor-pos">✅ {f}</span>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div style="color:#fca5a5;font-size:.78rem;font-weight:700;margin-bottom:6px;">⚠️ Risk Factors</div>', unsafe_allow_html=True)
        for f in factors.get("negative", []):
            st.markdown(f'<span class="factor-neg">⚠️ {f}</span>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 9: Trade Setup Generator
# ═══════════════════════════════════════════════════════════════════════════════

def render_trade_setup(bundle: AnalysisBundle, capital: float, risk_pct: float) -> None:
    """Section 9 — ATR-based trade setup with capital slider."""
    _section("💼 Trade Setup Generator")

    # Capital slider — recalculates without re-fetching
    slider_capital = st.slider(
        "Adjust Capital (₹)", min_value=1000, max_value=5_000_000,
        value=int(capital), step=1000, key="dd_ts_capital",
    )

    ma_result = bundle.ma_result or {"combined_score": 0.5}
    ts = compute_trade_setup(bundle.current_price, bundle.df_daily, slider_capital, risk_pct, ma_result)

    # ATR floor info
    if ts.atr <= bundle.current_price * 0.006:
        st.info("ℹ️ ATR floor applied — stock may be illiquid or low-volatility. Widen stops manually.")

    sig_color = "#10b981" if ts.signal == "BUY" else ("#ef4444" if ts.signal == "AVOID" else "#f59e0b")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entry", f"₹{ts.entry:,.2f}")
    c2.metric("Target 1", f"₹{ts.target_1:,.2f}", f"+₹{ts.reward_per_share:,.2f}/share")
    c3.metric("Target 2", f"₹{ts.target_2:,.2f}")
    c4.metric("Stop Loss", f"₹{ts.stop_loss:,.2f}", f"-₹{ts.risk_per_share:,.2f}/share", delta_color="inverse")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Qty", ts.qty)
    c6.metric("Invested", f"₹{ts.invested:,.0f}")
    c7.metric("Max Profit", f"₹{ts.max_profit:,.0f}")
    c8.metric("R:R Ratio", f"{ts.risk_reward}x")

    # Copy-to-clipboard
    trade_text = (
        f"TRADE SETUP — {bundle.symbol}\n"
        f"Entry:    ₹{ts.entry:,.2f}\n"
        f"Target 1: ₹{ts.target_1:,.2f}\n"
        f"Target 2: ₹{ts.target_2:,.2f}\n"
        f"Stop Loss:₹{ts.stop_loss:,.2f}\n"
        f"Qty:      {ts.qty}\n"
        f"Invested: ₹{ts.invested:,.0f}\n"
        f"Max Profit:₹{ts.max_profit:,.0f}\n"
        f"Max Loss: ₹{ts.max_loss:,.0f}\n"
        f"R:R:      {ts.risk_reward}x\n"
        f"Signal:   {ts.signal} ({ts.confidence:.0%})\n"
        f"ATR:      ₹{ts.atr:,.2f}"
    )
    st.code(trade_text, language="text")


# ═══════════════════════════════════════════════════════════════════════════════
# Panel 10: Peer Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def render_peer_comparison(
    symbol: str,
    sector: str,
    df_daily: pd.DataFrame,
    peer_symbols: Optional[list] = None,
) -> None:
    """Section 10 — normalised price chart + metrics table."""
    _section("📊 Peer Comparison")

    with st.spinner("Loading peer data…"):
        try:
            peers = load_peer_data(symbol, sector)
        except Exception as e:
            st.warning(f"Peer comparison unavailable: {e}")
            return

    if not peers:
        st.info("No peer data available for this sector.")
        return

    # Normalised price chart (rebased to 100)
    fig = go.Figure()
    colors = ["#6366f1", "#10b981", "#f59e0b", "#06b6d4", "#ec4899"]
    for i, p in enumerate(peers):
        df_p = p.get("df", pd.DataFrame())
        if df_p.empty:
            continue
        norm = (df_p["Close"] / df_p["Close"].iloc[0]) * 100
        is_target = p.get("is_target", False)
        fig.add_trace(go.Scatter(
            x=df_p.index, y=norm, mode="lines",
            line=dict(color=colors[i % len(colors)], width=2.5 if is_target else 1.5),
            name=f"★ {p['symbol']}" if is_target else p["symbol"],
        ))

    fig.add_hline(y=100, line_color="#475569", line_dash="dot",
                  annotation_text="Base (100)", annotation_font_color="#475569")
    layout = _plotly_layout("Normalised 6-Month Performance (Base = 100)", height=380)
    fig.update_layout(**layout)
    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    tbl_data = []
    for p in peers:
        tbl_data.append({
            "Stock":  ("★ " if p.get("is_target") else "") + p["symbol"],
            "Price":  f"₹{p['price']:,.2f}",
            "1M %":   f"{p['return_1m']:+.1f}%",
            "3M %":   f"{p['return_3m']:+.1f}%",
            "6M %":   f"{p['return_6m']:+.1f}%",
            "RSI":    p["rsi"],
            "Vol %":  f"{p['volatility']:.1f}%",
            "Sharpe": p["sharpe"],
        })

    if not tbl_data:
        st.info("No peer data available for this sector.")
        return

    df_tbl = pd.DataFrame(tbl_data)
    st.dataframe(df_tbl, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR — render_stock_analysis_tab()
# ═══════════════════════════════════════════════════════════════════════════════

def render_stock_analysis_tab(
    capital: float,
    risk_pct: float,
    symbol_list: list,
) -> None:
    """
    Full Deep Dive tab orchestrator.
    Wires symbol selector → data loading → all 10 panel renderers.
    """
    from backend.engines.stock_analysis_engine import load_analysis_bundle

    st.markdown("""
    <div style='margin-bottom:16px;'>
        <span style='font-size:1.4rem;font-weight:900;color:#f1f5f9;'>🔍 Deep Dive — Stock Analysis</span>
        <div style='color:#334155;font-size:0.8rem;margin-top:4px;'>
            Full technical + ML + risk + Monte Carlo + peer analysis for any NSE stock
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Inputs ────────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([2.5, 1, 1])
    with col1:
        default_idx = symbol_list.index("RELIANCE") if "RELIANCE" in symbol_list else 0
        dd_symbol = st.selectbox(
            "Select Stock", symbol_list, index=default_idx,
            key="sap_sym", label_visibility="visible",
        )
    with col2:
        dd_capital = st.number_input("Capital (₹)", value=int(capital),
                                      min_value=1000, step=1000, key="sap_cap")
    with col3:
        dd_risk = st.slider("Risk %", 0.5, 5.0, float(risk_pct * 100), 0.5, key="sap_risk") / 100

    analyse_btn = st.button("🔍 ANALYSE", type="primary", key="sap_btn")

    # ── Load bundle ───────────────────────────────────────────────────────────
    cache_key = f"sap_{dd_symbol}"
    if analyse_btn:
        with st.spinner(f"Loading full analysis for {dd_symbol}…"):
            try:
                bundle = load_analysis_bundle(dd_symbol, dd_capital, dd_risk)
                st.session_state[cache_key] = bundle
            except ValueError as e:
                st.error(f"❌ {e}")
                bundle = None
            except Exception as e:
                st.warning(f"⚠️ Could not load data for {dd_symbol}: {e}")
                bundle = None
    else:
        bundle = st.session_state.get(cache_key)

    if not bundle:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;'>
            <div style='font-size:3rem;margin-bottom:12px;'>🔍</div>
            <div style='font-size:1.4rem;font-weight:900;color:#f1f5f9;margin-bottom:8px;'>Stock Deep Dive</div>
            <div style='color:#334155;font-size:0.9rem;'>
                Select a stock above and click <b style='color:#3b82f6;'>ANALYSE</b>
            </div>
        </div>""", unsafe_allow_html=True)
        return

    st.markdown("---")

    # ── Panel 1: Price Overview ───────────────────────────────────────────────
    try:
        render_price_overview(bundle)
    except Exception as e:
        st.warning(f"⚠️ Price overview unavailable: {e}")

    st.markdown("---")

    # ── Panel 2: Technical Chart ──────────────────────────────────────────────
    try:
        render_technical_chart(
            bundle.df_daily, bundle.df_features,
            bundle.df_intra, bundle.trade_setup,
        )
    except Exception as e:
        st.warning(f"⚠️ Technical chart unavailable: {e}")

    st.markdown("---")

    # ── Panel 3: Multi-Timeframe Signals ─────────────────────────────────────
    try:
        render_multi_timeframe(
            bundle.symbol, bundle.df_intra,
            bundle.df_daily, bundle.df_weekly,
        )
    except Exception as e:
        st.warning(f"⚠️ Multi-timeframe signals unavailable: {e}")

    st.markdown("---")

    # ── Panel 4: ML Predictions ───────────────────────────────────────────────
    try:
        render_ml_predictions(bundle.df_daily, bundle.current_price)
    except Exception as e:
        st.warning(f"⚠️ ML predictions unavailable: {e}")

    st.markdown("---")

    # ── Panel 5: Risk Metrics ─────────────────────────────────────────────────
    try:
        render_risk_metrics(bundle.df_daily)
    except Exception as e:
        st.warning(f"⚠️ Risk metrics unavailable: {e}")

    st.markdown("---")

    # ── Panel 6: Monte Carlo ──────────────────────────────────────────────────
    try:
        render_monte_carlo(bundle.df_daily)
    except Exception as e:
        st.warning(f"⚠️ Monte Carlo unavailable: {e}")

    st.markdown("---")

    # ── Panel 7: Multi-Analyzer Scorecard ────────────────────────────────────
    try:
        render_multi_analyzer_scorecard(
            bundle.df_daily, bundle.sector, dd_capital, dd_risk,
        )
    except Exception as e:
        st.warning(f"⚠️ Multi-analyzer unavailable: {e}")

    st.markdown("---")

    # ── Panel 8: Global Factors ───────────────────────────────────────────────
    try:
        render_global_factors(bundle.sector)
    except Exception as e:
        st.warning(f"⚠️ Global factors unavailable: {e}")

    st.markdown("---")

    # ── Panel 9: Trade Setup ──────────────────────────────────────────────────
    try:
        render_trade_setup(bundle, dd_capital, dd_risk)
    except Exception as e:
        st.warning(f"⚠️ Trade setup unavailable: {e}")

    st.markdown("---")

    # ── Panel 10: Peer Comparison ─────────────────────────────────────────────
    try:
        render_peer_comparison(bundle.symbol, bundle.sector, bundle.df_daily)
    except Exception as e:
        st.warning(f"⚠️ Peer comparison unavailable: {e}")

"""QuantSignal India — Analytics Dashboard
Monte Carlo, Portfolio Optimization, Options Pricing, Risk Analysis.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.engines.data_service import DataService
from backend.engines.feature_engine import FeatureEngine
from backend.engines.signal_engine import SignalEngine
from backend.engines.monte_carlo import MonteCarloEngine
from backend.engines.risk_engine import RiskEngine
from backend.engines.portfolio_engine import PortfolioEngine
from backend.engines.options_engine import OptionsEngine
from backend.config import (
    ALL_SYMBOLS, NIFTY_INDEX, MIN_CONFIDENCE, MIN_RISK_REWARD,
    MAX_SIGNALS, RISK_FREE_RATE, MC_SIMULATIONS, MC_DAYS, SECTOR_MAP,
)

st.set_page_config(
    page_title="QuantSignal India — Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #0f172a; }
.block-container { padding-top: 2rem; }
h1, h2, h3 { color: #e2e8f0 !important; }
.stMetric label { color: #94a3b8 !important; }
.stMetric [data-testid="stMetricValue"] { color: #e2e8f0 !important; }
</style>
""", unsafe_allow_html=True)

CLEAN_SYMBOLS = sorted(set(s.replace(".NS", "") for s in ALL_SYMBOLS))

st.sidebar.title("📊 QS Analytics")
st.sidebar.caption(f"Universe: {len(ALL_SYMBOLS)} stocks")
page = st.sidebar.radio("Navigate", [
    "📊 Dashboard", "🎲 Monte Carlo", "💼 Portfolio",
    "📈 Options Pricing", "🛡️ Risk Analysis",
], label_visibility="collapsed")

# ── DASHBOARD ────────────────────────────────────────────────────────────────
if page == "📊 Dashboard":
    st.title("📊 Trading Signal Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    capital = col1.number_input("Capital (₹)", value=1_000_000, step=100_000, min_value=50_000)
    risk_pct = col2.number_input("Risk per Trade (%)", value=2.0, step=0.5, min_value=0.5, max_value=10.0) / 100
    min_conf = col3.number_input("Min Confidence", value=0.5, step=0.05, min_value=0.0, max_value=1.0)
    max_sig = col4.number_input("Max Signals", value=20, step=1, min_value=1, max_value=50)

    sector_filter = st.multiselect(
        "Filter by Sector (empty = all)",
        options=list(SECTOR_MAP.keys()), default=[],
    )

    with st.spinner("Loading NIFTY data..."):
        try:
            nifty_df = DataService.fetch_ohlcv(NIFTY_INDEX, period="2y")
            entry_score = FeatureEngine.compute_entry_score(nifty_df)
            nifty_features = FeatureEngine.compute_all_features(nifty_df)
            nifty_last = nifty_features.iloc[-1]
            nifty_price = float(nifty_df["Close"].iloc[-1])
            signal_label = "STRONG BUY" if entry_score >= 0.6 else ("MODERATE" if entry_score >= 0.3 else "DEFENSIVE")
            alloc = 100 if entry_score >= 0.6 else (50 if entry_score >= 0.3 else 15)
            st.markdown("---")
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("NIFTY 50", f"₹{nifty_price:,.0f}")
            mc2.metric("Entry Score", f"{entry_score:.2f}")
            mc3.metric("Signal", signal_label)
            mc4.metric("Allocation", f"{alloc}%")
            mc5.metric("RSI", f"{nifty_last.get('RSI', 0):.1f}")
        except Exception as e:
            st.warning(f"Could not fetch NIFTY data: {e}")

    st.markdown("---")
    if st.button("🚀 Generate Signals", type="primary", use_container_width=True):
        symbols = None
        if sector_filter:
            symbols = []
            for s in sector_filter:
                symbols.extend(SECTOR_MAP.get(s, []))
            symbols = list(dict.fromkeys(symbols))
        with st.spinner(f"Scanning {len(symbols or ALL_SYMBOLS)} stocks..."):
            engine = SignalEngine()
            signals = engine.generate_signals(
                symbols=symbols, min_confidence=min_conf,
                min_rr=MIN_RISK_REWARD, max_signals=max_sig,
                capital=capital, risk_pct=risk_pct,
            )
            st.session_state["signals"] = signals

    if "signals" in st.session_state and st.session_state["signals"]:
        signals = st.session_state["signals"]
        avg_conf = np.mean([s["confidence"] for s in signals])
        avg_rr = np.mean([s["risk_reward"] for s in signals])
        long_ct = sum(1 for s in signals if s["direction"] == "LONG")
        sc1, sc2, sc3, sc4 = st.columns(4)
        sc1.metric("Signals", len(signals))
        sc2.metric("Avg Confidence", f"{avg_conf:.2f}")
        sc3.metric("Avg R:R", f"{avg_rr:.1f}x")
        sc4.metric("Long / Short", f"{long_ct} / {len(signals) - long_ct}")

        df = pd.DataFrame(signals)
        cols = ["symbol", "direction", "entry", "target", "stop_loss",
                "confidence", "risk_reward", "rsi", "position_size", "position_value", "rationale"]
        cols = [c for c in cols if c in df.columns]
        df_d = df[cols].copy()
        df_d.index = range(1, len(df_d) + 1)
        df_d.columns = ["Symbol", "Dir", "Entry ₹", "Target ₹", "SL ₹",
                         "Confidence", "R:R", "RSI", "Qty", "Value ₹", "Rationale"]
        st.dataframe(
            df_d.style
            .background_gradient(subset=["Confidence"], cmap="RdYlGn", vmin=0, vmax=1)
            .format({"Entry ₹": "₹{:,.2f}", "Target ₹": "₹{:,.2f}", "SL ₹": "₹{:,.2f}",
                     "Value ₹": "₹{:,.0f}", "Confidence": "{:.2%}", "R:R": "{:.1f}x"}),
            use_container_width=True,
            height=min(500, 50 + 35 * len(df_d)),
        )
        fig = go.Figure(go.Bar(
            x=[s["symbol"] for s in signals],
            y=[s["confidence"] for s in signals],
            marker_color=["#10b981" if s["direction"] == "LONG" else "#ef4444" for s in signals],
            text=[f"{s['confidence']:.0%}" for s in signals], textposition="outside",
        ))
        fig.update_layout(title="Signal Confidence", template="plotly_dark",
                          paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                          yaxis_range=[0, 1.1], height=350, margin=dict(t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)

# ── MONTE CARLO ───────────────────────────────────────────────────────────────
elif page == "🎲 Monte Carlo":
    st.title("🎲 Monte Carlo Simulation")
    col1, col2, col3 = st.columns(3)
    mc_symbol = col1.selectbox("Stock", CLEAN_SYMBOLS,
                                index=CLEAN_SYMBOLS.index("RELIANCE") if "RELIANCE" in CLEAN_SYMBOLS else 0)
    mc_days = col2.number_input("Forecast Days", value=30, min_value=5, max_value=252)
    mc_sims = col3.number_input("Simulations", value=10000, min_value=100, max_value=50000, step=1000)

    if st.button("▶️ Run Simulation", type="primary"):
        with st.spinner(f"Running {mc_sims:,} simulations for {mc_symbol}..."):
            df = DataService.fetch_ohlcv(f"{mc_symbol}.NS", period="1y")
            if df.empty:
                st.error(f"No data for {mc_symbol}")
            else:
                st.session_state["mc_result"] = MonteCarloEngine.simulate(df["Close"], mc_sims, mc_days)
                st.session_state["mc_symbol"] = mc_symbol

    if "mc_result" in st.session_state:
        r = st.session_state["mc_result"]
        sym = st.session_state.get("mc_symbol", "")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Current Price", f"₹{r['current_price']:,.2f}")
        delta_exp = r["expected_price"] - r["current_price"]
        m2.metric("Expected Price", f"₹{r['expected_price']:,.2f}",
                  f"{'+'if delta_exp>0 else ''}{delta_exp:,.0f}")
        m3.metric("Prob of Profit", f"{r['prob_profit']:.1%}")
        m4.metric("VaR 95%", f"{r['var_95']:.2%}")
        m5, m6, m7, m8 = st.columns(4)
        m5.metric("P5 (Bear)", f"₹{r['p5']:,.2f}")
        m6.metric("P25", f"₹{r['p25']:,.2f}")
        m7.metric("P75", f"₹{r['p75']:,.2f}")
        m8.metric("P95 (Bull)", f"₹{r['p95']:,.2f}")

        fig = go.Figure()
        for path in r["sample_paths"][:50]:
            fig.add_trace(go.Scatter(y=path, mode="lines",
                                     line=dict(width=0.4, color="rgba(99,102,241,0.12)"),
                                     showlegend=False, hoverinfo="skip"))
        mean_path = np.mean(r["sample_paths"], axis=0).tolist()
        fig.add_trace(go.Scatter(y=mean_path, mode="lines",
                                  line=dict(width=2.5, color="#6366f1"), name="Mean Path"))
        fig.add_hline(y=r["current_price"], line_dash="dash", line_color="#f59e0b",
                      annotation_text="Current")
        fig.update_layout(title=f"{sym} — {mc_sims:,} Simulations ({mc_days}d)",
                          template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                          xaxis_title="Days", yaxis_title="Price (₹)", height=450)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Histogram(x=r["final_distribution"], nbinsx=80,
                                       marker_color="rgba(99,102,241,0.6)"))
        fig2.add_vline(x=r["current_price"], line_dash="dash", line_color="#f59e0b",
                       annotation_text="Current")
        fig2.add_vline(x=r["p5"], line_dash="dot", line_color="#ef4444", annotation_text="P5")
        fig2.add_vline(x=r["p95"], line_dash="dot", line_color="#10b981", annotation_text="P95")
        fig2.update_layout(title="Final Price Distribution", template="plotly_dark",
                            paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ── PORTFOLIO ─────────────────────────────────────────────────────────────────
elif page == "💼 Portfolio":
    st.title("💼 Portfolio Optimization")
    default_syms = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK",
                    "HAL", "BEL", "RVNL", "NTPC", "SBIN"]
    selected = st.multiselect("Select Stocks (min 2)", CLEAN_SYMBOLS, default=default_syms)
    n_port = st.slider("Random Portfolios", 1000, 50000, 10000, 1000)

    if len(selected) < 2:
        st.warning("Select at least 2 stocks.")
    elif st.button("🔄 Optimize Portfolio", type="primary"):
        with st.spinner(f"Simulating {n_port:,} portfolios..."):
            price_dict = DataService.fetch_multiple([f"{s}.NS" for s in selected], period="1y")
            result = PortfolioEngine.optimize(price_dict, n_port)
            st.session_state["port_result"] = result

    if "port_result" in st.session_state:
        r = st.session_state["port_result"]
        if "error" in r:
            st.error(r["error"])
        else:
            pc1, pc2 = st.columns(2)
            for col, key, label in [(pc1, "max_sharpe", "⭐ Max Sharpe"),
                                     (pc2, "min_variance", "🛡️ Min Variance")]:
                with col:
                    st.markdown(f"### {label}")
                    st.metric("Annual Return", f"{r[key]['return']:.1%}")
                    st.metric("Volatility", f"{r[key]['volatility']:.1%}")
                    st.metric("Sharpe Ratio", f"{r[key]['sharpe']:.2f}")
                    w_df = pd.DataFrame(list(r[key]["weights"].items()), columns=["Stock", "Weight"])
                    w_df["Weight %"] = (w_df["Weight"] * 100).round(1)
                    st.dataframe(w_df[["Stock", "Weight %"]], use_container_width=True, hide_index=True)

            scatter_df = pd.DataFrame(r["scatter"])
            fig = px.scatter(scatter_df, x="volatility", y="return", color="sharpe",
                             color_continuous_scale="Viridis", opacity=0.5)
            fig.add_trace(go.Scatter(x=[r["max_sharpe"]["volatility"]], y=[r["max_sharpe"]["return"]],
                                      mode="markers", marker=dict(size=16, color="#10b981", symbol="star"),
                                      name="Max Sharpe"))
            fig.add_trace(go.Scatter(x=[r["min_variance"]["volatility"]], y=[r["min_variance"]["return"]],
                                      mode="markers", marker=dict(size=16, color="#06b6d4", symbol="diamond"),
                                      name="Min Variance"))
            fig.update_layout(title="Efficient Frontier", template="plotly_dark",
                               paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                               xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=500)
            st.plotly_chart(fig, use_container_width=True)

# ── OPTIONS PRICING ───────────────────────────────────────────────────────────
elif page == "📈 Options Pricing":
    st.title("📈 Black-Scholes Options Pricing")
    col1, col2, col3, col4, col5 = st.columns(5)
    spot = col1.number_input("Spot Price ₹", value=2500.0, step=50.0, min_value=1.0)
    strike = col2.number_input("Strike Price ₹", value=2500.0, step=50.0, min_value=1.0)
    expiry_days = col3.number_input("Days to Expiry", value=30, min_value=1, max_value=365)
    rate = col4.number_input("Risk-Free Rate", value=0.065, step=0.005, format="%.3f")
    vol = col5.number_input("Volatility (σ)", value=0.20, step=0.01, min_value=0.01, max_value=2.0, format="%.2f")

    if st.button("💰 Price Options", type="primary"):
        result = OptionsEngine.black_scholes(spot, strike, expiry_days / 365.0, rate, vol)
        if "error" in result:
            st.error(result["error"])
        else:
            st.session_state["opt_result"] = result

    if "opt_result" in st.session_state:
        r = st.session_state["opt_result"]
        oc1, oc2 = st.columns(2)
        for col, side, label in [(oc1, "call", "🟢 Call Option (CE)"),
                                   (oc2, "put", "🔴 Put Option (PE)")]:
            with col:
                st.markdown(f"### {label}")
                st.markdown(f"## ₹{r[side]['price']:.2f}")
                for lbl, key in [("Delta (Δ)", "delta"), ("Gamma (Γ)", "gamma"),
                                   ("Theta (Θ)", "theta"), ("Vega (ν)", "vega"), ("Rho (ρ)", "rho")]:
                    st.text(f"{lbl}: {r[side][key]:.4f}")

        spots = np.linspace(spot * 0.8, spot * 1.2, 50)
        deltas_c, gammas, vegas = [], [], []
        for s in spots:
            res = OptionsEngine.black_scholes(s, strike, expiry_days / 365.0, rate, vol)
            if "error" not in res:
                deltas_c.append(res["call"]["delta"])
                gammas.append(res["call"]["gamma"])
                vegas.append(res["call"]["vega"])
            else:
                deltas_c.append(0); gammas.append(0); vegas.append(0)
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Delta", "Gamma", "Vega"])
        fig.add_trace(go.Scatter(x=spots, y=deltas_c, line=dict(color="#6366f1")), row=1, col=1)
        fig.add_trace(go.Scatter(x=spots, y=gammas, line=dict(color="#10b981")), row=1, col=2)
        fig.add_trace(go.Scatter(x=spots, y=vegas, line=dict(color="#f59e0b")), row=1, col=3)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                          height=300, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

# ── RISK ANALYSIS ─────────────────────────────────────────────────────────────
elif page == "🛡️ Risk Analysis":
    st.title("🛡️ Risk Analysis")
    risk_sym = st.selectbox("Select Stock", CLEAN_SYMBOLS,
                             index=CLEAN_SYMBOLS.index("RELIANCE") if "RELIANCE" in CLEAN_SYMBOLS else 0)
    if st.button("📊 Analyze Risk", type="primary"):
        with st.spinner(f"Computing risk metrics for {risk_sym}..."):
            df = DataService.fetch_ohlcv(f"{risk_sym}.NS", period="2y")
            if df.empty:
                st.error(f"No data for {risk_sym}")
            else:
                metrics = RiskEngine.compute_all(df["Close"])
                stress = RiskEngine.stress_test(df["Close"].pct_change().dropna())
                st.session_state["risk_result"] = {
                    "metrics": metrics, "stress": stress,
                    "symbol": risk_sym, "prices": df["Close"],
                }

    if "risk_result" in st.session_state:
        r = st.session_state["risk_result"]
        m = r["metrics"]
        st.markdown(f"### {r['symbol']} Risk Metrics")
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Sharpe Ratio", f"{m['sharpe_ratio']:.2f}")
        rc2.metric("Sortino Ratio", f"{m['sortino_ratio']:.2f}")
        rc3.metric("Annual Volatility", f"{m['volatility_annual']:.1%}")
        rc4.metric("Total Return", f"{m['total_return']:.1%}")

        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Historical VaR", f"{m['var_historical']['var']:.2%}")
        vc2.metric("Parametric VaR", f"{m['var_parametric']['var']:.2%}")
        vc3.metric("MC VaR", f"{m['var_monte_carlo']['var']:.2%}")
        vc4.metric("CVaR", f"{m['cvar']['cvar']:.2%}")

        dd = m["max_drawdown"]
        st.metric("Max Drawdown", f"{dd['max_drawdown']:.1%}",
                  f"{dd.get('peak_date','')} → {dd.get('trough_date','')}")

        stress_df = pd.DataFrame(r["stress"])
        stress_df["Shock"] = stress_df["shock"].apply(lambda x: f"{x:+.0%}")
        stress_df["Impact (Ann.)"] = stress_df["annualized_impact"].apply(lambda x: f"{x:+.1%}")
        st.dataframe(stress_df[["scenario", "Shock", "Impact (Ann.)"]], use_container_width=True, hide_index=True)

        prices = r["prices"]
        cummax = prices.cummax()
        drawdown = (prices - cummax) / cummax
        fig = make_subplots(rows=2, cols=1, row_heights=[0.7, 0.3],
                             shared_xaxes=True, vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=prices.index, y=prices, mode="lines",
                                  line=dict(color="#6366f1"), name="Price"), row=1, col=1)
        fig.add_trace(go.Scatter(x=drawdown.index, y=drawdown, mode="lines", fill="tozeroy",
                                  line=dict(color="#ef4444", width=1),
                                  fillcolor="rgba(239,68,68,0.2)", name="Drawdown"), row=2, col=1)
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0f172a", plot_bgcolor="#1e293b",
                          height=500, margin=dict(t=20, b=40))
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.caption("Data: yfinance | Not financial advice")
st.sidebar.caption(f"Universe: {len(ALL_SYMBOLS)} stocks | 13 sectors")

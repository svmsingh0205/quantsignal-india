"""
QuantSignal India — Enhanced Analytics Dashboard v4.0
Multi-Analyzer | Capital-Aware | Global Macro | Indian Market Intelligence
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
from backend.engines.multi_analyzer import (
    MultiAnalyzer, TechnicalAnalyzer, MomentumAnalyzer,
    VolumeAnalyzer, GlobalMacroAnalyzer, GeopoliticalAnalyzer,
    GLOBAL_INDICES,
)
from backend.config import (
    ALL_SYMBOLS, NIFTY_INDEX, MIN_CONFIDENCE, MIN_RISK_REWARD,
    MAX_SIGNALS, RISK_FREE_RATE, MC_SIMULATIONS, MC_DAYS, SECTOR_MAP,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantSignal India — Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #0a0f1e !important; }
.block-container { padding-top: 1.5rem !important; max-width: 100% !important; }
h1, h2, h3 { color: #e2e8f0 !important; }
.stMetric label { color: #94a3b8 !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
.stMetric [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.3rem !important; font-weight: 700 !important; }
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0d1b2e 0%, #0a1628 100%) !important;
    border: 1px solid #1e3a5f !important; border-radius: 10px !important;
    padding: 12px 16px !important;
}
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    color: #fff !important; border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; font-size: 0.88rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.35) !important;
}
.stButton > button:hover { background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg, #060c1a 0%, #040810 100%) !important; border-right: 1px solid #0f2040 !important; }
section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.72rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.05em; }
.stTabs [data-baseweb="tab-list"] { background: #080f1e !important; border-radius: 10px !important; padding: 4px !important; border: 1px solid #0f2040 !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #475569 !important; border-radius: 8px !important; font-weight: 600 !important; font-size: 0.82rem !important; }
.stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1e40af, #2563eb) !important; color: #fff !important; }
.analyzer-card {
    background: linear-gradient(135deg, #0d1b2e, #0a1628);
    border: 1px solid #1e3a5f; border-radius: 10px; padding: 14px;
    margin-bottom: 8px;
}
.signal-bullish { color: #10b981 !important; font-weight: 700; }
.signal-bearish { color: #ef4444 !important; font-weight: 700; }
.signal-neutral  { color: #f59e0b !important; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

CLEAN_SYMBOLS = sorted(set(s.replace(".NS", "") for s in ALL_SYMBOLS))
ALL_ANALYZERS = ["Technical", "Momentum", "Volume", "Global Macro", "Geopolitical"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 QS Analytics v4")
st.sidebar.caption(f"Universe: {len(ALL_SYMBOLS)} stocks · 13 sectors")

page = st.sidebar.radio("Navigate", [
    "📊 Dashboard",
    "🔬 Multi-Analyzer",
    "🌍 Global Macro",
    "🎲 Monte Carlo",
    "💼 Portfolio",
    "📈 Options Pricing",
    "🛡️ Risk Analysis",
], label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("**⚙️ Global Settings**")

# Capital input — global, used across all pages
capital = st.sidebar.number_input(
    "Capital (₹)", value=1_000_000, step=100_000, min_value=10_000,
    help="All position sizing and allocation calculations use this capital",
)
risk_pct = st.sidebar.number_input(
    "Risk per Trade (%)", value=2.0, step=0.5, min_value=0.5, max_value=10.0,
    help="% of capital risked per trade",
) / 100

st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 Analyzer Filters**")
enabled_analyzers = st.sidebar.multiselect(
    "Active Analyzers",
    options=ALL_ANALYZERS,
    default=ALL_ANALYZERS,
    help="Enable/disable individual analyzers. Weights auto-normalize.",
)
if not enabled_analyzers:
    enabled_analyzers = ALL_ANALYZERS
    st.sidebar.warning("All analyzers re-enabled (none selected)")

st.sidebar.markdown("---")
st.sidebar.caption("Data: yfinance | Not financial advice")
st.sidebar.caption(f"Analyzers active: {len(enabled_analyzers)}/{len(ALL_ANALYZERS)}")


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD PAGE
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.title("📊 Trading Signal Dashboard")

    # ── Controls row ──────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    min_conf = col1.number_input("Min Confidence", value=0.50, step=0.05, min_value=0.0, max_value=1.0)
    max_sig  = col2.number_input("Max Signals", value=20, step=1, min_value=1, max_value=50)
    min_rr   = col3.number_input("Min Risk:Reward", value=1.5, step=0.1, min_value=0.5, max_value=5.0)

    sector_filter = st.multiselect(
        "Filter by Sector (empty = all)",
        options=list(SECTOR_MAP.keys()), default=[],
    )

    # ── NIFTY market overview ─────────────────────────────────────────────────
    with st.spinner("Loading NIFTY data..."):
        try:
            nifty_df = DataService.fetch_ohlcv(NIFTY_INDEX, period="2y")
            entry_score = FeatureEngine.compute_entry_score(nifty_df)
            nifty_features = FeatureEngine.compute_all_features(nifty_df)
            nifty_last = nifty_features.iloc[-1]
            nifty_price = float(nifty_df["Close"].iloc[-1])
            signal_label = "STRONG BUY" if entry_score >= 0.6 else ("MODERATE" if entry_score >= 0.3 else "DEFENSIVE")
            alloc = 100 if entry_score >= 0.6 else (50 if entry_score >= 0.3 else 15)
            deployed = round(capital * alloc / 100)
            st.markdown("---")
            mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
            mc1.metric("NIFTY 50", f"₹{nifty_price:,.0f}")
            mc2.metric("Entry Score", f"{entry_score:.2f}")
            mc3.metric("Signal", signal_label)
            mc4.metric("Allocation", f"{alloc}%")
            mc5.metric("RSI", f"{nifty_last.get('RSI', 0):.1f}")
            mc6.metric("Deploy Capital", f"₹{deployed:,.0f}")
        except Exception as e:
            st.warning(f"Could not fetch NIFTY data: {e}")

    st.markdown("---")

    # ── Signal generation ─────────────────────────────────────────────────────
    if st.button("🚀 Generate Signals", type="primary", use_container_width=True):
        symbols = None
        if sector_filter:
            symbols = []
            for s in sector_filter:
                symbols.extend(SECTOR_MAP.get(s, []))
            symbols = list(dict.fromkeys(symbols))
        with st.spinner(f"Scanning {len(symbols or ALL_SYMBOLS)} stocks with {len(enabled_analyzers)} analyzers..."):
            engine = SignalEngine()
            signals = engine.generate_signals(
                symbols=symbols,
                min_confidence=min_conf,
                min_rr=min_rr,
                max_signals=max_sig,
                capital=capital,
                risk_pct=risk_pct,
                enabled_analyzers=enabled_analyzers,
            )
            st.session_state["signals"] = signals

    if "signals" in st.session_state and st.session_state["signals"]:
        signals = st.session_state["signals"]
        avg_conf = np.mean([s["confidence"] for s in signals])
        avg_rr   = np.mean([s["risk_reward"] for s in signals])
        long_ct  = sum(1 for s in signals if s["direction"] == "LONG")
        total_deployed = sum(s.get("position_value", 0) for s in signals)
        total_profit   = sum(s.get("potential_profit", 0) for s in signals)

        sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
        sc1.metric("Signals", len(signals))
        sc2.metric("Avg Confidence", f"{avg_conf:.2f}")
        sc3.metric("Avg R:R", f"{avg_rr:.1f}x")
        sc4.metric("Long / Short", f"{long_ct} / {len(signals) - long_ct}")
        sc5.metric("Total Deployed", f"₹{total_deployed:,.0f}")
        sc6.metric("Max Potential Profit", f"₹{total_profit:,.0f}")

        # ── Signal table ──────────────────────────────────────────────────────
        df = pd.DataFrame(signals)
        display_cols = ["symbol", "direction", "entry", "target", "stop_loss",
                        "confidence", "multi_analyzer_score", "risk_reward",
                        "rsi", "position_size", "position_value",
                        "potential_profit", "potential_loss", "sector", "rationale"]
        display_cols = [c for c in display_cols if c in df.columns]
        df_d = df[display_cols].copy()
        df_d.index = range(1, len(df_d) + 1)
        col_rename = {
            "symbol": "Symbol", "direction": "Dir", "entry": "Entry ₹",
            "target": "Target ₹", "stop_loss": "SL ₹", "confidence": "Confidence",
            "multi_analyzer_score": "MA Score", "risk_reward": "R:R",
            "rsi": "RSI", "position_size": "Qty", "position_value": "Value ₹",
            "potential_profit": "Max Profit ₹", "potential_loss": "Max Loss ₹",
            "sector": "Sector", "rationale": "Rationale",
        }
        df_d.rename(columns=col_rename, inplace=True)
        fmt = {
            "Entry ₹": "₹{:,.2f}", "Target ₹": "₹{:,.2f}", "SL ₹": "₹{:,.2f}",
            "Value ₹": "₹{:,.0f}", "Max Profit ₹": "₹{:,.0f}", "Max Loss ₹": "₹{:,.0f}",
            "Confidence": "{:.2%}", "MA Score": "{:.2%}", "R:R": "{:.1f}x",
        }
        st.dataframe(
            df_d.style
            .background_gradient(subset=["Confidence"], cmap="RdYlGn", vmin=0, vmax=1)
            .background_gradient(subset=["MA Score"], cmap="RdYlGn", vmin=0, vmax=1)
            .format({k: v for k, v in fmt.items() if k in df_d.columns}),
            use_container_width=True,
            height=min(600, 50 + 35 * len(df_d)),
        )

        # ── Charts ────────────────────────────────────────────────────────────
        tab1, tab2, tab3 = st.tabs(["Confidence Chart", "Analyzer Breakdown", "Capital Allocation"])

        with tab1:
            fig = go.Figure(go.Bar(
                x=[s["symbol"] for s in signals],
                y=[s["confidence"] for s in signals],
                marker_color=["#10b981" if s["direction"] == "LONG" else "#ef4444" for s in signals],
                text=[f"{s['confidence']:.0%}" for s in signals], textposition="outside",
            ))
            fig.update_layout(title="Signal Confidence by Stock", template="plotly_dark",
                              paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
                              yaxis_range=[0, 1.1], height=350, margin=dict(t=40, b=40))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            # Multi-analyzer score breakdown for top signals
            top_n = min(10, len(signals))
            top_signals = signals[:top_n]
            analyzer_names = enabled_analyzers
            fig2 = go.Figure()
            colors = ["#6366f1", "#10b981", "#f59e0b", "#06b6d4", "#ec4899"]
            for i, an in enumerate(analyzer_names):
                scores = []
                for s in top_signals:
                    bd = s.get("analyzer_breakdown", {})
                    scores.append(bd.get(an, {}).get("score", 0.5) if bd else 0.5)
                fig2.add_trace(go.Bar(
                    name=an, x=[s["symbol"] for s in top_signals], y=scores,
                    marker_color=colors[i % len(colors)],
                ))
            fig2.update_layout(
                title="Analyzer Score Breakdown (Top 10 Signals)",
                barmode="group", template="plotly_dark",
                paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
                yaxis_range=[0, 1], height=380, margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            # Capital allocation pie
            top_by_value = sorted(signals, key=lambda x: x.get("position_value", 0), reverse=True)[:10]
            fig3 = go.Figure(go.Pie(
                labels=[s["symbol"] for s in top_by_value],
                values=[s.get("position_value", 0) for s in top_by_value],
                hole=0.4,
                marker_colors=px.colors.qualitative.Set3,
            ))
            fig3.update_layout(
                title=f"Capital Allocation (Total: ₹{total_deployed:,.0f})",
                template="plotly_dark", paper_bgcolor="#0a0f1e", height=400,
            )
            st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-ANALYZER PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Multi-Analyzer":
    st.title("🔬 Multi-Analyzer Deep Dive")
    st.caption("Run all analyzers on a single stock and see per-analyzer scores, reasoning, and capital-aware sizing.")

    col1, col2 = st.columns([2, 1])
    ma_symbol = col1.selectbox(
        "Select Stock",
        CLEAN_SYMBOLS,
        index=CLEAN_SYMBOLS.index("RELIANCE") if "RELIANCE" in CLEAN_SYMBOLS else 0,
    )
    ma_sector = col2.selectbox(
        "Sector (for Geopolitical analyzer)",
        [""] + list(SECTOR_MAP.keys()),
        index=0,
    )

    if st.button("🔬 Run Full Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Running {len(enabled_analyzers)} analyzers on {ma_symbol}..."):
            df = DataService.fetch_ohlcv(f"{ma_symbol}.NS", period="1y")
            if df.empty:
                st.error(f"No data for {ma_symbol}")
            else:
                ma = MultiAnalyzer(enabled_analyzers)
                result = ma.analyze(df, sector=ma_sector, capital=capital, risk_pct=risk_pct)
                st.session_state["ma_result"] = result
                st.session_state["ma_df"] = df
                st.session_state["ma_symbol"] = ma_symbol

    if "ma_result" in st.session_state:
        r = st.session_state["ma_result"]
        df = st.session_state["ma_df"]
        sym = st.session_state["ma_symbol"]

        # ── Combined score header ─────────────────────────────────────────────
        score = r["combined_score"]
        signal = r["signal"]
        color = "#10b981" if "BUY" in signal else ("#ef4444" if "SELL" in signal else "#f59e0b")

        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Combined Score", f"{score:.2%}")
        h2.metric("Signal", signal)
        h3.metric("Risk Amount", f"₹{r['risk_amount']:,.0f}")
        h4.metric("Analyzers Used", r["enabled_count"])

        st.markdown("---")

        # ── Per-analyzer cards ────────────────────────────────────────────────
        st.subheader("Analyzer Breakdown")
        breakdown = r.get("analyzers", {})
        cols = st.columns(min(len(breakdown), 3))
        for i, (name, data) in enumerate(breakdown.items()):
            col = cols[i % len(cols)]
            sig_color = "🟢" if data["signal"] == "BULLISH" else ("🔴" if data["signal"] == "BEARISH" else "🟡")
            with col:
                st.markdown(f"""
                <div class="analyzer-card">
                    <div style="font-size:0.85rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.05em">{name}</div>
                    <div style="font-size:1.6rem;font-weight:800;color:#f1f5f9;margin:4px 0">{data['score']:.0%}</div>
                    <div style="font-size:0.9rem">{sig_color} {data['signal']}</div>
                    <div style="font-size:0.75rem;color:#64748b;margin-top:4px">Weight: {data['weight']:.0%} · Contribution: {data['contribution']:.0%}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Score radar chart ─────────────────────────────────────────────────
        if breakdown:
            names_r = list(breakdown.keys())
            scores_r = [breakdown[n]["score"] for n in names_r]
            names_r_closed = names_r + [names_r[0]]
            scores_r_closed = scores_r + [scores_r[0]]
            fig_radar = go.Figure(go.Scatterpolar(
                r=scores_r_closed, theta=names_r_closed,
                fill="toself", fillcolor="rgba(99,102,241,0.2)",
                line=dict(color="#6366f1", width=2),
                name="Analyzer Scores",
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                                    gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                    angularaxis=dict(gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                    bgcolor="#0d1b2e",
                ),
                template="plotly_dark", paper_bgcolor="#0a0f1e",
                title=f"{sym} — Multi-Analyzer Radar", height=420,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ── Reasoning ─────────────────────────────────────────────────────────
        st.subheader("Signal Reasoning")
        reasons = r.get("reasoning", [])
        if reasons:
            for reason in reasons:
                icon = "✅" if any(w in reason.lower() for w in ["bullish", "rising", "positive", "low vix", "strong", "above"]) else "⚠️"
                st.markdown(f"{icon} {reason}")
        else:
            st.info("No specific reasoning generated.")

        # ── Analyzer details expander ─────────────────────────────────────────
        with st.expander("Raw Analyzer Details"):
            for name, data in breakdown.items():
                st.markdown(f"**{name}**")
                details = data.get("details", {})
                if details:
                    det_df = pd.DataFrame(list(details.items()), columns=["Metric", "Value"])
                    st.dataframe(det_df, use_container_width=True, hide_index=True)

        # ── Price chart with indicators ───────────────────────────────────────
        st.subheader(f"{sym} — Price & Indicators")
        features = FeatureEngine.compute_all_features(df)
        fig_price = make_subplots(
            rows=3, cols=1, row_heights=[0.6, 0.2, 0.2],
            shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=["Price + MAs + Bollinger", "RSI", "MACD Histogram"],
        )
        # Candlestick
        fig_price.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color="#10b981", decreasing_line_color="#ef4444",
            name="Price", showlegend=False,
        ), row=1, col=1)
        # MAs
        for ma_w, color in [(20, "#f59e0b"), (50, "#6366f1"), (200, "#06b6d4")]:
            col_name = f"MA{ma_w}"
            if col_name in features.columns:
                fig_price.add_trace(go.Scatter(
                    x=features.index, y=features[col_name], mode="lines",
                    line=dict(width=1.2, color=color), name=f"MA{ma_w}",
                ), row=1, col=1)
        # Bollinger
        if "BB_Upper" in features.columns:
            fig_price.add_trace(go.Scatter(
                x=features.index, y=features["BB_Upper"], mode="lines",
                line=dict(width=0.8, color="rgba(148,163,184,0.4)", dash="dot"),
                name="BB Upper", showlegend=False,
            ), row=1, col=1)
            fig_price.add_trace(go.Scatter(
                x=features.index, y=features["BB_Lower"], mode="lines",
                line=dict(width=0.8, color="rgba(148,163,184,0.4)", dash="dot"),
                fill="tonexty", fillcolor="rgba(148,163,184,0.05)",
                name="BB Lower", showlegend=False,
            ), row=1, col=1)
        # RSI
        if "RSI" in features.columns:
            fig_price.add_trace(go.Scatter(
                x=features.index, y=features["RSI"], mode="lines",
                line=dict(color="#f59e0b", width=1.5), name="RSI",
            ), row=2, col=1)
            fig_price.add_hline(y=70, line_dash="dash", line_color="#ef4444", row=2, col=1)
            fig_price.add_hline(y=30, line_dash="dash", line_color="#10b981", row=2, col=1)
        # MACD
        if "MACD_Hist" in features.columns:
            hist = features["MACD_Hist"]
            fig_price.add_trace(go.Bar(
                x=features.index, y=hist,
                marker_color=["#10b981" if v >= 0 else "#ef4444" for v in hist],
                name="MACD Hist",
            ), row=3, col=1)
        fig_price.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
            height=600, margin=dict(t=40, b=40), xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_price, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL MACRO PAGE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🌍 Global Macro":
    st.title("🌍 Global Macro Dashboard")
    st.caption("Live global indices, USD/INR, crude oil, gold — and their impact on Indian markets.")

    if st.button("🔄 Refresh Global Data", type="primary", use_container_width=True):
        with st.spinner("Fetching global market data..."):
            global_data = {}
            for name, ticker in GLOBAL_INDICES.items():
                try:
                    df_g = DataService.fetch_ohlcv(ticker, period="1mo")
                    if not df_g.empty:
                        price = float(df_g["Close"].iloc[-1])
                        chg_1d = float((df_g["Close"].iloc[-1] / df_g["Close"].iloc[-2] - 1) * 100) if len(df_g) >= 2 else 0
                        chg_1m = float((df_g["Close"].iloc[-1] / df_g["Close"].iloc[0] - 1) * 100) if len(df_g) >= 2 else 0
                        global_data[name] = {"price": price, "chg_1d": chg_1d, "chg_1m": chg_1m, "ticker": ticker}
                except Exception:
                    pass
            st.session_state["global_data"] = global_data

    if "global_data" in st.session_state:
        gd = st.session_state["global_data"]

        # ── Metrics grid ──────────────────────────────────────────────────────
        st.subheader("Key Global Indicators")
        items = list(gd.items())
        for row_start in range(0, len(items), 4):
            row_items = items[row_start:row_start + 4]
            cols = st.columns(len(row_items))
            for col, (name, data) in zip(cols, row_items):
                delta_str = f"{data['chg_1d']:+.2f}%"
                col.metric(name, f"{data['price']:,.2f}", delta_str)

        st.markdown("---")

        # ── Impact analysis ───────────────────────────────────────────────────
        st.subheader("India Market Impact Analysis")
        impact_rows = []
        for name, data in gd.items():
            chg = data["chg_1d"]
            if name == "VIX":
                impact = "🔴 Risk-Off" if data["price"] > 25 else ("🟡 Caution" if data["price"] > 18 else "🟢 Risk-On")
                note = f"VIX at {data['price']:.1f} — {'elevated fear' if data['price'] > 25 else 'normal'}"
            elif name == "S&P 500":
                impact = "🟢 Positive" if chg > 0.5 else ("🔴 Negative" if chg < -0.5 else "🟡 Neutral")
                note = "Global risk appetite indicator"
            elif name == "USD/INR":
                impact = "🔴 Bearish INR" if chg > 0.3 else ("🟢 Bullish INR" if chg < -0.3 else "🟡 Stable")
                note = "Rupee depreciation hurts importers (oil, metals)"
            elif name == "Crude Oil":
                impact = "🔴 Inflationary" if chg > 1 else ("🟢 Positive" if chg < -1 else "🟡 Neutral")
                note = "India imports ~85% crude — rising oil = CAD pressure"
            elif name == "Gold":
                impact = "🟡 Risk-Off" if chg > 0.5 else "🟡 Neutral"
                note = "Gold rising = safe haven demand"
            elif name == "DXY":
                impact = "🔴 EM Headwind" if chg > 0.3 else ("🟢 EM Tailwind" if chg < -0.3 else "🟡 Neutral")
                note = "Strong dollar = FII outflows from India"
            else:
                impact = "🟡 Monitor"
                note = "Global market indicator"
            impact_rows.append({"Indicator": name, "Price": f"{data['price']:,.2f}",
                                 "1D Change": f"{chg:+.2f}%", "India Impact": impact, "Note": note})

        impact_df = pd.DataFrame(impact_rows)
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

        # ── Correlation chart ─────────────────────────────────────────────────
        st.subheader("30-Day Performance Comparison")
        fig_global = go.Figure()
        colors_g = ["#6366f1", "#10b981", "#f59e0b", "#06b6d4", "#ec4899", "#84cc16"]
        plot_indices = ["S&P 500", "NASDAQ", "Nikkei 225", "Crude Oil", "Gold", "USD/INR"]
        for i, name in enumerate(plot_indices):
            if name in gd:
                try:
                    df_g = DataService.fetch_ohlcv(GLOBAL_INDICES[name], period="1mo")
                    if not df_g.empty:
                        normalized = (df_g["Close"] / df_g["Close"].iloc[0] - 1) * 100
                        fig_global.add_trace(go.Scatter(
                            x=df_g.index, y=normalized, mode="lines",
                            line=dict(color=colors_g[i % len(colors_g)], width=2),
                            name=name,
                        ))
                except Exception:
                    pass
        fig_global.add_hline(y=0, line_dash="dash", line_color="#475569")
        fig_global.update_layout(
            title="Normalized 30-Day Returns (%)",
            template="plotly_dark", paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
            yaxis_title="Return (%)", height=420, margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_global, use_container_width=True)

        # ── Sector impact matrix ──────────────────────────────────────────────
        st.subheader("Sector Sensitivity to Global Factors")
        sensitivity = {
            "Sector": ["IT", "Pharma", "Metals", "Energy", "Auto", "FMCG", "Finance", "Defence", "Infra"],
            "USD/INR ↑": ["🟢 +", "🟢 +", "🔴 -", "🔴 -", "🔴 -", "🟡 ~", "🔴 -", "🟡 ~", "🟡 ~"],
            "Crude ↑":   ["🟡 ~", "🟡 ~", "🟡 ~", "🟢 +", "🔴 -", "🔴 -", "🟡 ~", "🟡 ~", "🔴 -"],
            "VIX ↑":     ["🔴 -", "🟡 ~", "🔴 -", "🔴 -", "🔴 -", "🟢 +", "🔴 -", "🟢 +", "🟡 ~"],
            "S&P ↑":     ["🟢 +", "🟡 ~", "🟢 +", "🟡 ~", "🟢 +", "🟡 ~", "🟢 +", "🟡 ~", "🟡 ~"],
            "DXY ↑":     ["🟢 +", "🟢 +", "🔴 -", "🔴 -", "🔴 -", "🟡 ~", "🔴 -", "🟡 ~", "🟡 ~"],
        }
        st.dataframe(pd.DataFrame(sensitivity), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MONTE CARLO PAGE (preserved + capital-aware)
# ═══════════════════════════════════════════════════════════════════════════════
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
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Current Price", f"₹{r['current_price']:,.2f}")
        delta_exp = r["expected_price"] - r["current_price"]
        m2.metric("Expected Price", f"₹{r['expected_price']:,.2f}", f"{'+'if delta_exp>0 else ''}{delta_exp:,.0f}")
        m3.metric("Prob of Profit", f"{r['prob_profit']:.1%}")
        m4.metric("VaR 95%", f"{r['var_95']:.2%}")
        # Capital-aware expected P&L
        shares = int(capital * 0.05 / r["current_price"]) if r["current_price"] > 0 else 0
        exp_pnl = shares * (r["expected_price"] - r["current_price"])
        m5.metric("Expected P&L (5% alloc)", f"₹{exp_pnl:,.0f}", f"{shares} shares")

        m6, m7, m8, m9 = st.columns(4)
        m6.metric("P5 (Bear)", f"₹{r['p5']:,.2f}")
        m7.metric("P25", f"₹{r['p25']:,.2f}")
        m8.metric("P75", f"₹{r['p75']:,.2f}")
        m9.metric("P95 (Bull)", f"₹{r['p95']:,.2f}")

        fig = go.Figure()
        for path in r["sample_paths"][:50]:
            fig.add_trace(go.Scatter(y=path, mode="lines",
                                     line=dict(width=0.4, color="rgba(99,102,241,0.12)"),
                                     showlegend=False, hoverinfo="skip"))
        mean_path = np.mean(r["sample_paths"], axis=0).tolist()
        fig.add_trace(go.Scatter(y=mean_path, mode="lines",
                                  line=dict(width=2.5, color="#6366f1"), name="Mean Path"))
        fig.add_hline(y=r["current_price"], line_dash="dash", line_color="#f59e0b", annotation_text="Current")
        fig.update_layout(title=f"{sym} — {mc_sims:,} Simulations ({mc_days}d)",
                          template="plotly_dark", paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
                          xaxis_title="Days", yaxis_title="Price (₹)", height=450)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure(go.Histogram(x=r["final_distribution"], nbinsx=80,
                                       marker_color="rgba(99,102,241,0.6)"))
        fig2.add_vline(x=r["current_price"], line_dash="dash", line_color="#f59e0b", annotation_text="Current")
        fig2.add_vline(x=r["p5"], line_dash="dot", line_color="#ef4444", annotation_text="P5")
        fig2.add_vline(x=r["p95"], line_dash="dot", line_color="#10b981", annotation_text="P95")
        fig2.update_layout(title="Final Price Distribution", template="plotly_dark",
                            paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e", height=350)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO PAGE (preserved + capital-aware)
# ═══════════════════════════════════════════════════════════════════════════════
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
                    # Capital-aware allocation
                    w_df = pd.DataFrame(list(r[key]["weights"].items()), columns=["Stock", "Weight"])
                    w_df["Weight %"] = (w_df["Weight"] * 100).round(1)
                    w_df["Allocation ₹"] = (w_df["Weight"] * capital).apply(lambda x: f"₹{x:,.0f}")
                    st.dataframe(w_df[["Stock", "Weight %", "Allocation ₹"]],
                                 use_container_width=True, hide_index=True)

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
                               paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
                               xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=500)
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONS PRICING PAGE (preserved)
# ═══════════════════════════════════════════════════════════════════════════════
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
                # Capital-aware: how many contracts with current capital
                lot_size = 50  # typical NSE lot
                contracts = int(capital * 0.02 / (r[side]["price"] * lot_size)) if r[side]["price"] > 0 else 0
                st.caption(f"With ₹{capital:,.0f} capital (2% risk): ~{contracts} contracts (lot={lot_size})")
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
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
                          height=300, showlegend=False, margin=dict(t=40))
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# RISK ANALYSIS PAGE (preserved + capital-aware)
# ═══════════════════════════════════════════════════════════════════════════════
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

        # Capital-aware risk metrics
        var_val = abs(m['var_historical']['var'])
        capital_at_risk = round(capital * var_val)
        st.metric(
            "Capital at Risk (1-day, 95% VaR)",
            f"₹{capital_at_risk:,.0f}",
            f"{var_val:.2%} of ₹{capital:,.0f}",
        )

        dd = m["max_drawdown"]
        st.metric("Max Drawdown", f"{dd['max_drawdown']:.1%}",
                  f"{dd.get('peak_date','')} → {dd.get('trough_date','')}")

        stress_df = pd.DataFrame(r["stress"])
        stress_df["Shock"] = stress_df["shock"].apply(lambda x: f"{x:+.0%}")
        stress_df["Impact (Ann.)"] = stress_df["annualized_impact"].apply(lambda x: f"{x:+.1%}")
        stress_df["Capital Impact ₹"] = stress_df["shock"].apply(lambda x: f"₹{capital * x:,.0f}")
        st.dataframe(stress_df[["scenario", "Shock", "Impact (Ann.)", "Capital Impact ₹"]],
                     use_container_width=True, hide_index=True)

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
        fig.update_layout(template="plotly_dark", paper_bgcolor="#0a0f1e", plot_bgcolor="#0d1b2e",
                          height=500, margin=dict(t=20, b=40))
        fig.update_yaxes(title_text="Price (₹)", row=1, col=1)
        fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
        st.plotly_chart(fig, use_container_width=True)

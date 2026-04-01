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

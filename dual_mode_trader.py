"""
QuantSignal India — Dual-Mode Trading Platform v6.0
Intraday Mode  : Auto top-10 picks for any date, no manual filters needed
Delivery Mode  : Swing/Positional with 10/20/30/60-day or custom holding periods
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta, time as dtime
import sys, os, time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.engines.smart_intraday import SmartIntradayEngine, SYMBOL_TO_SECTOR
from backend.engines.delivery_engine import DeliveryEngine, HOLDING_PERIODS, SEASONAL_BIAS
from backend.engines.data_service import DataService
from backend.engines.prediction_engine import PredictionEngine
from backend.engines.stock_metadata import StockMetadata, GLOBAL_FACTORS, PENNY_MAX_PRICE
from backend.engines.intraday_engine import get_market_status
from backend.intraday_config import INTRADAY_STOCKS, SECTOR_GROUPS

st.set_page_config(
    page_title="QuantSignal India — Dual Mode",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif!important;}
.stApp{background:#04080f!important;}
.main .block-container{padding:0.5rem 1.5rem 2rem!important;max-width:100%!important;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#080f1e,#060b18)!important;border-right:1px solid #0f2040!important;}
section[data-testid="stSidebar"] label{color:#94a3b8!important;font-size:0.72rem!important;font-weight:600!important;text-transform:uppercase;letter-spacing:.06em;}
div[data-testid="stMetric"]{background:linear-gradient(135deg,#0a1628,#080f1e)!important;border:1px solid #0f2a4a!important;border-radius:12px!important;padding:14px 18px!important;}
div[data-testid="stMetricValue"]{color:#f1f5f9!important;font-size:1.3rem!important;font-weight:800!important;}
div[data-testid="stMetricLabel"]{color:#475569!important;font-size:.68rem!important;text-transform:uppercase!important;letter-spacing:.07em!important;}
.stButton>button{background:linear-gradient(135deg,#1e40af,#2563eb)!important;color:#fff!important;border:none!important;border-radius:10px!important;font-weight:700!important;font-size:.88rem!important;box-shadow:0 4px 14px rgba(37,99,235,.4)!important;transition:all .2s!important;width:100%!important;}
.stButton>button:hover{background:linear-gradient(135deg,#1d4ed8,#3b82f6)!important;transform:translateY(-2px)!important;}
.stTabs [data-baseweb="tab-list"]{background:#080f1e!important;border-radius:12px!important;padding:5px!important;border:1px solid #0f2040!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:#475569!important;border-radius:9px!important;font-weight:600!important;font-size:.82rem!important;padding:8px 14px!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,#1e40af,#2563eb)!important;color:#fff!important;}
h1{color:#f8fafc!important;font-weight:900!important;font-size:1.8rem!important;}
h2,h3{color:#e2e8f0!important;font-weight:700!important;}
p,li{color:#94a3b8!important;}
hr{border-color:#0f2040!important;margin:.8rem 0!important;}
.mode-card{background:linear-gradient(135deg,#0a1628,#080f1e);border:2px solid #0f2a4a;border-radius:16px;padding:24px;cursor:pointer;transition:border-color .2s,transform .15s;}
.mode-card:hover{border-color:#2563eb;transform:translateY(-2px);}
.mode-card.active{border-color:#2563eb;background:linear-gradient(135deg,#0d1f3c,#0a1628);}
.trade-card{background:linear-gradient(135deg,#0a1628,#080f1e);border:1px solid #0f2a4a;border-radius:14px;padding:20px;margin-bottom:10px;}
.stat-card{background:#0a1628;border:1px solid #0f2040;border-radius:12px;padding:14px;text-align:center;}
.badge-buy{display:inline-block;background:linear-gradient(135deg,#065f46,#059669);color:#ecfdf5;padding:4px 14px;border-radius:7px;font-weight:800;font-size:.85rem;}
.badge-sell{display:inline-block;background:linear-gradient(135deg,#7f1d1d,#dc2626);color:#fef2f2;padding:4px 14px;border-radius:7px;font-weight:800;font-size:.85rem;}
.badge-watch{display:inline-block;background:linear-gradient(135deg,#78350f,#d97706);color:#fffbeb;padding:4px 14px;border-radius:7px;font-weight:800;font-size:.85rem;}
.badge-sector{display:inline-block;background:rgba(99,102,241,.12);color:#a5b4fc;padding:3px 10px;border-radius:6px;font-size:.72rem;border:1px solid rgba(99,102,241,.25);margin-right:4px;}
.reason-chip{display:inline-block;background:rgba(6,182,212,.08);color:#67e8f9;padding:3px 10px;border-radius:6px;margin:2px;font-size:.7rem;border:1px solid rgba(6,182,212,.2);}
.factor-pos{display:inline-block;background:rgba(5,150,105,.1);color:#6ee7b7;padding:4px 12px;border-radius:6px;margin:2px;font-size:.75rem;border:1px solid rgba(5,150,105,.2);}
.factor-neg{display:inline-block;background:rgba(220,38,38,.1);color:#fca5a5;padding:4px 12px;border-radius:6px;margin:2px;font-size:.75rem;border:1px solid rgba(220,38,38,.2);}
.live-dot{display:inline-block;width:8px;height:8px;background:#10b981;border-radius:50%;animation:pulse 1.5s infinite;margin-right:6px;vertical-align:middle;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.5;transform:scale(1.4)}}
.intraday-header{background:linear-gradient(135deg,#0d1f3c,#0a1628);border:2px solid #1e40af;border-radius:16px;padding:20px;margin-bottom:16px;}
.delivery-header{background:linear-gradient(135deg,#0d2818,#0a1f12);border:2px solid #065f46;border-radius:16px;padding:20px;margin-bottom:16px;}
.rank-badge{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;background:linear-gradient(135deg,#1e40af,#2563eb);color:#fff;border-radius:50%;font-weight:900;font-size:.82rem;}
::-webkit-scrollbar{width:6px;height:6px;}
::-webkit-scrollbar-track{background:#04080f;}
::-webkit-scrollbar-thumb{background:#1e3a5f;border-radius:3px;}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
ALL_SYMBOLS_CLEAN = sorted(set(s.replace(".NS", "") for s in INTRADAY_STOCKS))

def market_open() -> bool:
    now = datetime.now()
    return now.weekday() < 5 and dtime(9, 15) <= now.time() <= dtime(15, 30)

def _layout(title="", height=450):
    return dict(
        template="plotly_dark", paper_bgcolor="#04080f", plot_bgcolor="#080f1e",
        height=height, margin=dict(t=35 if title else 10, b=10, l=50, r=20),
        font=dict(family="Inter", size=11, color="#94a3b8"),
        legend=dict(orientation="h", y=1.02, bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(gridcolor="#0a1628", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#0a1628", showgrid=True, zeroline=False),
        title=dict(text=title, font=dict(size=13, color="#e2e8f0")) if title else None,
    )

def signal_badge(sig: str) -> str:
    if "BUY" in sig.upper():
        return f'<span class="badge-buy">{sig}</span>'
    if "SELL" in sig.upper() or "AVOID" in sig.upper():
        return f'<span class="badge-sell">{sig}</span>'
    return f'<span class="badge-watch">{sig}</span>'

def score_bar(score: float, width: int = 120) -> str:
    pct = int(score * 100)
    color = "#10b981" if score >= 0.65 else ("#f59e0b" if score >= 0.50 else "#ef4444")
    return (f'<div style="background:#0f2040;border-radius:4px;width:{width}px;height:8px;overflow:hidden;">'
            f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;"></div></div>'
            f'<span style="color:{color};font-size:.72rem;font-weight:700;">{pct}%</span>')

# ── Sidebar ───────────────────────────────────────────────────────────────────
is_open = market_open()
mkt_status = get_market_status()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:10px 0 16px;'>
        <div style='font-size:2rem;'>⚡</div>
        <div style='font-size:1.2rem;font-weight:900;color:#f1f5f9;'>QuantSignal India</div>
        <div style='font-size:.62rem;color:#334155;letter-spacing:.12em;text-transform:uppercase;'>DUAL-MODE ENGINE v6.0</div>
    </div>""", unsafe_allow_html=True)

    dot = '<span class="live-dot"></span>' if is_open else "🔴 "
    mkt_color = "#10b981" if is_open else "#ef4444"
    st.markdown(f'<div style="color:{mkt_color};font-weight:700;font-size:.82rem;">{dot}{mkt_status}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#1e3a5f;font-size:.65rem;">{datetime.now().strftime("%d %b %Y  %I:%M %p")}</div>', unsafe_allow_html=True)

    st.markdown("---")
    # ── Mode selector ─────────────────────────────────────────────────────────
    st.markdown('<div style="color:#475569;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">🎯 Trading Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "mode",
        ["📡 Intraday", "📦 Delivery"],
        label_visibility="collapsed",
        horizontal=True,
    )

    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">💰 Capital & Risk</div>', unsafe_allow_html=True)
    capital = st.number_input("Capital (₹)", value=50_000, min_value=1_000, max_value=10_000_000, step=5_000)
    risk_pct = st.slider("Risk per Trade (%)", 0.5, 5.0, 2.0, 0.5) / 100

    st.markdown("---")
    # ── Mode-specific sidebar controls ───────────────────────────────────────
    if "Intraday" in mode:
        st.markdown('<div style="color:#475569;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">📅 Intraday Settings</div>', unsafe_allow_html=True)
        trade_date = st.date_input("Trading Date", value=date.today(), max_value=date.today())
        intra_top_n = st.slider("Top N Stocks", 5, 20, 10, 1)
        intra_sector_filter = st.multiselect("Sector Filter (optional)", list(SECTOR_GROUPS.keys()), default=[])
        intra_stock_filter = st.multiselect("Specific Stocks (optional)", ALL_SYMBOLS_CLEAN, default=[])
        auto_refresh = st.checkbox("🔄 Auto-refresh (5 min)", value=False)
    else:
        st.markdown('<div style="color:#475569;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;margin-bottom:8px;">📦 Delivery Settings</div>', unsafe_allow_html=True)
        holding_choice = st.select_slider(
            "Holding Period",
            options=["10 Days", "20 Days", "30 Days", "60 Days", "Custom"],
            value="30 Days",
        )
        if holding_choice == "Custom":
            custom_days = st.number_input("Custom Days", min_value=5, max_value=365, value=45, step=5)
            holding_days = custom_days
        else:
            holding_days = HOLDING_PERIODS[holding_choice]

        delivery_top_n = st.slider("Top N Stocks", 5, 20, 10, 1)
        delivery_sector_filter = st.multiselect("Sector Filter (optional)", list(SECTOR_GROUPS.keys()), default=[])
        delivery_stock_filter = st.multiselect("Specific Stocks (optional)", ALL_SYMBOLS_CLEAN, default=[])

    st.markdown("---")
    st.markdown(f'<div style="color:#1e3a5f;font-size:.65rem;">Universe: <b style="color:#3b82f6;">{len(INTRADAY_STOCKS)}</b> stocks · 14 sectors</div>', unsafe_allow_html=True)
    st.caption("Not financial advice. Paper trade first.")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
mode_color = "#2563eb" if "Intraday" in mode else "#059669"
mode_icon = "📡" if "Intraday" in mode else "📦"
mode_label = "Intraday Trading" if "Intraday" in mode else "Delivery / Swing"

live_badge = ('<span style="background:#065f46;color:#6ee7b7;padding:2px 9px;border-radius:5px;font-size:.68rem;font-weight:800;">● LIVE</span>'
              if is_open else
              '<span style="background:#7f1d1d;color:#fca5a5;padding:2px 9px;border-radius:5px;font-size:.68rem;font-weight:800;">● CLOSED</span>')

st.markdown(f"""
<div style='display:flex;align-items:center;gap:12px;margin-bottom:4px;flex-wrap:wrap;'>
    <span style='font-size:1.7rem;font-weight:900;color:#f8fafc;'>{mode_icon} QuantSignal — {mode_label}</span>
    {live_badge}
    <span style='background:{mode_color}22;color:{mode_color};padding:2px 10px;border-radius:6px;font-size:.72rem;font-weight:700;border:1px solid {mode_color}44;'>
        {"AUTO TOP-10" if "Intraday" in mode else f"HOLDING: {holding_choice if 'Delivery' in mode else ''}"}
    </span>
</div>
<div style='color:#334155;font-size:.78rem;'>
    Capital: <b style='color:#3b82f6;'>₹{capital:,.0f}</b> &nbsp;·&nbsp;
    Risk/Trade: <b style='color:#f59e0b;'>{risk_pct*100:.1f}%</b> &nbsp;·&nbsp;
    {datetime.now().strftime("%d %b %Y %I:%M %p")}
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# ██████████████████  INTRADAY MODE  ██████████████████████████████████████████
# ══════════════════════════════════════════════════════════════════════════════
if "Intraday" in mode:

    tab_live, tab_chart, tab_sector = st.tabs(["🏆 Top 10 Picks", "📊 Charts & Analysis", "🌍 Sector Themes"])

    # ── Controls row ─────────────────────────────────────────────────────────
    with tab_live:
        st.markdown(f"""
        <div class="intraday-header">
            <div style='font-size:1.1rem;font-weight:900;color:#93c5fd;'>📡 Smart Intraday Scanner</div>
            <div style='color:#1e40af;font-size:.82rem;margin-top:4px;'>
                Auto-selects top 10 stocks for <b style='color:#60a5fa;'>{trade_date.strftime("%d %B %Y")}</b>
                using VWAP · EMA · RSI · Supertrend · MACD · ML · Volume · Geopolitical themes
            </div>
        </div>""", unsafe_allow_html=True)

        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        with col_btn1:
            scan_btn = st.button(f"🚀 Auto-Scan Top 10 for {trade_date.strftime('%d %b')}", type="primary", use_container_width=True)
        with col_btn2:
            refresh_btn = st.button("🔄 Refresh", use_container_width=True)
        with col_btn3:
            clear_btn = st.button("🗑️ Clear", use_container_width=True)

        if clear_btn:
            for k in ["intra_result"]:
                st.session_state.pop(k, None)
            st.rerun()

        if scan_btn or refresh_btn:
            universe_intra = None
            if intra_stock_filter:
                universe_intra = [f"{s}.NS" for s in intra_stock_filter]
            elif intra_sector_filter:
                universe_intra = []
                for s in intra_sector_filter:
                    universe_intra.extend(SECTOR_GROUPS.get(s, []))
                universe_intra = list(dict.fromkeys(universe_intra))

            engine = SmartIntradayEngine(capital=capital)
            prog = st.progress(0, text="🚀 Scanning stocks with combined signals...")

            # Run in background with progress updates
            import threading
            result_holder = {}

            def _run():
                result_holder["data"] = engine.get_top10(
                    trade_date=trade_date,
                    universe=universe_intra,
                )

            t = threading.Thread(target=_run)
            t.start()
            step = 0
            while t.is_alive():
                step = min(step + 2, 95)
                prog.progress(step, text=f"📡 Scanning {len(universe_intra or INTRADAY_STOCKS)} stocks...")
                _time.sleep(0.3)
            t.join()
            prog.progress(100, text="✅ Done!")
            _time.sleep(0.3)
            prog.empty()

            st.session_state["intra_result"] = result_holder.get("data", {})

        # ── Results ───────────────────────────────────────────────────────────
        if "intra_result" in st.session_state and st.session_state["intra_result"]:
            res = st.session_state["intra_result"]
            top10 = res.get("top10", [])

            if not top10:
                st.warning("No strong signals found. Market may be choppy or data unavailable.")
            else:
                # Summary metrics
                m1, m2, m3, m4, m5, m6 = st.columns(6)
                m1.metric("Scanned", res["scanned"])
                m2.metric("Candidates", res["candidates_found"])
                m3.metric("Top Picks", len(top10))
                m4.metric("Best Score", f"{top10[0]['composite']:.0%}", top10[0]["symbol"])
                m5.metric("Best R:R", f"{top10[0]['risk_reward']}x")
                m6.metric("Sectors", len(res["sector_distribution"]))

                st.markdown(f"<div style='color:#334155;font-size:.72rem;margin-bottom:12px;'>Generated: {res['generated_at']}</div>", unsafe_allow_html=True)
                st.markdown("---")

                # ── Top 10 cards ──────────────────────────────────────────────
                st.markdown("### 🏆 Top 10 Intraday Picks")
                for i, t in enumerate(top10):
                    rank_color = ["#f59e0b", "#94a3b8", "#cd7c2f"] + ["#475569"] * 7
                    sig_color = "#10b981" if t["signal"] == "BUY" else "#f59e0b"
                    st.markdown(f"""
                    <div class="trade-card">
                        <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px;'>
                            <span class="rank-badge" style="background:{rank_color[i]};">#{i+1}</span>
                            <span style='font-size:1.2rem;font-weight:900;color:#f1f5f9;'>{t["symbol"]}</span>
                            <span class="badge-sector">{t["sector"]}</span>
                            <span style='background:{sig_color}22;color:{sig_color};padding:3px 12px;border-radius:7px;font-weight:800;font-size:.85rem;'>{t["signal"]}</span>
                            {"<span style='background:#4c1d9522;color:#c4b5fd;padding:3px 8px;border-radius:5px;font-size:.7rem;font-weight:700;'>🪙 PENNY</span>" if t["is_penny"] else ""}
                        </div>
                        <div style='display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin-bottom:10px;'>
                            <div><div style='color:#475569;font-size:.65rem;text-transform:uppercase;'>Entry</div><div style='color:#f1f5f9;font-weight:700;'>₹{t["price"]:,.2f}</div></div>
                            <div><div style='color:#475569;font-size:.65rem;text-transform:uppercase;'>Target 1</div><div style='color:#10b981;font-weight:700;'>₹{t["target1"]:,.2f}</div></div>
                            <div><div style='color:#475569;font-size:.65rem;text-transform:uppercase;'>Target 2</div><div style='color:#06b6d4;font-weight:700;'>₹{t["target2"]:,.2f}</div></div>
                            <div><div style='color:#475569;font-size:.65rem;text-transform:uppercase;'>Stop Loss</div><div style='color:#ef4444;font-weight:700;'>₹{t["stop_loss"]:,.2f}</div></div>
                            <div><div style='color:#475569;font-size:.65rem;text-transform:uppercase;'>R:R</div><div style='color:#f59e0b;font-weight:700;'>{t["risk_reward"]}x</div></div>
                            <div><div style='color:#475569;font-size:.65rem;text-transform:uppercase;'>Qty</div><div style='color:#94a3b8;font-weight:700;'>{t["qty"]}</div></div>
                        </div>
                        <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:10px;'>
                            <div style='color:#64748b;font-size:.72rem;'>Score: {score_bar(t["composite"])}</div>
                            <div style='color:#64748b;font-size:.72rem;'>RSI: <b style='color:#a78bfa;'>{t["rsi"]:.0f}</b></div>
                            <div style='color:#64748b;font-size:.72rem;'>Vol: <b style='color:#06b6d4;'>{t["vol_ratio"]:.1f}x</b></div>
                            <div style='color:#64748b;font-size:.72rem;'>ST: <b style='color:{"#10b981" if t["supertrend"]=="BUY" else "#ef4444"};'>{t["supertrend"]}</b></div>
                        </div>
                        <div>{''.join(f'<span class="reason-chip">{r}</span>' for r in t["reasons"][:5])}</div>
                        <div style='margin-top:8px;color:#334155;font-size:.72rem;'>
                            Invest: <b style='color:#3b82f6;'>₹{t["invested"]:,.0f}</b> &nbsp;·&nbsp;
                            Max Profit: <b style='color:#10b981;'>+₹{t["potential_profit"]:,.0f}</b> &nbsp;·&nbsp;
                            Max Loss: <b style='color:#ef4444;'>-₹{t["potential_loss"]:,.0f}</b>
                        </div>
                    </div>""", unsafe_allow_html=True)

                # ── Summary table ─────────────────────────────────────────────
                st.markdown("---")
                st.markdown("### 📋 Quick Reference Table")
                tbl_data = [{
                    "#": i + 1,
                    "Stock": t["symbol"],
                    "Sector": t["sector"],
                    "Entry ₹": f"₹{t['price']:,.2f}",
                    "Target1 ₹": f"₹{t['target1']:,.2f}",
                    "Target2 ₹": f"₹{t['target2']:,.2f}",
                    "SL ₹": f"₹{t['stop_loss']:,.2f}",
                    "Score": f"{t['composite']:.0%}",
                    "R:R": f"{t['risk_reward']}x",
                    "RSI": t["rsi"],
                    "Vol": f"{t['vol_ratio']:.1f}x",
                    "ST": t["supertrend"],
                    "Signal": t["signal"],
                } for i, t in enumerate(top10)]
                st.dataframe(pd.DataFrame(tbl_data), use_container_width=True, hide_index=True,
                             height=min(450, 55 + 38 * len(tbl_data)))

        else:
            # Welcome state
            st.markdown(f"""
            <div style='text-align:center;padding:50px 20px;'>
                <div style='font-size:3rem;margin-bottom:12px;'>📡</div>
                <div style='font-size:1.5rem;font-weight:900;color:#f1f5f9;margin-bottom:8px;'>Smart Intraday Scanner</div>
                <div style='color:#334155;font-size:.9rem;margin-bottom:24px;'>
                    Click <b style='color:#3b82f6;'>Auto-Scan Top 10</b> to get the best intraday picks for
                    <b style='color:#60a5fa;'>{trade_date.strftime("%d %B %Y")}</b>
                </div>
                <div style='color:#1e3a5f;font-size:.78rem;'>
                    Combines: VWAP · EMA crossover · RSI · Supertrend · MACD · ML prediction · Volume spike · Geopolitical themes
                </div>
            </div>""", unsafe_allow_html=True)

    # ── Charts tab ────────────────────────────────────────────────────────────
    with tab_chart:
        st.markdown("### 📊 Intraday Chart Analysis")
        if "intra_result" in st.session_state and st.session_state["intra_result"].get("top10"):
            top10 = st.session_state["intra_result"]["top10"]
            chart_sym = st.selectbox("Select Stock", [t["symbol"] for t in top10], key="intra_chart_sym")
            sel = next((t for t in top10 if t["symbol"] == chart_sym), None)
            if sel:
                with st.spinner(f"Loading chart for {chart_sym}..."):
                    from backend.engines.intraday_engine import IntradayEngine as _IE
                    _ie = _IE(capital=capital)
                    df_c = _ie.fetch_intraday(f"{chart_sym}.NS", period="5d", interval="5m")
                    if not df_c.empty:
                        df_c = _ie.add_indicators(df_c)
                        dc = df_c.tail(120)
                        fig = make_subplots(rows=3, cols=1, row_heights=[.60, .20, .20],
                                            shared_xaxes=True, vertical_spacing=.02)
                        fig.add_trace(go.Candlestick(
                            x=dc.index, open=dc["Open"], high=dc["High"], low=dc["Low"], close=dc["Close"],
                            increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
                            decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
                            name="Price",
                        ), row=1, col=1)
                        for col_name, color, dash, lbl in [
                            ("VWAP", "#f59e0b", "dot", "VWAP"),
                            ("EMA9", "#6366f1", "solid", "EMA9"),
                            ("EMA21", "#ec4899", "solid", "EMA21"),
                        ]:
                            if col_name in dc.columns:
                                fig.add_trace(go.Scatter(x=dc.index, y=dc[col_name], mode="lines",
                                    line=dict(color=color, width=1.2, dash=dash), name=lbl), row=1, col=1)
                        fig.add_hline(y=sel["price"], line_color="#6366f1", line_width=1.5, annotation_text="ENTRY", row=1, col=1)
                        fig.add_hline(y=sel["target1"], line_color="#10b981", line_dash="dash", annotation_text="T1", row=1, col=1)
                        fig.add_hline(y=sel["target2"], line_color="#06b6d4", line_dash="dash", annotation_text="T2", row=1, col=1)
                        fig.add_hline(y=sel["stop_loss"], line_color="#ef4444", line_dash="dash", annotation_text="SL", row=1, col=1)
                        if "RSI" in dc.columns:
                            fig.add_trace(go.Scatter(x=dc.index, y=dc["RSI"], mode="lines",
                                line=dict(color="#a78bfa", width=1.5), name="RSI"), row=2, col=1)
                            fig.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=.8, row=2, col=1)
                            fig.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=.8, row=2, col=1)
                        vcols = ["#10b981" if c >= o else "#ef4444" for c, o in zip(dc["Close"], dc["Open"])]
                        fig.add_trace(go.Bar(x=dc.index, y=dc["Volume"], marker_color=vcols, name="Vol", opacity=.8), row=3, col=1)
                        fig.update_layout(**_layout(f"{chart_sym} — 5m Intraday", height=520))
                        fig.update_layout(xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                        # Score breakdown
                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc1.metric("Composite Score", f"{sel['composite']:.0%}")
                        sc2.metric("Intraday TA", f"{sel['intra_score']:.0%}")
                        sc3.metric("ML Confidence", f"{sel['ml_confidence']:.0%}")
                        sc4.metric("Geo Score", f"{sel['geo_score']:.0%}")
        else:
            st.info("Run the scan first to see charts.")

    # ── Sector themes tab ─────────────────────────────────────────────────────
    with tab_sector:
        st.markdown("### 🌍 Active Geopolitical & Macro Themes")
        month = datetime.now().month
        seasonal = SEASONAL_BIAS.get(month, {"bias": 0, "note": ""})
        bias_color = "#10b981" if seasonal["bias"] > 0 else "#ef4444"
        st.markdown(f"""
        <div style='background:#0a1628;border:1px solid #0f2040;border-radius:12px;padding:14px;margin-bottom:16px;'>
            <span style='color:#475569;font-size:.72rem;font-weight:700;text-transform:uppercase;'>📅 Seasonal Bias — {datetime.now().strftime("%B")}</span>
            <div style='color:{bias_color};font-weight:700;font-size:1rem;margin-top:4px;'>{seasonal["bias"]*100:+.0f}% monthly bias</div>
            <div style='color:#64748b;font-size:.78rem;margin-top:2px;'>{seasonal["note"]}</div>
        </div>""", unsafe_allow_html=True)

        cols_theme = st.columns(2)
        for i, (sector, data) in enumerate(GLOBAL_FACTORS.items()):
            with cols_theme[i % 2]:
                theme_color = "#10b981" if "STRONG" in data["theme"] else ("#f59e0b" if "MODERATE" in data["theme"] else "#94a3b8")
                st.markdown(f"""
                <div style='background:#0a1628;border:1px solid #0f2040;border-radius:12px;padding:14px;margin-bottom:10px;'>
                    <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;'>
                        <span style='font-weight:700;color:#e2e8f0;font-size:.9rem;'>{sector}</span>
                        <span style='color:{theme_color};font-size:.72rem;font-weight:700;'>{data["theme"]}</span>
                    </div>
                    <div>{''.join(f'<span class="factor-pos">✅ {f}</span>' for f in data["positive"][:2])}</div>
                    <div style='margin-top:4px;'>{''.join(f'<span class="factor-neg">⚠️ {f}</span>' for f in data["negative"][:1])}</div>
                </div>""", unsafe_allow_html=True)

        if auto_refresh and "intra_result" in st.session_state:
            _time.sleep(300)
            st.rerun()

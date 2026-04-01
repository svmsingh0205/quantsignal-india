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

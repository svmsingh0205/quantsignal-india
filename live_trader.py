"""
QuantSignal India — Production Trading Platform v5.0
Full-screen responsive UI | 500+ stocks | ML predictions | Penny stock reports
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, time as dtime, timedelta
import sys, os, time as _time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.engines.intraday_engine import IntradayEngine, get_market_status
from backend.engines.data_service import DataService
from backend.engines.prediction_engine import PredictionEngine
from backend.engines.stock_metadata import StockMetadata, GLOBAL_FACTORS, PENNY_MAX_PRICE
from backend.intraday_config import INTRADAY_STOCKS, SECTOR_GROUPS

st.set_page_config(
    page_title="QuantSignal India",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── FULL PRODUCTION CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: #04080f !important; }
.main .block-container {
    padding: 0.5rem 1.5rem 2rem 1.5rem !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #080f1e 0%, #060b18 100%) !important;
    border-right: 1px solid #0f2040 !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1rem !important; }
section[data-testid="stSidebar"] label { color: #94a3b8 !important; font-size: 0.75rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: 0.06em; }
section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span { color: #cbd5e1 !important; }

/* ── Metrics ── */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #0a1628 0%, #080f1e 100%) !important;
    border: 1px solid #0f2a4a !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    transition: border-color 0.2s, transform 0.15s !important;
}
div[data-testid="stMetric"]:hover { border-color: #1d4ed8 !important; transform: translateY(-1px) !important; }
div[data-testid="stMetricLabel"] { color: #475569 !important; font-size: 0.68rem !important; text-transform: uppercase !important; letter-spacing: 0.07em !important; font-weight: 600 !important; }
div[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.35rem !important; font-weight: 800 !important; line-height: 1.2 !important; }
div[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 0.88rem !important; padding: 0.55rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(37,99,235,0.4) !important;
    transition: all 0.2s !important; width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    box-shadow: 0 6px 20px rgba(37,99,235,0.55) !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #080f1e !important; border-radius: 12px !important;
    padding: 5px !important; gap: 3px !important;
    border: 1px solid #0f2040 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #475569 !important;
    border-radius: 9px !important; font-weight: 600 !important;
    font-size: 0.82rem !important; padding: 8px 14px !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1e40af, #2563eb) !important;
    color: #fff !important; box-shadow: 0 3px 10px rgba(37,99,235,0.4) !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #94a3b8 !important; }

/* ── Progress ── */
.stProgress > div > div { background: linear-gradient(90deg, #1d4ed8, #06b6d4) !important; border-radius: 4px !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden !important; border: 1px solid #0f2040 !important; }
.stDataFrame thead th { background: #080f1e !important; color: #94a3b8 !important; font-size: 0.72rem !important; text-transform: uppercase !important; }

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #0a1628 !important; border: 1px solid #0f2040 !important;
    border-radius: 10px !important; color: #e2e8f0 !important;
    font-weight: 600 !important;
}
.streamlit-expanderContent { background: #060b18 !important; border: 1px solid #0f2040 !important; border-top: none !important; border-radius: 0 0 10px 10px !important; }

/* ── Inputs ── */
.stSelectbox > div > div, .stMultiSelect > div > div {
    background: #0a1628 !important; border: 1px solid #0f2040 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
}
.stSlider > div > div > div { background: #1d4ed8 !important; }
.stNumberInput > div > div > input { background: #0a1628 !important; border: 1px solid #0f2040 !important; color: #e2e8f0 !important; border-radius: 8px !important; }
.stDateInput > div > div > input { background: #0a1628 !important; border: 1px solid #0f2040 !important; color: #e2e8f0 !important; border-radius: 8px !important; }

/* ── Text ── */
h1 { color: #f8fafc !important; font-weight: 900 !important; font-size: 1.8rem !important; letter-spacing: -0.02em !important; }
h2 { color: #e2e8f0 !important; font-weight: 800 !important; font-size: 1.3rem !important; }
h3 { color: #cbd5e1 !important; font-weight: 700 !important; font-size: 1.05rem !important; }
p, li { color: #94a3b8 !important; }
hr { border-color: #0f2040 !important; margin: 0.8rem 0 !important; }

/* ── Custom badges ── */
.badge-buy { display:inline-block; background:linear-gradient(135deg,#065f46,#059669); color:#ecfdf5; padding:5px 16px; border-radius:8px; font-weight:800; font-size:0.9rem; letter-spacing:0.03em; }
.badge-sell { display:inline-block; background:linear-gradient(135deg,#7f1d1d,#dc2626); color:#fef2f2; padding:5px 16px; border-radius:8px; font-weight:800; font-size:0.9rem; }
.badge-watch { display:inline-block; background:linear-gradient(135deg,#78350f,#d97706); color:#fffbeb; padding:5px 16px; border-radius:8px; font-weight:800; font-size:0.9rem; }
.badge-penny { display:inline-block; background:linear-gradient(135deg,#4c1d95,#7c3aed); color:#f5f3ff; padding:3px 10px; border-radius:6px; font-weight:700; font-size:0.72rem; }
.badge-sector { display:inline-block; background:rgba(99,102,241,0.12); color:#a5b4fc; padding:3px 10px; border-radius:6px; font-size:0.72rem; border:1px solid rgba(99,102,241,0.25); margin-right:4px; }
.reason-chip { display:inline-block; background:rgba(6,182,212,0.08); color:#67e8f9; padding:3px 10px; border-radius:6px; margin:2px; font-size:0.7rem; border:1px solid rgba(6,182,212,0.2); }
.factor-pos { display:inline-block; background:rgba(5,150,105,0.1); color:#6ee7b7; padding:4px 12px; border-radius:6px; margin:2px; font-size:0.75rem; border:1px solid rgba(5,150,105,0.2); }
.factor-neg { display:inline-block; background:rgba(220,38,38,0.1); color:#fca5a5; padding:4px 12px; border-radius:6px; margin:2px; font-size:0.75rem; border:1px solid rgba(220,38,38,0.2); }

/* ── Cards ── */
.trade-card { background:linear-gradient(135deg,#0a1628,#080f1e); border:1px solid #0f2a4a; border-radius:16px; padding:22px; transition:border-color 0.2s; }
.trade-card:hover { border-color: #1d4ed8; }
.stat-card { background:#0a1628; border:1px solid #0f2040; border-radius:12px; padding:16px; text-align:center; }
.stat-num { font-size:1.8rem; font-weight:900; color:#3b82f6; }
.stat-label { font-size:0.68rem; color:#475569; text-transform:uppercase; letter-spacing:0.07em; margin-top:4px; }
.price-big { font-size:2.6rem; font-weight:900; color:#10b981; line-height:1; }
.stock-title { font-size:1.5rem; font-weight:900; color:#f1f5f9; }
.report-card { background:#0a1628; border:1px solid #0f2040; border-radius:14px; padding:20px; margin-bottom:12px; }
.penny-header { background:linear-gradient(135deg,#2e1065,#4c1d95); border:1px solid #6d28d9; border-radius:14px; padding:16px 20px; margin-bottom:16px; }

/* ── Live dot ── */
.live-dot { display:inline-block; width:8px; height:8px; background:#10b981; border-radius:50%; animation:pulse 1.5s infinite; margin-right:6px; vertical-align:middle; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.4)} }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #04080f; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────────────────────
SYMBOL_TO_SECTOR: dict[str, str] = {}
for _sec, _syms in SECTOR_GROUPS.items():
    for _s in _syms:
        SYMBOL_TO_SECTOR[_s.replace(".NS", "")] = _sec

ALL_SYMBOLS_CLEAN = sorted(set(s.replace(".NS", "") for s in INTRADAY_STOCKS))

def market_open() -> bool:
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)

def time_to_close() -> str:
    now = datetime.now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now >= close:
        return "Closed"
    diff = close - now
    h, m = divmod(diff.seconds // 60, 60)
    return f"{h}h {m}m left" if h else f"{m}m left"

def get_universe(sector_filter: list, stock_filter: list, penny_only: bool) -> list[str]:
    if stock_filter:
        return [f"{s}.NS" for s in stock_filter]
    if sector_filter:
        out = []
        for s in sector_filter:
            out.extend(SECTOR_GROUPS.get(s, []))
        syms = list(dict.fromkeys(out))
    else:
        syms = INTRADAY_STOCKS[:]
    if penny_only:
        # We'll filter by price during scan, but pre-filter known penny stocks
        penny_known = [s for s in syms if any(p in s for p in [
            "YESBANK","IDEA","SUZLON","RPOWER","JPPOWER","UCOBANK",
            "MAHABANK","PSB","CENTRALBK","BANKINDIA","NHPC","SJVN",
            "IRFC","RECLTD","PFC","NBCC","BHEL","NATIONALUM","HINDCOPPER",
        ])]
        return penny_known if penny_known else syms
    return syms

def _chart_layout(title="", height=450):
    return dict(
        template="plotly_dark", paper_bgcolor="#04080f", plot_bgcolor="#080f1e",
        height=height, margin=dict(t=30 if title else 10, b=10, l=50, r=20),
        font=dict(family="Inter", size=11, color="#94a3b8"),
        legend=dict(orientation="h", y=1.02, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#0a1628", showgrid=True, zeroline=False),
        yaxis=dict(gridcolor="#0a1628", showgrid=True, zeroline=False),
        title=dict(text=title, font=dict(size=13, color="#e2e8f0")) if title else None,
    )

# ── PARALLEL SCAN ─────────────────────────────────────────────────────────────
def _scan_one(args):
    sym, capital, max_price, min_conf, penny_only = args
    try:
        engine = IntradayEngine(capital=capital)
        df = engine.fetch_intraday(sym, period="5d", interval="5m")
        if df.empty or len(df) < 30:
            return None
        price = float(df["Close"].iloc[-1])
        if price > max_price:
            return None
        if penny_only and price > PENNY_MAX_PRICE:
            return None
        df = engine.add_indicators(df)
        sc = engine.score_stock(df)
        atr = max(sc["atr"], price * 0.005)
        entry = round(price, 2)
        sl = round(max(price - 1.5 * atr, price * 0.93), 2)
        t1 = round(price + 2.0 * atr, 2)
        t2 = round(price + 3.0 * atr, 2)
        risk = entry - sl
        rr = round((t1 - entry) / risk, 2) if risk > 0 else 0
        qty = max(1, int(capital // price))
        clean = sym.replace(".NS", "")
        sector = SYMBOL_TO_SECTOR.get(clean, "Other")
        report = StockMetadata.generate_report(
            symbol=clean, price=price, sector=sector,
            confidence=sc["score"], direction="UP" if sc["score"] >= 0.55 else "NEUTRAL",
            predicted_return=0, volatility=0.2, rsi=sc["rsi"],
            entry=entry, target=t1, stop_loss=sl, mode="intraday",
        )
        return {
            "symbol": clean, "yf_symbol": sym, "sector": sector,
            "price": entry, "qty": qty, "invested": round(qty * price, 2),
            "target_1": t1, "target_2": t2, "stop_loss": sl,
            "confidence": sc["score"], "risk_reward": rr,
            "profit": round(qty * (t1 - entry), 2),
            "loss": round(qty * (entry - sl), 2),
            "rsi": sc["rsi"], "vwap": sc["vwap"],
            "vol_ratio": sc["vol_ratio"], "supertrend": sc["supertrend"],
            "reasons": sc["reasons"],
            "signal": "BUY" if sc["score"] >= min_conf else "WATCH",
            "is_penny": price <= PENNY_MAX_PRICE,
            "risk_level": report["risk_level"],
            "holding": report["holding_duration"],
            "df": df,
        }
    except Exception:
        return None

def run_scan(universe, capital, max_price, min_conf, penny_only=False):
    prog = st.progress(0, text="🚀 Initialising parallel scan...")
    results, done, total = [], 0, len(universe)
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(_scan_one, (sym, capital, max_price, min_conf, penny_only)): sym
                   for sym in universe}
        for fut in as_completed(futures):
            done += 1
            prog.progress(done / total, text=f"📡 {done}/{total} — {futures[fut].replace('.NS','')}")
            res = fut.result()
            if res:
                results.append(res)
    prog.empty()
    results.sort(key=lambda x: (x["confidence"], x["risk_reward"]), reverse=True)
    return results

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

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
is_open = market_open()
mkt = get_market_status()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:12px 0 18px;'>
        <div style='font-size:2.2rem;'>⚡</div>
        <div style='font-size:1.25rem;font-weight:900;color:#f1f5f9;letter-spacing:-0.02em;'>QuantSignal</div>
        <div style='font-size:0.65rem;color:#334155;letter-spacing:0.12em;text-transform:uppercase;'>INDIA TRADING ENGINE v5.0</div>
    </div>""", unsafe_allow_html=True)

    dot = '<span class="live-dot"></span>' if is_open else "🔴 "
    mkt_color = "#10b981" if is_open else "#ef4444"
    st.markdown(f'<div style="color:{mkt_color};font-weight:700;font-size:0.85rem;">{dot}{mkt}</div>', unsafe_allow_html=True)
    if is_open:
        st.markdown(f'<div style="color:#475569;font-size:0.72rem;">⏱ {time_to_close()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#1e3a5f;font-size:0.68rem;margin-top:2px;">{datetime.now().strftime("%d %b %Y  %I:%M %p")}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">💰 Capital & Risk</div>', unsafe_allow_html=True)
    capital = st.number_input("Capital (₹)", value=5000, min_value=500, max_value=10_000_000, step=500)
    min_conf = st.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)
    max_price = st.number_input("Max Stock Price (₹)", value=int(capital * 0.95), min_value=5, max_value=5_000_000)

    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">🔍 Filters</div>', unsafe_allow_html=True)

    sector_filter = st.multiselect("Sectors (empty = all)", list(SECTOR_GROUPS.keys()), default=[], key="sf")
    stock_filter = st.multiselect("Specific Stocks", ALL_SYMBOLS_CLEAN, default=[], key="stf")
    penny_only = st.checkbox("🪙 Penny Stocks Only (≤₹50)", value=False)

    st.markdown("---")
    st.markdown('<div style="color:#475569;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">📅 Date Range</div>', unsafe_allow_html=True)
    date_from = st.date_input("From", value=date.today() - timedelta(days=30))
    date_to = st.date_input("To", value=date.today())

    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
    if auto_refresh:
        refresh_sec = st.select_slider("Interval", [60, 120, 300, 600], value=300,
                                        format_func=lambda x: f"{x//60}m")
    else:
        refresh_sec = 300

    st.markdown("---")
    st.markdown(f'<div style="color:#1e3a5f;font-size:0.68rem;">Universe: <b style="color:#3b82f6;">{len(INTRADAY_STOCKS)}</b> stocks | 14 sectors</div>', unsafe_allow_html=True)
    st.caption("Not financial advice. Paper trade first.")

# ── TOP HEADER ────────────────────────────────────────────────────────────────
universe = get_universe(sector_filter, stock_filter, penny_only)

hcol1, hcol2, hcol3, hcol4 = st.columns([3.5, 1, 1, 1])
with hcol1:
    live_badge = ('<span style="background:#065f46;color:#6ee7b7;padding:2px 9px;border-radius:5px;font-size:0.68rem;font-weight:800;letter-spacing:0.05em;">● LIVE</span>'
                  if is_open else
                  '<span style="background:#7f1d1d;color:#fca5a5;padding:2px 9px;border-radius:5px;font-size:0.68rem;font-weight:800;">● CLOSED</span>')
    penny_badge = '<span style="background:#4c1d95;color:#c4b5fd;padding:2px 9px;border-radius:5px;font-size:0.68rem;font-weight:800;margin-left:6px;">🪙 PENNY MODE</span>' if penny_only else ""
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:2px;flex-wrap:wrap;'>
        <span style='font-size:1.7rem;font-weight:900;color:#f8fafc;letter-spacing:-0.03em;'>⚡ QuantSignal India</span>
        {live_badge}{penny_badge}
    </div>
    <div style='color:#334155;font-size:0.8rem;'>
        <b style='color:#2563eb;'>{len(universe)}</b> stocks scanning •
        <b style='color:#2563eb;'>14 sectors</b> •
        {datetime.now().strftime("%d %b %Y %I:%M %p")}
    </div>""", unsafe_allow_html=True)
with hcol2:
    scan_btn = st.button("🔍 SCAN NOW", type="primary", use_container_width=True)
with hcol3:
    best_btn = st.button("⚡ BEST TRADE", use_container_width=True)
with hcol4:
    predict_btn = st.button("🔮 PREDICT ALL", use_container_width=True)

st.markdown("---")

# ── TRIGGER SCAN ──────────────────────────────────────────────────────────────
if scan_btn or best_btn:
    trades = run_scan(universe, capital, max_price, min_conf, penny_only)
    st.session_state["trades"] = trades
    st.session_state["buys"] = [t for t in trades if t["signal"] == "BUY"]
    st.session_state["pennies"] = [t for t in trades if t["is_penny"]]
    st.session_state["scan_time"] = datetime.now().strftime("%I:%M:%S %p")
    st.session_state["scan_date"] = datetime.now().strftime("%d %b %Y")

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📡 Live Signals",
    "🪙 Penny Stocks",
    "🔮 Next-Day Picks",
    "📈 Forecast",
    "🔎 Stock Explorer",
    "📋 Reports",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE INTRADAY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if "buys" not in st.session_state:
        # Welcome screen
        st.markdown(f"""
        <div style='text-align:center;padding:40px 20px;'>
            <div style='font-size:3.5rem;margin-bottom:12px;'>⚡</div>
            <div style='font-size:1.8rem;font-weight:900;color:#f1f5f9;margin-bottom:8px;'>Live Intraday Signal Engine</div>
            <div style='color:#334155;font-size:0.95rem;margin-bottom:32px;'>
                Scanning <b style='color:#2563eb;'>{len(universe)}</b> NSE stocks in parallel using VWAP, EMA, RSI, Supertrend & MACD
            </div>
        </div>""", unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        for col, icon, num, lbl in [(c1,"📡",len(universe),"Stocks"),(c2,"🏭","14","Sectors"),(c3,"📊","7","Indicators"),(c4,"🤖","ML+TA","Engine")]:
            col.markdown(f'<div class="stat-card"><div style="font-size:1.6rem;">{icon}</div><div class="stat-num">{num}</div><div class="stat-label">{lbl}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🌍 Active Geopolitical Themes")
        themes = [
            ("🛡️","India-US Defence Deal","HAL, BEL, BDL, GRSE","35%+ sector rally"),
            ("🏦","PSU Bank Outperformance","Canara, Indian Bank, BOI","+31-68% YTD"),
            ("🏗️","Infra Supercycle","RVNL, IRFC, KEC","₹11L cr budget"),
            ("⚡","Energy Security","NTPC, Suzlon, SJVN","Renewables push"),
            ("🧪","China+1 Chemicals","SRF, Deepak Nitrite","Supply chain shift"),
            ("🚗","EV PLI Scheme","Tata Motors, M&M, Olectra","Govt incentives"),
        ]
        tc = st.columns(3)
        for i, (icon, title, stocks, note) in enumerate(themes):
            tc[i%3].markdown(f'<div class="stat-card" style="text-align:left;margin-bottom:10px;"><div style="font-size:1.1rem;font-weight:800;color:#e2e8f0;">{icon} {title}</div><div style="color:#475569;font-size:0.78rem;margin-top:4px;">{stocks}</div><div style="color:#2563eb;font-size:0.78rem;font-weight:700;margin-top:2px;">{note}</div></div>', unsafe_allow_html=True)
    else:
        buys = st.session_state["buys"]
        all_t = st.session_state["trades"]
        scan_time = st.session_state.get("scan_time","")

        # Stats row
        s1,s2,s3,s4,s5,s6 = st.columns(6)
        s1.metric("Scanned", len(all_t))
        s2.metric("BUY Signals", len(buys))
        s3.metric("WATCH", len(all_t)-len(buys))
        penny_ct = len([t for t in all_t if t["is_penny"]])
        s4.metric("Penny Stocks", penny_ct)
        if buys:
            s5.metric("Best Confidence", f"{buys[0]['confidence']:.0%}", buys[0]["symbol"])
            s6.metric("Best R:R", f"{buys[0]['risk_reward']}x", buys[0]["sector"])
        st.markdown("---")

        if not buys:
            st.warning("No BUY signals found. Try lowering Min Confidence or removing filters.")
        else:
            best = buys[0]
            st.markdown(f"### 🏆 #1 Best Trade — {scan_time}")
            cl, cm, cr = st.columns([2.5, 2, 1.2])
            with cl:
                penny_tag = '<span class="badge-penny">🪙 PENNY</span> ' if best["is_penny"] else ""
                st.markdown(f"""
                <div class="trade-card">
                    <div style='margin-bottom:6px;'>
                        <span class="badge-sector">{best["sector"]}</span>
                        {penny_tag}
                        <span style='color:{best["risk_level"].split()[0]};font-size:0.72rem;font-weight:700;'>{best["risk_level"]}</span>
                    </div>
                    <div class="stock-title">{best["symbol"]}</div>
                    <div class="price-big">₹{best["price"]:,.2f}</div>
                    <div style='margin-top:12px;'>
                        <span class="badge-buy">BUY &nbsp;{best["confidence"]:.0%} Confidence</span>
                    </div>
                    <div style='margin-top:10px;'>
                        {''.join(f'<span class="reason-chip">{r}</span>' for r in best["reasons"])}
                    </div>
                    <div style='margin-top:10px;color:#334155;font-size:0.75rem;'>⏱ {best["holding"]}</div>
                </div>""", unsafe_allow_html=True)
            with cm:
                st.metric("🎯 Target 1", f"₹{best['target_1']:,.2f}", f"+₹{best['profit']:,.2f}")
                st.metric("🎯 Target 2", f"₹{best['target_2']:,.2f}")
                st.metric("🛑 Stop Loss", f"₹{best['stop_loss']:,.2f}", f"-₹{best['loss']:,.2f}", delta_color="inverse")
            with cr:
                st.metric("Qty", best["qty"])
                st.metric("Invest", f"₹{best['invested']:,.0f}")
                st.metric("R:R", f"{best['risk_reward']}x")
                st.metric("RSI", best["rsi"])

            # Chart
            if "df" in best and not best["df"].empty:
                dc = best["df"].tail(100)
                fig = make_subplots(rows=3, cols=1, row_heights=[0.60,0.20,0.20],
                                    shared_xaxes=True, vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(
                    x=dc.index, open=dc["Open"], high=dc["High"], low=dc["Low"], close=dc["Close"],
                    name="Price", increasing=dict(line=dict(color="#10b981"),fillcolor="#10b981"),
                    decreasing=dict(line=dict(color="#ef4444"),fillcolor="#ef4444"),
                ), row=1, col=1)
                for cn, col, dash, lbl in [("VWAP","#f59e0b","dot","VWAP"),("EMA9","#6366f1","solid","EMA9"),("EMA21","#ec4899","solid","EMA21")]:
                    if cn in dc.columns:
                        fig.add_trace(go.Scatter(x=dc.index, y=dc[cn], mode="lines",
                            line=dict(color=col,width=1.2,dash=dash), name=lbl), row=1, col=1)
                fig.add_hline(y=best["price"], line_color="#6366f1", line_width=1.5, annotation_text="ENTRY", row=1, col=1)
                fig.add_hline(y=best["target_1"], line_color="#10b981", line_dash="dash", annotation_text="T1", row=1, col=1)
                fig.add_hline(y=best["stop_loss"], line_color="#ef4444", line_dash="dash", annotation_text="SL", row=1, col=1)
                if "RSI" in dc.columns:
                    fig.add_trace(go.Scatter(x=dc.index, y=dc["RSI"], mode="lines",
                        line=dict(color="#a78bfa",width=1.5), name="RSI",
                        fill="tozeroy", fillcolor="rgba(167,139,250,0.05)"), row=2, col=1)
                    fig.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=0.8, row=2, col=1)
                    fig.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=0.8, row=2, col=1)
                vcols = ["#10b981" if c >= o else "#ef4444" for c,o in zip(dc["Close"],dc["Open"])]
                fig.add_trace(go.Bar(x=dc.index, y=dc["Volume"], marker_color=vcols, name="Vol", opacity=0.8), row=3, col=1)
                layout = _chart_layout(height=500)
                layout["xaxis_rangeslider_visible"] = False
                fig.update_layout(**layout)
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            # Sector + confidence charts
            col_sec, col_conf = st.columns(2)
            with col_sec:
                st.markdown("### 📊 BUY Signals by Sector")
                sc_cnt = {}
                for t in buys:
                    sc_cnt[t["sector"]] = sc_cnt.get(t["sector"], 0) + 1
                sc_df = pd.DataFrame(sorted(sc_cnt.items(), key=lambda x: x[1], reverse=True), columns=["Sector","Count"])
                fig_s = go.Figure(go.Bar(x=sc_df["Count"], y=sc_df["Sector"], orientation="h",
                    marker=dict(color=sc_df["Count"], colorscale=[[0,"#0f2a4a"],[1,"#06b6d4"]]),
                    text=sc_df["Count"], textposition="outside"))
                fig_s.update_layout(**_chart_layout(height=280))
                fig_s.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_s, use_container_width=True)
            with col_conf:
                st.markdown("### 🎯 Top 15 Confidence")
                top15 = buys[:15]
                fig_c = go.Figure(go.Bar(
                    x=[t["symbol"] for t in top15], y=[t["confidence"] for t in top15],
                    marker=dict(color=[t["confidence"] for t in top15],
                                colorscale=[[0,"#dc2626"],[0.5,"#f59e0b"],[1,"#10b981"]], cmin=0, cmax=1),
                    text=[f"{t['confidence']:.0%}" for t in top15], textposition="outside"))
                fig_c.update_layout(**_chart_layout(height=280))
                fig_c.update_layout(yaxis=dict(range=[0,1.15], tickformat=".0%"))
                st.plotly_chart(fig_c, use_container_width=True)

            st.markdown("---")
            st.markdown(f"### 📋 All BUY Signals ({len(buys)} stocks)")
            tbl = pd.DataFrame([{
                "#": i+1, "Stock": t["symbol"], "Sector": t["sector"],
                "Price": f"₹{t['price']:,.2f}", "Target": f"₹{t['target_1']:,.2f}",
                "SL": f"₹{t['stop_loss']:,.2f}", "Conf": f"{t['confidence']:.0%}",
                "R:R": f"{t['risk_reward']}x", "Qty": t["qty"],
                "Invest": f"₹{t['invested']:,.0f}", "Profit": f"+₹{t['profit']:,.2f}",
                "Loss": f"-₹{t['loss']:,.2f}", "RSI": t["rsi"],
                "Vol": f"{t['vol_ratio']:.1f}x", "ST": t["supertrend"],
                "Risk": t["risk_level"], "Penny": "🪙" if t["is_penny"] else "",
            } for i, t in enumerate(buys)])
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=min(600, 55+38*len(tbl)))

            st.markdown("---")
            st.markdown("### 🔎 Stock Details (Top 15)")
            for t in buys[:15]:
                icon = "🟢" if t["confidence"] >= 0.65 else "🟡"
                penny_tag = " 🪙" if t["is_penny"] else ""
                with st.expander(f"{icon} {t['symbol']}{penny_tag} [{t['sector']}] ₹{t['price']:,.2f} • {t['confidence']:.0%} • R:R {t['risk_reward']}x"):
                    d1,d2,d3,d4,d5 = st.columns(5)
                    d1.metric("Entry", f"₹{t['price']:,.2f}")
                    d2.metric("Target 1", f"₹{t['target_1']:,.2f}", f"+₹{t['profit']:,.2f}")
                    d3.metric("Stop Loss", f"₹{t['stop_loss']:,.2f}", f"-₹{t['loss']:,.2f}", delta_color="inverse")
                    d4.metric("R:R", f"{t['risk_reward']}x")
                    d5.metric("Risk", t["risk_level"])
                    st.markdown("".join(f'<span class="reason-chip">{r}</span>' for r in t["reasons"]), unsafe_allow_html=True)
                    # Global factors
                    factors = StockMetadata.get_global_factors(t["sector"])
                    st.markdown(f'<div style="margin-top:8px;font-size:0.72rem;color:#475569;font-weight:700;">🌍 SECTOR THEME: <span style="color:#3b82f6;">{factors["theme"]}</span></div>', unsafe_allow_html=True)
                    pos_html = "".join(f'<span class="factor-pos">✅ {f}</span>' for f in factors["positive"][:3])
                    neg_html = "".join(f'<span class="factor-neg">⚠️ {f}</span>' for f in factors["negative"][:2])
                    st.markdown(pos_html + neg_html, unsafe_allow_html=True)

    if auto_refresh and "buys" in st.session_state:
        _time.sleep(refresh_sec)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — PENNY STOCKS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="penny-header">
        <div style='font-size:1.3rem;font-weight:900;color:#c4b5fd;'>🪙 Penny Stock Scanner</div>
        <div style='color:#7c3aed;font-size:0.82rem;margin-top:4px;'>
            Stocks priced ≤ ₹50 • High risk, high reward • Always use strict stop-loss
        </div>
    </div>""", unsafe_allow_html=True)

    col_p1, col_p2, col_p3 = st.columns([2, 1, 1])
    with col_p1:
        penny_max_price_filter = st.slider("Max Price (₹)", 5, 50, 50, 5)
    with col_p2:
        penny_min_conf = st.slider("Min Confidence", 0.30, 0.80, 0.40, 0.05, key="penny_conf")
    with col_p3:
        penny_scan_btn = st.button("🪙 Scan Penny Stocks", type="primary", use_container_width=True)

    if penny_scan_btn:
        penny_universe = get_universe([], [], True)
        penny_trades = run_scan(penny_universe, capital, penny_max_price_filter, penny_min_conf, penny_only=True)
        st.session_state["penny_trades"] = penny_trades

    if "penny_trades" in st.session_state:
        pt = st.session_state["penny_trades"]
        if not pt:
            st.info("No penny stock signals found. Try lowering confidence or increasing max price.")
        else:
            # Summary
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Penny Stocks Found", len(pt))
            pm2.metric("BUY Signals", len([t for t in pt if t["signal"]=="BUY"]))
            if pt:
                pm3.metric("Cheapest", f"₹{min(t['price'] for t in pt):,.2f}")
                pm4.metric("Best Confidence", f"{max(t['confidence'] for t in pt):.0%}")

            st.markdown("---")
            st.markdown("### 🪙 Penny Stock Detailed Report")

            for t in pt[:20]:
                signal_color = "#10b981" if t["signal"]=="BUY" else "#f59e0b"
                factors = StockMetadata.get_global_factors(t["sector"])
                with st.expander(
                    f"🪙 {t['symbol']} — ₹{t['price']:,.2f} — {t['signal']} — Conf {t['confidence']:.0%} — {t['risk_level']}"
                ):
                    rc1, rc2, rc3, rc4, rc5 = st.columns(5)
                    rc1.metric("Price", f"₹{t['price']:,.2f}")
                    rc2.metric("Target", f"₹{t['target_1']:,.2f}", f"+{((t['target_1']-t['price'])/t['price']*100):.1f}%")
                    rc3.metric("Stop Loss", f"₹{t['stop_loss']:,.2f}", f"-{((t['price']-t['stop_loss'])/t['price']*100):.1f}%", delta_color="inverse")
                    rc4.metric("R:R Ratio", f"{t['risk_reward']}x")
                    rc5.metric("Qty (₹{:,.0f})".format(capital), t["qty"])

                    st.markdown(f"""
                    <div style='display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;'>
                        <div class="report-card">
                            <div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;margin-bottom:8px;'>📊 Technical Signals</div>
                            {''.join(f'<span class="reason-chip">{r}</span>' for r in t["reasons"])}
                            <div style='margin-top:8px;color:#64748b;font-size:0.75rem;'>
                                RSI: <b style='color:#a78bfa;'>{t["rsi"]}</b> &nbsp;|&nbsp;
                                VWAP: <b style='color:#f59e0b;'>₹{t["vwap"]}</b> &nbsp;|&nbsp;
                                Volume: <b style='color:#06b6d4;'>{t["vol_ratio"]:.1f}x</b> &nbsp;|&nbsp;
                                Supertrend: <b style='color:{"#10b981" if t["supertrend"]=="BUY" else "#ef4444"};'>{t["supertrend"]}</b>
                            </div>
                        </div>
                        <div class="report-card">
                            <div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;margin-bottom:8px;'>⚠️ Risk Assessment</div>
                            <div style='font-size:1.1rem;font-weight:800;'>{t["risk_level"]}</div>
                            <div style='color:#64748b;font-size:0.75rem;margin-top:6px;'>
                                Sector: <b style='color:#a5b4fc;'>{t["sector"]}</b><br>
                                Holding: <b style='color:#f59e0b;'>{t["holding"]}</b><br>
                                Invest: <b style='color:#10b981;'>₹{t["invested"]:,.0f}</b>
                            </div>
                        </div>
                    </div>
                    <div class="report-card" style='margin-top:12px;'>
                        <div style='font-size:0.72rem;color:#475569;font-weight:700;text-transform:uppercase;margin-bottom:8px;'>🌍 Sector Theme: <span style='color:#3b82f6;'>{factors["theme"]}</span></div>
                        <div>{''.join(f'<span class="factor-pos">✅ {f}</span>' for f in factors["positive"][:3])}</div>
                        <div style='margin-top:6px;'>{''.join(f'<span class="factor-neg">⚠️ {f}</span>' for f in factors["negative"][:2])}</div>
                    </div>
                    <div style='margin-top:10px;background:rgba(220,38,38,0.08);border:1px solid rgba(220,38,38,0.2);border-radius:8px;padding:10px;font-size:0.75rem;color:#fca5a5;'>
                        ⚠️ <b>Penny Stock Warning:</b> High volatility, low liquidity. Use strict stop-loss. Never invest more than 2-5% of capital in a single penny stock.
                    </div>""", unsafe_allow_html=True)

                    if "df" in t and not t["df"].empty:
                        dc = t["df"].tail(60)
                        fm = go.Figure()
                        fm.add_trace(go.Candlestick(
                            x=dc.index, open=dc["Open"], high=dc["High"], low=dc["Low"], close=dc["Close"],
                            increasing=dict(line=dict(color="#10b981"),fillcolor="#10b981"),
                            decreasing=dict(line=dict(color="#ef4444"),fillcolor="#ef4444"),
                        ))
                        if "VWAP" in dc.columns:
                            fm.add_trace(go.Scatter(x=dc.index, y=dc["VWAP"], mode="lines",
                                line=dict(color="#f59e0b",width=1.2,dash="dot"), name="VWAP"))
                        fm.add_hline(y=t["price"], line_color="#6366f1", annotation_text="Entry")
                        fm.add_hline(y=t["target_1"], line_color="#10b981", line_dash="dash", annotation_text="T1")
                        fm.add_hline(y=t["stop_loss"], line_color="#ef4444", line_dash="dash", annotation_text="SL")
                        fm.update_layout(**_chart_layout(height=280))
                        fm.update_layout(xaxis_rangeslider_visible=False, showlegend=True)
                        st.plotly_chart(fm, use_container_width=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:50px;background:#0a1628;border:1px solid #0f2040;border-radius:16px;'>
            <div style='font-size:2.5rem;'>🪙</div>
            <div style='font-size:1.1rem;color:#94a3b8;margin-top:10px;'>Click "Scan Penny Stocks" to find high-potential low-price stocks</div>
            <div style='color:#334155;margin-top:6px;font-size:0.82rem;'>Stocks priced ≤ ₹50 with strong technical signals</div>
        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — NEXT-DAY PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🔮 Next-Day ML Predictions")
    st.markdown('<div style="color:#334155;font-size:0.82rem;margin-bottom:16px;">Ensemble ML (GBM + RF + Ridge) predicts tomorrow\'s direction. Best for swing entry planning.</div>', unsafe_allow_html=True)

    nd_c1, nd_c2, nd_c3, nd_c4 = st.columns([2, 1, 1, 1])
    with nd_c1:
        nd_sectors = st.multiselect("Filter Sectors", list(SECTOR_GROUPS.keys()), default=[], key="nd_sec")
    with nd_c2:
        nd_stocks = st.multiselect("Specific Stocks", ALL_SYMBOLS_CLEAN, default=[], key="nd_stk")
    with nd_c3:
        nd_min_conf = st.slider("Min Confidence", 0.50, 0.90, 0.60, 0.05, key="nd_conf")
    with nd_c4:
        nd_btn = st.button("🔮 Run Predictions", type="primary", use_container_width=True)

    if nd_btn or predict_btn:
        if nd_stocks:
            nd_universe = [f"{s}.NS" for s in nd_stocks]
        elif nd_sectors:
            nd_universe = []
            for s in nd_sectors:
                nd_universe.extend(SECTOR_GROUPS.get(s, []))
            nd_universe = list(dict.fromkeys(nd_universe))
        else:
            nd_universe = INTRADAY_STOCKS[:120]

        prog2 = st.progress(0, text="🔮 Running ML predictions...")
        nd_results = []
        total2 = len(nd_universe)

        def _predict_one(sym):
            try:
                df = DataService.fetch_ohlcv(sym, period="1y")
                if df.empty or len(df) < 100:
                    return None
                pred = PredictionEngine.predict_next_day(df)
                if "error" in pred:
                    return None
                clean = sym.replace(".NS", "")
                sector = SYMBOL_TO_SECTOR.get(clean, "Other")
                report = StockMetadata.generate_report(
                    symbol=clean, price=pred["current_price"], sector=sector,
                    confidence=pred["confidence"], direction=pred["direction"],
                    predicted_return=pred["predicted_return"],
                    volatility=pred.get("volatility", 20) / 100,
                    rsi=pred.get("rsi", 50),
                    entry=pred["current_price"],
                    target=pred["predicted_price"],
                    stop_loss=pred["current_price"] * 0.97,
                    mode="swing",
                )
                return {
                    "symbol": clean, "sector": sector,
                    "price": pred["current_price"],
                    "predicted": pred["predicted_price"],
                    "return_pct": pred["predicted_return"],
                    "direction": pred["direction"],
                    "confidence": pred["confidence"],
                    "rsi": pred.get("rsi", 50),
                    "volatility": pred.get("volatility", 20),
                    "conditions": pred.get("market_conditions", []),
                    "risk_level": report["risk_level"],
                    "signal": report["signal"],
                    "is_penny": pred["current_price"] <= PENNY_MAX_PRICE,
                    "factors": StockMetadata.get_global_factors(sector),
                }
            except Exception:
                return None

        done2 = 0
        with ThreadPoolExecutor(max_workers=10) as ex:
            futs = {ex.submit(_predict_one, sym): sym for sym in nd_universe}
            for fut in as_completed(futs):
                done2 += 1
                prog2.progress(done2 / total2, text=f"🔮 {done2}/{total2}")
                res = fut.result()
                if res and res["direction"] == "UP" and res["confidence"] >= nd_min_conf:
                    nd_results.append(res)

        prog2.empty()
        nd_results.sort(key=lambda x: (x["confidence"], x["return_pct"]), reverse=True)
        st.session_state["nd_results"] = nd_results

    if "nd_results" in st.session_state:
        nd_results = st.session_state["nd_results"]
        if not nd_results:
            st.info("No strong UP predictions found. Try lowering confidence or changing filters.")
        else:
            nm1, nm2, nm3, nm4 = st.columns(4)
            nm1.metric("UP Predictions", len(nd_results))
            nm2.metric("Best Return", f"{nd_results[0]['return_pct']:+.2f}%", nd_results[0]["symbol"])
            nm3.metric("Avg Confidence", f"{np.mean([r['confidence'] for r in nd_results]):.0%}")
            nm4.metric("Avg Expected Return", f"{np.mean([r['return_pct'] for r in nd_results]):+.2f}%")

            st.markdown("---")
            st.markdown("### 🏆 Top 5 Next-Day Picks")
            cols_nd = st.columns(min(5, len(nd_results)))
            for col, r in zip(cols_nd, nd_results[:5]):
                ret_color = "#10b981" if r["return_pct"] > 0 else "#ef4444"
                penny_tag = "🪙 " if r["is_penny"] else ""
                with col:
                    st.markdown(f"""
                    <div class="stat-card" style="text-align:center;">
                        <div style="font-size:0.65rem;color:#334155;">{r["sector"]}</div>
                        <div style="font-size:1rem;font-weight:900;color:#f1f5f9;margin:4px 0;">{penny_tag}{r["symbol"]}</div>
                        <div style="font-size:1.2rem;color:#94a3b8;">₹{r["price"]:,.2f}</div>
                        <div style="font-size:1.1rem;font-weight:800;color:{ret_color};">{r["return_pct"]:+.2f}%</div>
                        <div style="font-size:0.72rem;color:#475569;">→ ₹{r["predicted"]:,.2f}</div>
                        <div style="margin-top:8px;"><span class="badge-buy" style="font-size:0.75rem;padding:3px 10px;">UP {r["confidence"]:.0%}</span></div>
                        <div style="font-size:0.68rem;color:#334155;margin-top:4px;">{r["risk_level"]}</div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            nd_tbl = pd.DataFrame([{
                "#": i+1, "Stock": r["symbol"], "Sector": r["sector"],
                "Current": f"₹{r['price']:,.2f}", "Predicted": f"₹{r['predicted']:,.2f}",
                "Return": f"{r['return_pct']:+.2f}%", "Direction": r["direction"],
                "Confidence": f"{r['confidence']:.0%}", "Signal": r["signal"],
                "RSI": r["rsi"], "Volatility": f"{r['volatility']:.1f}%",
                "Risk": r["risk_level"], "Penny": "🪙" if r["is_penny"] else "",
            } for i, r in enumerate(nd_results)])
            st.dataframe(nd_tbl, use_container_width=True, hide_index=True, height=min(500, 55+38*len(nd_tbl)))

            # Scatter
            fig_nd = go.Figure(go.Scatter(
                x=[r["return_pct"] for r in nd_results],
                y=[r["confidence"] for r in nd_results],
                mode="markers+text",
                text=[r["symbol"] for r in nd_results],
                textposition="top center",
                marker=dict(
                    size=[max(8, r["confidence"]*22) for r in nd_results],
                    color=[r["return_pct"] for r in nd_results],
                    colorscale=[[0,"#dc2626"],[0.5,"#f59e0b"],[1,"#10b981"]],
                    showscale=True, colorbar=dict(title="Return %", tickfont=dict(color="#94a3b8")),
                ),
            ))
            fig_nd.add_vline(x=0, line_color="#334155", line_dash="dash")
            fig_nd.update_layout(**_chart_layout("Next-Day: Expected Return vs Confidence", height=400))
            fig_nd.update_layout(xaxis_title="Expected Return (%)", yaxis_title="Confidence")
            st.plotly_chart(fig_nd, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MULTI-PERIOD FORECAST
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📈 Multi-Period Price Forecast")
    st.markdown('<div style="color:#334155;font-size:0.82rem;margin-bottom:16px;">ML forecasts for 10, 20, 30 days and 3, 6 months with confidence bands and global factor analysis.</div>', unsafe_allow_html=True)

    mp_c1, mp_c2, mp_c3 = st.columns([2, 1, 1])
    with mp_c1:
        mp_symbol = st.selectbox("Select Stock", ALL_SYMBOLS_CLEAN,
                                  index=ALL_SYMBOLS_CLEAN.index("RELIANCE") if "RELIANCE" in ALL_SYMBOLS_CLEAN else 0,
                                  key="mp_sym")
    with mp_c2:
        mp_mode = st.selectbox("Trading Mode", ["📡 Intraday", "📈 Swing (1-4 weeks)", "💼 Long-Term (3-12 months)"], key="mp_mode")
    with mp_c3:
        mp_btn = st.button("📈 Generate Forecast", type="primary", use_container_width=True)

    if mp_btn:
        with st.spinner(f"Running ML forecast for {mp_symbol}..."):
            df_mp = DataService.fetch_ohlcv(f"{mp_symbol}.NS", period="2y")
            if df_mp.empty:
                st.error(f"No data for {mp_symbol}")
            else:
                nd_pred = PredictionEngine.predict_next_day(df_mp)
                multi_pred = PredictionEngine.predict_multi_horizon(df_mp)
                sector = SYMBOL_TO_SECTOR.get(mp_symbol, "Other")
                factors = StockMetadata.get_global_factors(sector)
                st.session_state["mp_result"] = {
                    "symbol": mp_symbol, "next_day": nd_pred,
                    "multi": multi_pred, "df": df_mp,
                    "sector": sector, "factors": factors,
                }

    if "mp_result" in st.session_state:
        r = st.session_state["mp_result"]
        sym = r["symbol"]
        nd = r["next_day"]
        multi = r["multi"]
        df_hist = r["df"]
        factors = r["factors"]
        sector = r["sector"]

        if "error" not in nd:
            dir_color = "#10b981" if nd["direction"]=="UP" else ("#ef4444" if nd["direction"]=="DOWN" else "#f59e0b")
            dir_badge = f'<span style="background:{dir_color}22;color:{dir_color};padding:3px 10px;border-radius:6px;font-weight:800;font-size:0.82rem;">{nd["direction"]}</span>'
            st.markdown(f"""
            <div class="trade-card" style="margin-bottom:16px;">
                <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>
                    <span style='font-size:1.3rem;font-weight:900;color:#f1f5f9;'>{sym}</span>
                    {dir_badge}
                    <span class="badge-sector">{sector}</span>
                    <span style='color:#94a3b8;'>Current: <b style='color:#f1f5f9;'>₹{nd["current_price"]:,.2f}</b></span>
                    <span style='color:#94a3b8;'>Tomorrow: <b style='color:{dir_color};'>₹{nd["predicted_price"]:,.2f} ({nd["predicted_return"]:+.2f}%)</b></span>
                    <span style='color:#94a3b8;'>Confidence: <b style='color:#3b82f6;'>{nd["confidence"]:.0%}</b></span>
                </div>
                <div style='margin-top:10px;color:#475569;font-size:0.78rem;'>
                    {'&nbsp;•&nbsp;'.join(nd.get("market_conditions", []))}
                </div>
            </div>""", unsafe_allow_html=True)

        # Horizon cards
        horizon_labels = {
            "10_days": ("10 Days","📅"), "20_days": ("20 Days","📅"),
            "30_days": ("30 Days","📅"), "3_months": ("3 Months","📆"), "6_months": ("6 Months","📆"),
        }
        cols_h = st.columns(5)
        for col, (key, (label, icon)) in zip(cols_h, horizon_labels.items()):
            pred = multi.get(key, {})
            if pred and "error" not in pred:
                ret = pred["predicted_return"]
                ret_color = "#10b981" if ret > 0 else "#ef4444"
                dir_icon = "↑" if pred["direction"]=="UP" else ("↓" if pred["direction"]=="DOWN" else "→")
                with col:
                    st.markdown(f"""
                    <div class="stat-card" style="text-align:center;">
                        <div style="font-size:0.68rem;color:#334155;">{icon} {label}</div>
                        <div style="font-size:1.3rem;font-weight:900;color:{ret_color};margin:6px 0;">{dir_icon} {ret:+.1f}%</div>
                        <div style="font-size:0.85rem;color:#94a3b8;">₹{pred["predicted_price"]:,.0f}</div>
                        <div style="font-size:0.68rem;color:#334155;">₹{pred["price_low"]:,.0f} – ₹{pred["price_high"]:,.0f}</div>
                        <div style="font-size:0.68rem;color:#3b82f6;margin-top:4px;">Conf: {pred["confidence"]:.0%}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Forecast chart
        current_price = float(df_hist["Close"].iloc[-1])
        horizons_plot = []
        for key, (label, _) in horizon_labels.items():
            pred = multi.get(key, {})
            if pred and "error" not in pred:
                horizons_plot.append({
                    "label": label, "days": pred["horizon_days"],
                    "price": pred["predicted_price"], "low": pred["price_low"],
                    "high": pred["price_high"], "direction": pred["direction"],
                })

        if horizons_plot:
            fig_mp = go.Figure()
            hist_60 = df_hist["Close"].tail(60)
            x_hist = list(range(-len(hist_60), 0))
            fig_mp.add_trace(go.Scatter(x=x_hist, y=hist_60.values, mode="lines",
                line=dict(color="#6366f1", width=2), name="Historical"))
            x_fore = [h["days"] for h in horizons_plot]
            y_fore = [h["price"] for h in horizons_plot]
            y_low = [h["low"] for h in horizons_plot]
            y_high = [h["high"] for h in horizons_plot]
            colors = ["#10b981" if h["direction"]=="UP" else "#ef4444" for h in horizons_plot]
            fig_mp.add_trace(go.Scatter(x=x_fore+x_fore[::-1], y=y_high+y_low[::-1],
                fill="toself", fillcolor="rgba(99,102,241,0.08)", line=dict(color="rgba(0,0,0,0)"),
                name="Price Range"))
            fig_mp.add_trace(go.Scatter(x=x_fore, y=y_fore, mode="markers+lines+text",
                text=[h["label"] for h in horizons_plot], textposition="top center",
                marker=dict(size=12, color=colors, line=dict(color="white",width=2)),
                line=dict(color="#94a3b8", width=1.5, dash="dash"), name="Forecast"))
            fig_mp.add_hline(y=current_price, line_color="#f59e0b", line_dash="dot",
                              annotation_text=f"Current ₹{current_price:,.2f}")
            fig_mp.update_layout(**_chart_layout(f"{sym} — Price Forecast", height=420))
            fig_mp.update_layout(xaxis_title="Days from Today", yaxis_title="Price (₹)")
            st.plotly_chart(fig_mp, use_container_width=True)

        # Global factors
        st.markdown("---")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f"### 🌍 Sector Theme: {factors['theme']}")
            st.markdown("**Positive Factors:**")
            for f in factors["positive"]:
                st.markdown(f'<span class="factor-pos">✅ {f}</span>', unsafe_allow_html=True)
        with col_f2:
            st.markdown("### ⚠️ Risk Factors")
            st.markdown("**Negative Factors:**")
            for f in factors["negative"]:
                st.markdown(f'<span class="factor-neg">⚠️ {f}</span>', unsafe_allow_html=True)

        # Historical chart
        st.markdown("---")
        st.markdown(f"### 📊 {sym} — Historical Chart")
        fig_hist = make_subplots(rows=4, cols=1, row_heights=[0.50,0.17,0.17,0.16],
                                  shared_xaxes=True, vertical_spacing=0.02)
        fig_hist.add_trace(go.Candlestick(
            x=df_hist.index, open=df_hist["Open"], high=df_hist["High"],
            low=df_hist["Low"], close=df_hist["Close"], name="Price",
            increasing=dict(line=dict(color="#10b981"),fillcolor="#10b981"),
            decreasing=dict(line=dict(color="#ef4444"),fillcolor="#ef4444"),
        ), row=1, col=1)
        for w, color in [(20,"#f59e0b"),(50,"#6366f1"),(200,"#ec4899")]:
            if len(df_hist) >= w:
                fig_hist.add_trace(go.Scatter(x=df_hist.index, y=df_hist["Close"].rolling(w).mean(),
                    mode="lines", line=dict(color=color,width=1), name=f"MA{w}"), row=1, col=1)
        delta = df_hist["Close"].diff()
        gain = delta.where(delta>0,0).rolling(14).mean()
        loss = (-delta.where(delta<0,0)).rolling(14).mean()
        rsi_vals = 100 - (100/(1+gain/loss.replace(0,np.nan)))
        fig_hist.add_trace(go.Scatter(x=df_hist.index, y=rsi_vals, mode="lines",
            line=dict(color="#a78bfa",width=1.5), name="RSI"), row=2, col=1)
        fig_hist.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=0.8, row=2, col=1)
        fig_hist.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=0.8, row=2, col=1)
        ema12 = df_hist["Close"].ewm(span=12).mean()
        ema26 = df_hist["Close"].ewm(span=26).mean()
        macd = ema12 - ema26
        macd_sig = macd.ewm(span=9).mean()
        macd_hist_vals = macd - macd_sig
        fig_hist.add_trace(go.Scatter(x=df_hist.index, y=macd, mode="lines",
            line=dict(color="#06b6d4",width=1.2), name="MACD"), row=3, col=1)
        fig_hist.add_trace(go.Scatter(x=df_hist.index, y=macd_sig, mode="lines",
            line=dict(color="#f59e0b",width=1.2), name="Signal"), row=3, col=1)
        fig_hist.add_trace(go.Bar(x=df_hist.index, y=macd_hist_vals,
            marker_color=["#10b981" if v>=0 else "#ef4444" for v in macd_hist_vals],
            name="Histogram", opacity=0.7), row=3, col=1)
        vcols2 = ["#10b981" if c>=o else "#ef4444" for c,o in zip(df_hist["Close"],df_hist["Open"])]
        fig_hist.add_trace(go.Bar(x=df_hist.index, y=df_hist["Volume"],
            marker_color=vcols2, name="Volume", opacity=0.8), row=4, col=1)
        fig_hist.update_layout(**_chart_layout(height=680))
        fig_hist.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5 — STOCK EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🔎 Stock Explorer")
    st.markdown('<div style="color:#334155;font-size:0.82rem;margin-bottom:16px;">Search any NSE stock. Get full analysis: chart, indicators, predictions, sector comparison, and trading signals.</div>', unsafe_allow_html=True)

    ex_c1, ex_c2, ex_c3, ex_c4 = st.columns([2, 1, 1, 1])
    with ex_c1:
        ex_sym = st.selectbox("Search Stock", ALL_SYMBOLS_CLEAN,
                               index=ALL_SYMBOLS_CLEAN.index("RELIANCE") if "RELIANCE" in ALL_SYMBOLS_CLEAN else 0,
                               key="ex_sym")
    with ex_c2:
        ex_period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3, key="ex_period")
    with ex_c3:
        ex_mode = st.selectbox("Mode", ["📡 Intraday", "📈 Swing", "💼 Long-Term"], key="ex_mode")
    with ex_c4:
        ex_btn = st.button("🔎 Analyse", type="primary", use_container_width=True)

    if ex_btn:
        with st.spinner(f"Analysing {ex_sym}..."):
            df_ex = DataService.fetch_ohlcv(f"{ex_sym}.NS", period=ex_period)
            df_ex_1y = DataService.fetch_ohlcv(f"{ex_sym}.NS", period="1y")
            pred_ex = PredictionEngine.predict_next_day(df_ex_1y) if len(df_ex_1y) >= 100 else {}
            multi_ex = PredictionEngine.predict_multi_horizon(df_ex_1y) if len(df_ex_1y) >= 100 else {}
            sector_ex = SYMBOL_TO_SECTOR.get(ex_sym, "Other")
            factors_ex = StockMetadata.get_global_factors(sector_ex)
            st.session_state["ex_result"] = {
                "symbol": ex_sym, "df": df_ex, "pred": pred_ex,
                "multi": multi_ex, "sector": sector_ex, "factors": factors_ex,
                "period": ex_period, "mode": ex_mode,
            }

    if "ex_result" in st.session_state:
        r = st.session_state["ex_result"]
        df_ex = r["df"]
        sym_ex = r["symbol"]
        pred_ex = r["pred"]
        sector_ex = r["sector"]
        factors_ex = r["factors"]

        if df_ex.empty:
            st.error(f"No data for {sym_ex}")
        else:
            price_ex = float(df_ex["Close"].iloc[-1])
            price_start = float(df_ex["Close"].iloc[0])
            total_ret = (price_ex / price_start - 1) * 100
            high_52 = float(df_ex["High"].max())
            low_52 = float(df_ex["Low"].min())
            avg_vol = float(df_ex["Volume"].mean())
            returns_ex = df_ex["Close"].pct_change().dropna()
            vol_ann = float(returns_ex.std() * np.sqrt(252) * 100)
            rsi_ex = 50.0
            if len(df_ex) >= 14:
                delta_ex = df_ex["Close"].diff()
                gain_ex = delta_ex.where(delta_ex>0,0).rolling(14).mean()
                loss_ex = (-delta_ex.where(delta_ex<0,0)).rolling(14).mean()
                rsi_series = 100 - (100/(1+gain_ex/loss_ex.replace(0,np.nan)))
                rsi_ex = float(rsi_series.iloc[-1]) if not rsi_series.empty else 50.0

            # Key stats
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Current Price", f"₹{price_ex:,.2f}")
            m2.metric("Period Return", f"{total_ret:+.1f}%")
            m3.metric("52W High", f"₹{high_52:,.2f}")
            m4.metric("52W Low", f"₹{low_52:,.2f}")
            m5.metric("Avg Volume", f"{avg_vol/1e6:.1f}M")
            m6.metric("Ann. Volatility", f"{vol_ann:.1f}%")

            # Trading signals
            st.markdown("---")
            mode_str = r["mode"]
            is_intraday = "Intraday" in mode_str
            is_longterm = "Long" in mode_str

            if pred_ex and "error" not in pred_ex:
                report_ex = StockMetadata.generate_report(
                    symbol=sym_ex, price=price_ex, sector=sector_ex,
                    confidence=pred_ex["confidence"], direction=pred_ex["direction"],
                    predicted_return=pred_ex["predicted_return"],
                    volatility=vol_ann/100, rsi=rsi_ex,
                    entry=price_ex,
                    target=pred_ex["predicted_price"],
                    stop_loss=price_ex * 0.97,
                    mode="intraday" if is_intraday else ("swing" if not is_longterm else "longterm"),
                )
                sig_color = "#10b981" if report_ex["buy_sell"]=="BUY" else "#ef4444"
                st.markdown(f"""
                <div class="trade-card" style="margin-bottom:16px;">
                    <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:10px;'>
                        <span style='font-size:1.2rem;font-weight:900;color:#f1f5f9;'>{sym_ex}</span>
                        <span class="badge-sector">{sector_ex}</span>
                        <span style='background:{sig_color}22;color:{sig_color};padding:4px 14px;border-radius:8px;font-weight:800;font-size:0.9rem;'>{report_ex["buy_sell"]}</span>
                        <span style='color:#94a3b8;font-size:0.82rem;'>{report_ex["risk_level"]}</span>
                    </div>
                    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;'>
                        <div><div style='color:#475569;font-size:0.68rem;text-transform:uppercase;'>Entry</div><div style='color:#f1f5f9;font-weight:700;'>₹{report_ex["entry"]:,.2f}</div></div>
                        <div><div style='color:#475569;font-size:0.68rem;text-transform:uppercase;'>Target</div><div style='color:#10b981;font-weight:700;'>₹{report_ex["target"]:,.2f} (+{report_ex["reward_pct"]:.1f}%)</div></div>
                        <div><div style='color:#475569;font-size:0.68rem;text-transform:uppercase;'>Stop Loss</div><div style='color:#ef4444;font-weight:700;'>₹{report_ex["stop_loss"]:,.2f} (-{report_ex["risk_pct"]:.1f}%)</div></div>
                        <div><div style='color:#475569;font-size:0.68rem;text-transform:uppercase;'>Holding</div><div style='color:#f59e0b;font-weight:700;font-size:0.82rem;'>{report_ex["holding_duration"]}</div></div>
                    </div>
                    <div style='margin-top:10px;color:#334155;font-size:0.75rem;'>
                        Confidence: <b style='color:#3b82f6;'>{pred_ex["confidence"]:.0%}</b> &nbsp;|&nbsp;
                        Next-day: <b style='color:{"#10b981" if pred_ex["predicted_return"]>0 else "#ef4444"};'>{pred_ex["predicted_return"]:+.2f}%</b> &nbsp;|&nbsp;
                        RSI: <b style='color:#a78bfa;'>{rsi_ex:.0f}</b> &nbsp;|&nbsp;
                        Volatility: <b style='color:#f59e0b;'>{vol_ann:.1f}%</b>
                    </div>
                </div>""", unsafe_allow_html=True)

            # Full chart
            fig_ex = make_subplots(rows=4, cols=1, row_heights=[0.50,0.17,0.17,0.16],
                                    shared_xaxes=True, vertical_spacing=0.02)
            fig_ex.add_trace(go.Candlestick(
                x=df_ex.index, open=df_ex["Open"], high=df_ex["High"],
                low=df_ex["Low"], close=df_ex["Close"], name="Price",
                increasing=dict(line=dict(color="#10b981"),fillcolor="#10b981"),
                decreasing=dict(line=dict(color="#ef4444"),fillcolor="#ef4444"),
            ), row=1, col=1)
            for w, color in [(20,"#f59e0b"),(50,"#6366f1"),(200,"#ec4899")]:
                if len(df_ex) >= w:
                    fig_ex.add_trace(go.Scatter(x=df_ex.index, y=df_ex["Close"].rolling(w).mean(),
                        mode="lines", line=dict(color=color,width=1), name=f"MA{w}"), row=1, col=1)
            # Bollinger
            ma20_ex = df_ex["Close"].rolling(20).mean()
            std20_ex = df_ex["Close"].rolling(20).std()
            fig_ex.add_trace(go.Scatter(x=df_ex.index, y=ma20_ex+2*std20_ex, mode="lines",
                line=dict(color="#1e3a5f",width=0.8,dash="dash"), name="BB Upper", showlegend=False), row=1, col=1)
            fig_ex.add_trace(go.Scatter(x=df_ex.index, y=ma20_ex-2*std20_ex, mode="lines",
                line=dict(color="#1e3a5f",width=0.8,dash="dash"), name="BB Lower",
                fill="tonexty", fillcolor="rgba(30,58,95,0.08)", showlegend=False), row=1, col=1)
            # RSI
            delta_ex2 = df_ex["Close"].diff()
            gain_ex2 = delta_ex2.where(delta_ex2>0,0).rolling(14).mean()
            loss_ex2 = (-delta_ex2.where(delta_ex2<0,0)).rolling(14).mean()
            rsi_ex2 = 100 - (100/(1+gain_ex2/loss_ex2.replace(0,np.nan)))
            fig_ex.add_trace(go.Scatter(x=df_ex.index, y=rsi_ex2, mode="lines",
                line=dict(color="#a78bfa",width=1.5), name="RSI"), row=2, col=1)
            fig_ex.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=0.8, row=2, col=1)
            fig_ex.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=0.8, row=2, col=1)
            # MACD
            ema12_ex = df_ex["Close"].ewm(span=12).mean()
            ema26_ex = df_ex["Close"].ewm(span=26).mean()
            macd_ex = ema12_ex - ema26_ex
            macd_sig_ex = macd_ex.ewm(span=9).mean()
            macd_hist_ex = macd_ex - macd_sig_ex
            fig_ex.add_trace(go.Scatter(x=df_ex.index, y=macd_ex, mode="lines",
                line=dict(color="#06b6d4",width=1.2), name="MACD"), row=3, col=1)
            fig_ex.add_trace(go.Scatter(x=df_ex.index, y=macd_sig_ex, mode="lines",
                line=dict(color="#f59e0b",width=1.2), name="Signal"), row=3, col=1)
            fig_ex.add_trace(go.Bar(x=df_ex.index, y=macd_hist_ex,
                marker_color=["#10b981" if v>=0 else "#ef4444" for v in macd_hist_ex],
                name="Hist", opacity=0.7), row=3, col=1)
            vcols_ex = ["#10b981" if c>=o else "#ef4444" for c,o in zip(df_ex["Close"],df_ex["Open"])]
            fig_ex.add_trace(go.Bar(x=df_ex.index, y=df_ex["Volume"],
                marker_color=vcols_ex, name="Volume", opacity=0.8), row=4, col=1)
            fig_ex.update_layout(**_chart_layout(f"{sym_ex} — {r['period']} Chart", height=680))
            fig_ex.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_ex, use_container_width=True)

            # Global factors
            st.markdown("---")
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                st.markdown(f"### 🌍 {sector_ex} — {factors_ex['theme']}")
                for f in factors_ex["positive"]:
                    st.markdown(f'<span class="factor-pos">✅ {f}</span>', unsafe_allow_html=True)
            with col_f2:
                st.markdown("### ⚠️ Risk Factors")
                for f in factors_ex["negative"]:
                    st.markdown(f'<span class="factor-neg">⚠️ {f}</span>', unsafe_allow_html=True)

            # Multi-horizon summary
            if r["multi"]:
                st.markdown("---")
                st.markdown("### 📈 Price Forecast Summary")
                horizon_labels2 = {"10_days":"10D","20_days":"20D","30_days":"30D","3_months":"3M","6_months":"6M"}
                hcols = st.columns(5)
                for col, (key, label) in zip(hcols, horizon_labels2.items()):
                    pred = r["multi"].get(key, {})
                    if pred and "error" not in pred:
                        ret = pred["predicted_return"]
                        ret_color = "#10b981" if ret > 0 else "#ef4444"
                        col.markdown(f'<div class="stat-card" style="text-align:center;"><div style="font-size:0.68rem;color:#334155;">{label}</div><div style="font-size:1.1rem;font-weight:900;color:{ret_color};">{ret:+.1f}%</div><div style="font-size:0.72rem;color:#475569;">₹{pred["predicted_price"]:,.0f}</div><div style="font-size:0.65rem;color:#3b82f6;">{pred["confidence"]:.0%}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 6 — REPORTS
# ═══════════════════════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### 📋 Structured Reports")
    st.markdown('<div style="color:#334155;font-size:0.82rem;margin-bottom:16px;">Generate detailed reports for any stock or sector. Includes risk assessment, predictions, and global factor analysis.</div>', unsafe_allow_html=True)

    rp_c1, rp_c2, rp_c3 = st.columns([2, 1, 1])
    with rp_c1:
        rp_sym = st.selectbox("Select Stock", ALL_SYMBOLS_CLEAN,
                               index=ALL_SYMBOLS_CLEAN.index("RELIANCE") if "RELIANCE" in ALL_SYMBOLS_CLEAN else 0,
                               key="rp_sym")
    with rp_c2:
        rp_type = st.selectbox("Report Type", ["📊 Full Analysis", "🪙 Penny Stock Report", "💼 Long-Term Investment", "📡 Intraday Report"], key="rp_type")
    with rp_c3:
        rp_btn = st.button("📋 Generate Report", type="primary", use_container_width=True)

    if rp_btn:
        with st.spinner(f"Generating report for {rp_sym}..."):
            df_rp = DataService.fetch_ohlcv(f"{rp_sym}.NS", period="1y")
            pred_rp = PredictionEngine.predict_next_day(df_rp) if len(df_rp) >= 100 else {}
            multi_rp = PredictionEngine.predict_multi_horizon(df_rp) if len(df_rp) >= 100 else {}
            sector_rp = SYMBOL_TO_SECTOR.get(rp_sym, "Other")
            factors_rp = StockMetadata.get_global_factors(sector_rp)
            price_rp = float(df_rp["Close"].iloc[-1]) if not df_rp.empty else 0
            returns_rp = df_rp["Close"].pct_change().dropna() if not df_rp.empty else pd.Series()
            vol_rp = float(returns_rp.std() * np.sqrt(252)) if len(returns_rp) > 0 else 0.2
            rsi_rp = 50.0
            if len(df_rp) >= 14:
                d = df_rp["Close"].diff()
                g = d.where(d>0,0).rolling(14).mean()
                l = (-d.where(d<0,0)).rolling(14).mean()
                rsi_rp = float((100 - (100/(1+g/l.replace(0,np.nan)))).iloc[-1])
            report_rp = StockMetadata.generate_report(
                symbol=rp_sym, price=price_rp, sector=sector_rp,
                confidence=pred_rp.get("confidence", 0.5) if pred_rp else 0.5,
                direction=pred_rp.get("direction", "NEUTRAL") if pred_rp else "NEUTRAL",
                predicted_return=pred_rp.get("predicted_return", 0) if pred_rp else 0,
                volatility=vol_rp, rsi=rsi_rp,
                entry=price_rp,
                target=pred_rp.get("predicted_price", price_rp*1.05) if pred_rp else price_rp*1.05,
                stop_loss=price_rp * 0.97,
                mode="intraday" if "Intraday" in rp_type else ("swing" if "Long" not in rp_type else "longterm"),
            )
            st.session_state["rp_result"] = {
                "symbol": rp_sym, "report": report_rp, "pred": pred_rp,
                "multi": multi_rp, "factors": factors_rp, "sector": sector_rp,
                "df": df_rp, "type": rp_type, "vol": vol_rp, "rsi": rsi_rp,
            }

    if "rp_result" in st.session_state:
        r = st.session_state["rp_result"]
        rep = r["report"]
        pred_rp = r["pred"]
        factors_rp = r["factors"]
        is_penny_rp = rep["is_penny"]

        # Report header
        sig_color = "#10b981" if rep["buy_sell"]=="BUY" else "#ef4444"
        penny_warn = ""
        if is_penny_rp:
            penny_warn = """
            <div style='background:rgba(124,58,237,0.1);border:1px solid #6d28d9;border-radius:10px;padding:12px;margin-bottom:12px;'>
                <b style='color:#c4b5fd;'>🪙 PENNY STOCK REPORT</b>
                <div style='color:#7c3aed;font-size:0.78rem;margin-top:4px;'>
                    This is a penny stock (≤₹50). High risk. Use strict stop-loss. Max 2-5% capital allocation.
                </div>
            </div>"""

        st.markdown(f"""
        {penny_warn}
        <div class="trade-card">
            <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:14px;'>
                <span style='font-size:1.5rem;font-weight:900;color:#f1f5f9;'>{rep["symbol"]}</span>
                <span class="badge-sector">{rep["sector"]}</span>
                <span style='background:{sig_color}22;color:{sig_color};padding:5px 16px;border-radius:8px;font-weight:800;'>{rep["buy_sell"]}</span>
                <span style='color:#94a3b8;font-size:0.82rem;'>{rep["risk_level"]}</span>
                {'<span class="badge-penny">🪙 PENNY</span>' if is_penny_rp else ""}
            </div>
            <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:16px;'>
                <div>
                    <div style='color:#475569;font-size:0.68rem;text-transform:uppercase;font-weight:700;margin-bottom:6px;'>📊 Price Levels</div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Current: <b style='color:#f1f5f9;'>₹{rep["entry"]:,.2f}</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Target: <b style='color:#10b981;'>₹{rep["target"]:,.2f} (+{rep["reward_pct"]:.1f}%)</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Stop Loss: <b style='color:#ef4444;'>₹{rep["stop_loss"]:,.2f} (-{rep["risk_pct"]:.1f}%)</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Holding: <b style='color:#f59e0b;'>{rep["holding_duration"]}</b></div>
                </div>
                <div>
                    <div style='color:#475569;font-size:0.68rem;text-transform:uppercase;font-weight:700;margin-bottom:6px;'>🤖 ML Prediction</div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Direction: <b style='color:{"#10b981" if rep["direction"]=="UP" else "#ef4444"};'>{rep["direction"]}</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Expected Return: <b style='color:{"#10b981" if rep["predicted_return"]>0 else "#ef4444"};'>{rep["predicted_return"]:+.2f}%</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Confidence: <b style='color:#3b82f6;'>{rep["confidence"]:.0%}</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Signal: <b style='color:#f59e0b;'>{rep["signal"]}</b></div>
                </div>
                <div>
                    <div style='color:#475569;font-size:0.68rem;text-transform:uppercase;font-weight:700;margin-bottom:6px;'>📈 Risk Metrics</div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Risk Level: <b>{rep["risk_level"]}</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Volatility: <b style='color:#f59e0b;'>{rep["vol"]*100:.1f}%</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>RSI: <b style='color:#a78bfa;'>{rep["rsi"]:.0f}</b></div>
                    <div style='color:#94a3b8;font-size:0.82rem;'>Price Class: <b style='color:#94a3b8;'>{rep["price_class"]}</b></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Multi-horizon table
        if r["multi"]:
            st.markdown("---")
            st.markdown("### 📅 Price Forecast by Horizon")
            horizon_data = []
            for key, label in [("10_days","10 Days"),("20_days","20 Days"),("30_days","30 Days"),("3_months","3 Months"),("6_months","6 Months")]:
                pred = r["multi"].get(key, {})
                if pred and "error" not in pred:
                    horizon_data.append({
                        "Horizon": label,
                        "Direction": pred["direction"],
                        "Expected Return": f"{pred['predicted_return']:+.2f}%",
                        "Predicted Price": f"₹{pred['predicted_price']:,.2f}",
                        "Price Range": f"₹{pred['price_low']:,.0f} – ₹{pred['price_high']:,.0f}",
                        "Confidence": f"{pred['confidence']:.0%}",
                    })
            if horizon_data:
                st.dataframe(pd.DataFrame(horizon_data), use_container_width=True, hide_index=True)

        # Global factors
        st.markdown("---")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.markdown(f"### 🌍 Sector Theme: {factors_rp['theme']}")
            for f in factors_rp["positive"]:
                st.markdown(f'<span class="factor-pos">✅ {f}</span>', unsafe_allow_html=True)
        with col_f2:
            st.markdown("### ⚠️ Risk Factors")
            for f in factors_rp["negative"]:
                st.markdown(f'<span class="factor-neg">⚠️ {f}</span>', unsafe_allow_html=True)

        # Price chart
        if not r["df"].empty:
            st.markdown("---")
            df_rp_chart = r["df"]
            fig_rp = go.Figure()
            fig_rp.add_trace(go.Candlestick(
                x=df_rp_chart.index, open=df_rp_chart["Open"], high=df_rp_chart["High"],
                low=df_rp_chart["Low"], close=df_rp_chart["Close"], name="Price",
                increasing=dict(line=dict(color="#10b981"),fillcolor="#10b981"),
                decreasing=dict(line=dict(color="#ef4444"),fillcolor="#ef4444"),
            ))
            for w, color in [(20,"#f59e0b"),(50,"#6366f1"),(200,"#ec4899")]:
                if len(df_rp_chart) >= w:
                    fig_rp.add_trace(go.Scatter(x=df_rp_chart.index, y=df_rp_chart["Close"].rolling(w).mean(),
                        mode="lines", line=dict(color=color,width=1), name=f"MA{w}"))
            fig_rp.update_layout(**_chart_layout(f"{r['symbol']} — 1 Year Chart", height=420))
            fig_rp.update_layout(xaxis_rangeslider_visible=False)
            st.plotly_chart(fig_rp, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;color:#1e3a5f;font-size:0.68rem;padding:4px 0;flex-wrap:wrap;gap:8px;'>
    <span>⚡ QuantSignal India v5.0 &nbsp;|&nbsp; {len(INTRADAY_STOCKS)} stocks &nbsp;|&nbsp; 14 sectors &nbsp;|&nbsp; 6 tabs</span>
    <span>Data: Yahoo Finance (15-min delayed) &nbsp;|&nbsp; Not financial advice &nbsp;|&nbsp; Always use stop-loss</span>
    <span>{datetime.now().strftime("%d %b %Y %I:%M %p")}</span>
</div>""", unsafe_allow_html=True)

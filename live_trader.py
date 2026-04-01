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

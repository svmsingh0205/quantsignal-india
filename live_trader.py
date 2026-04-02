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
from datetime import datetime, date, time as dtime, timedelta, timezone
import sys, os, time as _time

# ── IST = UTC + 5:30 — hardcoded offset, works on every server/cloud ─────────
_IST = timezone(timedelta(hours=5, minutes=30))

def _now_ist() -> datetime:
    """Return current datetime in IST (UTC+5:30). No pytz/zoneinfo needed."""
    return datetime.now(timezone.utc).astimezone(_IST).replace(tzinfo=None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.engines.intraday_engine import IntradayEngine, get_market_status
from backend.engines.data_service import DataService
from backend.engines.prediction_engine import PredictionEngine
from backend.engines.stock_metadata import StockMetadata, GLOBAL_FACTORS, PENNY_MAX_PRICE
from backend.engines.risk_engine import RiskEngine
from backend.intraday_config import INTRADAY_STOCKS, SECTOR_GROUPS

# New universe — graceful fallback if import fails
try:
    from backend.engines.universe import MASTER_UNIVERSE, SECTOR_UNIVERSE, UniverseEngine, PennyStockEngine
except Exception:
    MASTER_UNIVERSE = []
    SECTOR_UNIVERSE = SECTOR_GROUPS
    UniverseEngine = None
    PennyStockEngine = None

# Market data router — lazy, only used for status badge
_data_router = None
def _get_router():
    global _data_router
    if _data_router is None:
        try:
            from backend.engines.market_data_router import router
            _data_router = router
        except Exception:
            _data_router = None
    return _data_router

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

# Expanded universe: MASTER_UNIVERSE (1,500+) merged with INTRADAY_STOCKS
_combined_universe = list(dict.fromkeys(
    [s.replace(".NS", "") for s in INTRADAY_STOCKS] + MASTER_UNIVERSE
))
ALL_SYMBOLS_CLEAN = sorted(set(_combined_universe))

# ── EXTENDED NSE/BSE SEARCH LIST (all liquid symbols) ─────────────────────────
# Combines the intraday universe + common NSE symbols not in intraday list
_EXTRA_NSE = [
    "NIFTY50","SENSEX","BANKNIFTY","FINNIFTY",
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "KOTAKBANK","LT","ITC","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI","HCLTECH",
    "SUNPHARMA","TITAN","ULTRACEMCO","WIPRO","NESTLEIND","BAJAJFINSV","TATAMOTORS",
    "POWERGRID","NTPC","ONGC","JSWSTEEL","M&M","TATASTEEL","ADANIENT","ADANIPORTS",
    "COALINDIA","GRASIM","TECHM","INDUSINDBK","HINDALCO","DRREDDY","DIVISLAB",
    "CIPLA","BPCL","EICHERMOT","BRITANNIA","APOLLOHOSP","TATACONSUM","HEROMOTOCO",
    "SBILIFE","HDFCLIFE","BAJAJ-AUTO","UPL","SHRIRAMFIN","ADANIGREEN","AMBUJACEM",
    "AUROPHARMA","BANKBARODA","BERGEPAINT","BOSCHLTD","CANBK","CHOLAFIN","COLPAL",
    "DABUR","DLF","GAIL","GODREJCP","HAVELLS","ICICIPRULI","INDHOTEL","INDUSTOWER",
    "IOC","IRCTC","JINDALSTEL","LICI","LODHA","LUPIN","MARICO","MCDOWELL-N",
    "MUTHOOTFIN","NAUKRI","NHPC","NMDC","OBEROIRLTY","OFSS","PAGEIND","PIDILITIND",
    "PIIND","PNB","RECLTD","SAIL","SIEMENS","SRF","TATAPOWER","TORNTPHARM","TRENT",
    "UNIONBANK","VBL","VEDL","ZOMATO","ZYDUSLIFE","PFC","HAL","BEL","BDL","GRSE",
    "MAZDOCK","MTAR","BEML","COCHINSHIP","RVNL","IRFC","IRCON","KECL","NBCC",
    "SJVN","SUZLON","INOXWIND","YESBANK","IDEA","RPOWER","JPPOWER","UCOBANK",
    "INDIANB","BANKINDIA","CENTRALBK","MAHABANK","PSB","JKBANK","PERSISTENT",
    "COFORGE","KPITTECH","TATAELXSI","LTIM","MPHASIS","ZENSARTECH","MASTEK",
    "POLICYBZR","PAYTM","JIOFIN","ANGELONE","LAURUSLABS","GRANULES","NATCOPHARM",
    "IPCALAB","GLENMARK","ALKEM","FORTIS","MAXHEALTH","METROPOLIS","DEEPAKNTR",
    "NAVINFLUOR","FLUOROCHEM","CLEAN","FINEORG","TATACHEM","GNFC","COROMANDEL",
    "PRESTIGE","BRIGADE","SOBHA","PHOENIXLTD","SHREECEM","JKCEMENT","RAMCOCEM",
    "TATACOMM","HFCL","STLTECH","RAILTEL","ROUTE","TANLA","OLECTRA","ASHOKLEY",
    "TVSMOTOR","BALKRISIND","APOLLOTYRE","MOTHERSON","BHARATFORG","EXIDEIND",
    "AMARARAJA","EMAMILTD","JYOTHYLAB","RADICO","BIKAJI","DEVYANI","WESTLIFE",
    "JUBLFOOD","MANAPPURAM","LICHSGFIN","PNBHOUSING","CANFINHOME","CREDITACC",
    "UJJIVANSFB","EQUITASBNK","SBICARD","HDFCAMC","NIPPONLIFE","AAVAS","HOMEFIRST",
    "NATIONALUM","HINDCOPPER","MOIL","WELCORP","APLAPOLLO","JINDALSAW","JSWINFRA",
    "RATNAMANI","CONCOR","TIINDIA","GPPL","ADANIGAS","IGL","MGL","GUJGASLTD",
    "MRPL","PETRONET","OIL","TORNTPOWER","JSWENERGY","CESC","TRIL","GMRINFRA",
    "IRB","ASHOKA","KNRCON","PSPPROJECT","CAPACITE","SADBHAV","HGINFRA","PNCINFRA",
    "NCC","KALPATPOWR","DATAPATTNS","MIDHANI","PARAS","DPSL","ZENTEC","ASTRA",
    "IDEAFORGE","SOLARA","ATUL","VINATI","GALAXYSURF","ROSSARI","ANUPAM","NOCIL",
    "NEOGEN","PCBL","IGPL","BASF","AKZOINDIA","KANSAINER","BERGER","PIDILITIND",
    "GODREJPROP","MAHLIFE","KOLTEPATIL","SUNTECK","NUVOCO","DALMIA","JKLAKSHMI",
    "BIRLACORPN","HEIDELBERG","PRISM","ORIENT","NAUKRI","INFOEDGE","JUSTDIAL",
    "INDIAMART","CARTRADE","EASEMYTRIP","NYKAA","SWIGGY","ONMOBILE","GTLINFRA",
    "LALPATHLAB","KRSNAA","VIJAYA","THYROCARE","BIOCON","STRIDES","SEQUENT","DIVI",
    "ABBOTINDIA","PFIZER","SANOFI","GLAXO","ERIS","JBCHEPHARM","AJANTPHARM",
    "SUVEN","IPCALAB","TORNTPHARM","ZYDUSLIFE","ALKEM","NATCOPHARM","GRANULES",
    "LAURUSLABS","GLENMARK","AUROPHARMA","LUPIN","CIPLA","DRREDDY","SUNPHARMA",
    "DIVISLAB","APOLLOHOSP","FORTIS","MAXHEALTH","METROPOLIS","THYROCARE",
]
_ALL_SEARCH_SYMBOLS = sorted(set(ALL_SYMBOLS_CLEAN + [s for s in _EXTRA_NSE if s not in ALL_SYMBOLS_CLEAN]))

def market_open() -> bool:
    now = _now_ist()
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)

def time_to_close() -> str:
    now = _now_ist()
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

def _quick_search(symbol: str) -> dict | None:
    """
    Fetch a real-time price snapshot for any NSE/BSE symbol.
    Returns None (never a fake dict) if data is unavailable or price is zero.
    """
    for suffix in (".NS", ".BO"):
        yf_sym = symbol if symbol.endswith((".NS", ".BO")) else f"{symbol}{suffix}"
        data = DataService.fetch_live_price(yf_sym)
        if data and data.get("price", 0) > 0:
            data["sector"] = SYMBOL_TO_SECTOR.get(symbol, "Other")
            return data
    logger.warning("_quick_search: no valid data for %s", symbol)
    return None

def _render_indicator_grid(indicators: list) -> None:
    """Render a row of indicator cards."""
    import html as _html
    cols = st.columns(min(4, len(indicators)))
    for i, ind in enumerate(indicators):
        col = cols[i % len(cols)]
        score = ind["score"]
        color = "#10b981" if score >= 0.60 else ("#ef4444" if score <= 0.40 else "#f59e0b")
        icon = "🟢" if score >= 0.60 else ("🔴" if score <= 0.40 else "🟡")
        # Escape HTML special chars in dynamic text to prevent rendering issues
        safe_name = _html.escape(str(ind["name"]))
        safe_note = _html.escape(str(ind["note"])[:60])
        safe_signal = _html.escape(str(ind["signal"]))
        safe_trend = _html.escape(str(ind["trend"]))
        ellipsis = "…" if len(str(ind["note"])) > 60 else ""
        with col:
            st.markdown(f"""
            <div class="stat-card" style="text-align:left;margin-bottom:8px;">
                <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;font-weight:700;">{safe_name}</div>
                <div style="font-size:1.2rem;font-weight:900;color:{color};margin:4px 0;">{icon} {score:.0%}</div>
                <div style="font-size:0.72rem;color:#94a3b8;">{safe_signal} · {safe_trend}</div>
                <div style="font-size:0.65rem;color:#334155;margin-top:4px;">{safe_note}{ellipsis}</div>
            </div>""", unsafe_allow_html=True)


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
from backend.engines.data_validator import get_validated_tick, generate_signal, clear_live_cache
from backend.engines.market_context import get_market_context, get_market_bias_score, get_position_size_multiplier
from backend.engines.broker_feed import get_live_tick, get_ohlcv, get_data_source_status, clear_tick_cache, DataSource
from backend.engines.news_engine import get_news_sentiment


# ── _predict_one defined at module level so it's always available ─────────────
def _predict_one(sym: str) -> dict | None:
    """Fetch and predict next-day direction for a single symbol."""
    try:
        df = DataService.fetch_ohlcv(sym, period="1y")
        if df.empty or len(df) < 60:
            return None
        pred = PredictionEngine.predict_next_day(df)
        if not pred or "error" in pred:
            return None
        clean = sym.replace(".NS", "")
        sector = SYMBOL_TO_SECTOR.get(clean, "Other")
        price = pred["current_price"]
        if price <= 0:
            return None
        report = StockMetadata.generate_report(
            symbol=clean, price=price, sector=sector,
            confidence=pred["confidence"], direction=pred["direction"],
            predicted_return=pred["predicted_return"],
            volatility=pred.get("volatility", 20) / 100,
            rsi=pred.get("rsi", 50),
            entry=price, target=pred["predicted_price"],
            stop_loss=price * 0.97, mode="swing",
        )
        return {
            "symbol": clean, "sector": sector,
            "price": price,
            "predicted": pred["predicted_price"],
            "return_pct": pred["predicted_return"],
            "direction": pred["direction"],
            "confidence": pred["confidence"],
            "rsi": pred.get("rsi", 50),
            "volatility": pred.get("volatility", 20),
            "conditions": pred.get("market_conditions", []),
            "risk_level": report["risk_level"],
            "signal": report["signal"],
            "is_penny": price <= PENNY_MAX_PRICE,
            "factors": StockMetadata.get_global_factors(sector),
        }
    except Exception:
        return None


def _scan_one(args):
    """
    Production signal pipeline per symbol:
    1. Live tick from broker (Kite → Upstox → Yahoo)
    2. Freshness + cross-source validation
    3. OHLCV bars for TA scoring
    4. VWAP position check
    5. Market context bias adjustment
    6. Structured reasoning output
    """
    sym, capital, max_price, min_conf, penny_only, mkt_bias, psm = args
    try:
        # ── Step 1: Live validated tick ───────────────────────────────────────
        clean = sym.replace(".NS", "")
        tick = get_live_tick(clean)

        # ── Step 2: OHLCV bars for TA ─────────────────────────────────────────
        engine = IntradayEngine(capital=capital)
        df = engine.fetch_intraday(sym, period="5d", interval="5m")
        if df.empty or len(df) < 30:
            return None

        # ── Step 3: Price — validated tick > bar close ────────────────────────
        if tick and tick.is_valid and tick.ltp > 0:
            price          = tick.ltp
            validated_vwap = tick.vwap
            data_source    = tick.source
            bid            = tick.bid
            ask            = tick.ask
            live_volume    = tick.volume
        else:
            price          = float(df["Close"].iloc[-1])
            validated_vwap = 0.0
            data_source    = DataSource.YAHOO
            bid = ask = live_volume = 0

        if price > max_price:
            return None
        if penny_only and price > PENNY_MAX_PRICE:
            return None
        # Hard reject: never generate a report with zero or invalid price
        if price <= 0:
            logger.warning("_scan_one: price=0 for %s — skipping", sym)
            return None

        # ── Step 4: TA scoring ────────────────────────────────────────────────
        df = engine.add_indicators(df)
        sc = engine.score_stock(df)

        effective_vwap = validated_vwap if validated_vwap > 0 else sc["vwap"]

        # Step 5: Dynamic ATR from daily bars (fixes identical 1.33x RR bug)
        # 5m ATR is near-zero for many stocks -> always hits 0.5% floor -> RR=1.33x
        # Solution: use 14-day daily ATR which reflects real stock volatility
        df_daily_atr = DataService.fetch_ohlcv(sym, period="3mo", interval="1d")
        if not df_daily_atr.empty and len(df_daily_atr) >= 14:
            _h = df_daily_atr["High"]
            _l = df_daily_atr["Low"]
            _pc = df_daily_atr["Close"].shift(1)
            _tr = pd.concat([_h - _l, (_h - _pc).abs(), (_l - _pc).abs()], axis=1).max(axis=1)
            _atr_daily = float(_tr.rolling(14).mean().iloc[-1])
            atr = _atr_daily if (not np.isnan(_atr_daily) and _atr_daily > price * 0.005) else max(sc["atr"], price * 0.005)
            # Real volatility for report
            _vol_real = float(df_daily_atr["Close"].pct_change().std() * (252 ** 0.5))
        else:
            atr = max(sc["atr"], price * 0.005)
            _vol_real = 0.20
        # Step 6: Sector strength (per-sector geo score, not static 0.2)
        sector = SYMBOL_TO_SECTOR.get(clean, "Other")
        _GEO_MAP = {
            "🛡️ Defence": 0.85, "🏦 PSU Banks": 0.70, "🏗️ Infra/Rail": 0.80,
            "⚡ Energy": 0.72, "💻 IT/Tech": 0.68, "💊 Pharma": 0.65,
            "⚙️ Metals": 0.60, "🚗 Auto/EV": 0.63, "🛒 FMCG": 0.55,
            "💰 Finance": 0.67, "🧪 Chemicals": 0.62, "🏠 Realty/Cement": 0.58,
            "📡 Telecom": 0.60, "📈 Small/Mid Cap": 0.55,
        }
        geo_score = _GEO_MAP.get(sector, 0.55)

        # Step 7: Volume confirmation (0-0.25 contribution)
        vol_ratio = sc.get("vol_ratio", 1.0)
        vol_contrib = float(np.clip((vol_ratio - 1.0) / 4.0, 0.0, 0.25))

        # Step 8: VWAP position (proportional, not binary)
        vwap_position = "at VWAP"
        vwap_contrib = 0.0
        if effective_vwap > 0:
            vwap_pct = (price - effective_vwap) / effective_vwap
            vwap_contrib = float(np.clip(vwap_pct * 8, -0.08, 0.08))
            if vwap_pct > 0.002:
                vwap_position = f"above VWAP +{vwap_pct*100:.2f}% ✅"
            elif vwap_pct < -0.002:
                vwap_position = f"below VWAP {vwap_pct*100:.2f}% ⚠️"

        # Step 9: Composite confidence — weighted, per-stock, dynamic
        # TA 40% | Volume 20% | Sector 15% | VWAP 15% | Macro 10%
        mkt_contrib = (mkt_bias - 0.5) * 0.10
        adjusted_score = float(np.clip(
            0.40 * sc["score"]
            + 0.20 * (0.5 + vol_contrib)
            + 0.15 * geo_score
            + 0.15 * (0.5 + vwap_contrib)
            + 0.10 * (0.5 + mkt_contrib),
            0, 1
        ))

        # Step 10: Position sizing with real ATR
        effective_capital = capital * psm
        entry = round(price, 2)
        sl    = round(max(price - 1.5 * atr, price * 0.93), 2)
        t1    = round(price + 2.0 * atr, 2)
        t2    = round(price + 3.0 * atr, 2)
        risk  = entry - sl
        rr    = round((t1 - entry) / risk, 2) if risk > 0 else 0
        qty   = max(1, int(effective_capital // price))

        report = StockMetadata.generate_report(
            symbol=clean, price=price, sector=sector,
            confidence=adjusted_score,
            direction="UP" if adjusted_score >= 0.55 else "NEUTRAL",
            predicted_return=0, volatility=_vol_real, rsi=sc["rsi"],
            entry=entry, target=t1, stop_loss=sl, mode="intraday",
        )

        # Step 11: Structured reasoning (specific, not generic)
        reasoning_parts = list(sc["reasons"])
        reasoning_parts.append(f"Price {vwap_position}")
        if vol_ratio > 2.0:
            reasoning_parts.append(f"Volume surge {vol_ratio:.1f}x avg — strong momentum")
        elif vol_ratio > 1.3:
            reasoning_parts.append(f"Volume {vol_ratio:.1f}x avg — above average")
        if geo_score >= 0.75:
            reasoning_parts.append(f"Strong sector tailwind: {sector}")
        elif geo_score <= 0.57:
            reasoning_parts.append(f"Weak sector theme: {sector}")
        if mkt_bias >= 0.62:
            reasoning_parts.append("Bullish macro — risk-on")
        elif mkt_bias <= 0.40:
            reasoning_parts.append("Bearish macro — reduce size")
        reasoning_parts.append(f"ATR ₹{atr:.2f} | RR {rr}x | Sector score {geo_score:.0%}")

        quality = ("✅ Live" if data_source in (DataSource.KITE, DataSource.UPSTOX)
                   else "⚡ Yahoo" if data_source == DataSource.YAHOO else "⚠️ Cached")

        return {
            "symbol": clean, "yf_symbol": sym, "sector": sector,
            "price": entry, "qty": qty, "invested": round(qty * price, 2),
            "target_1": t1, "target_2": t2, "stop_loss": sl,
            "confidence": round(adjusted_score, 4), "risk_reward": rr,
            "profit": round(qty * (t1 - entry), 2),
            "loss": round(qty * (entry - sl), 2),
            "rsi": sc["rsi"], "vwap": effective_vwap,
            "vol_ratio": vol_ratio, "supertrend": sc["supertrend"],
            "atr_raw": atr,
            "reasons": reasoning_parts,
            "reasoning": " | ".join(reasoning_parts[:4]),
            "signal": "BUY" if adjusted_score >= min_conf else "WATCH",
            "is_penny": price <= PENNY_MAX_PRICE,
            "risk_level": report["risk_level"],
            "holding": report["holding_duration"],
            "data_quality": quality,
            "data_source": data_source,
            "bid": bid, "ask": ask,
            "df": df,
        }
    except Exception:
        return None


def run_scan(universe, capital, max_price, min_conf, penny_only=False):
    # Clear ALL caches before each scan so prices and bars are always fresh
    clear_tick_cache()
    DataService.clear_intraday_cache()

    # Fetch market context once — shared across all workers
    try:
        ctx = get_market_context()
        mkt_bias = ctx["market_bias_score"]
        psm = ctx["position_size_mult"]
        intraday_ok = ctx["intraday_filter"]
        mkt_label = ctx["market_bias_label"]
    except Exception:
        mkt_bias, psm, intraday_ok, mkt_label = 0.5, 0.75, True, "NEUTRAL"

    prog = st.progress(0, text=f"🚀 Market: {mkt_label} | PSM: {psm:.0%} | Scanning…")
    results, done, total = [], 0, len(universe)
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_scan_one, (sym, capital, max_price, min_conf, penny_only, mkt_bias, psm)): sym
                   for sym in universe}
        for fut in as_completed(futures):
            done += 1
            prog.progress(done / total, text=f"📡 {done}/{total} — {futures[fut].replace('.NS','')}")
            res = fut.result()
            if res:
                results.append(res)
    prog.empty()
    results.sort(key=lambda x: (x["confidence"], x["risk_reward"]), reverse=True)
    return results, mkt_label, mkt_bias

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
    st.markdown(f'<div style="color:#1e3a5f;font-size:0.68rem;margin-top:2px;">{_now_ist().strftime("%d %b %Y  %I:%M %p")} IST</div>', unsafe_allow_html=True)

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
    st.markdown('<div style="color:#475569;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:8px;">📅 Intraday Date</div>', unsafe_allow_html=True)

    _today_ist = _now_ist().date()
    # Next 7 weekdays (Mon–Fri) starting from today
    _weekdays = []
    _d = _today_ist
    while len(_weekdays) < 7:
        if _d.weekday() < 5:
            _weekdays.append(_d)
        _d += timedelta(days=1)

    _date_labels = []
    for _wd in _weekdays:
        if _wd == _today_ist:
            _date_labels.append(f"Today ({_wd.strftime('%d %b')})")
        elif _wd == _today_ist + timedelta(days=1) and _today_ist.weekday() < 4:
            _date_labels.append(f"Tomorrow ({_wd.strftime('%d %b')})")
        else:
            _date_labels.append(_wd.strftime("%a %d %b"))

    _sel_idx = st.selectbox(
        "Scan Date",
        options=list(range(len(_weekdays))),
        format_func=lambda i: _date_labels[i],
        index=0,
        key="intra_date_sel",
    )
    scan_date = _weekdays[_sel_idx]
    if scan_date > _today_ist:
        st.info(f"📅 Pre-market scan for {scan_date.strftime('%d %b %Y')} — uses latest available data as proxy.")

    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
    if auto_refresh:
        refresh_sec = st.select_slider("Interval", [60, 120, 300, 600], value=300,
                                        format_func=lambda x: f"{x//60}m")
    else:
        refresh_sec = 300

    st.markdown("---")
    st.markdown(f'<div style="color:#1e3a5f;font-size:0.68rem;">Universe: <b style="color:#3b82f6;">{len(ALL_SYMBOLS_CLEAN)}</b> stocks | {len(SECTOR_UNIVERSE)} sectors</div>', unsafe_allow_html=True)    # Data source status badge
    _src_status = _get_router().source_status() if _get_router() else {"primary_source": "yahoo", "is_realtime": False}
    _src_label = _src_status["primary_source"].upper()
    _src_color = "#10b981" if _src_status["is_realtime"] else "#f59e0b"
    _src_badge = "🟢 LIVE" if _src_status["is_realtime"] else "🟡 DELAYED"
    st.markdown(f'<div style="color:{_src_color};font-size:0.68rem;margin-top:2px;">Data: {_src_badge} · {_src_label}</div>', unsafe_allow_html=True)
    st.caption("Not financial advice. Paper trade first.")

# ── TOP HEADER ────────────────────────────────────────────────────────────────
universe = get_universe(sector_filter, stock_filter, penny_only)

# ── GLOBAL STOCK SEARCH BAR ───────────────────────────────────────────────────
hcol1, hcol_search, hcol2, hcol3, hcol4 = st.columns([2.8, 2.2, 0.9, 0.9, 0.9])
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
        <b style='color:#2563eb;'>{len(universe)}</b> stocks •
        <b style='color:#2563eb;'>14 sectors</b> •
        {_now_ist().strftime("%d %b %Y %I:%M %p")} IST •
        Scan: <b style='color:#f59e0b;'>{scan_date.strftime("%d %b %Y")}</b>
    </div>""", unsafe_allow_html=True)

with hcol_search:
    st.markdown('<div style="padding-top:4px;">', unsafe_allow_html=True)
    _search_cols = st.columns([3, 1])
    with _search_cols[0]:
        _search_sym = st.selectbox(
            "🔍 Search any NSE/BSE stock",
            options=[""] + _ALL_SEARCH_SYMBOLS,
            index=0,
            key="global_search_sym",
            label_visibility="collapsed",
            placeholder="🔍 Search NSE/BSE stock…",
        )
    with _search_cols[1]:
        _search_btn = st.button("Go", key="global_search_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with hcol2:
    scan_btn = st.button("🔍 SCAN", type="primary", use_container_width=True)
with hcol3:
    best_btn = st.button("⚡ BEST", use_container_width=True)
with hcol4:
    predict_btn = st.button("🔮 PREDICT", use_container_width=True)
    if predict_btn:
        # Flag for Tab 3 to pick up — doesn't run scan here
        st.session_state["_trigger_predict"] = True

# ── SEARCH RESULT CARD ────────────────────────────────────────────────────────
# Only trigger on explicit Go button click — not on every rerun
if _search_btn and _search_sym:
    st.session_state["_search_result_sym"] = _search_sym
    with st.spinner(f"Fetching {_search_sym}…"):
        _snap = _quick_search(_search_sym)
    st.session_state["_search_snap"] = _snap

# Display cached search result
if st.session_state.get("_search_result_sym"):
    _snap = st.session_state.get("_search_snap")
    _displayed_sym = st.session_state["_search_result_sym"]
    if not _snap:
        st.error(f"❌ Live data not available for **{_displayed_sym}**. The symbol may be invalid, delisted, or the market is closed. Please verify the symbol and try again.")
        st.info("💡 Tip: Use the exact NSE symbol (e.g. RELIANCE, TATAMOTORS, HDFCBANK). Indices like NIFTY50 are not tradeable symbols.")
    else:
        _chg_color = "#10b981" if _snap["chg"] >= 0 else "#ef4444"
        _arrow = "▲" if _snap["chg"] >= 0 else "▼"
        _sec = _snap["sector"]
        _factors = StockMetadata.get_global_factors(_sec)
        _sc1, _sc2, _sc3, _sc4, _sc5, _sc6 = st.columns([2, 1, 1, 1, 1, 1])
        with _sc1:
            st.markdown(f"""
            <div style='background:linear-gradient(135deg,#0a1628,#080f1e);border:1px solid #0f2a4a;
                        border-radius:14px;padding:16px 20px;'>
                <div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>
                    <span style='font-size:1.3rem;font-weight:900;color:#f1f5f9;'>{_snap["symbol"]}</span>
                    <span class="badge-sector">{_sec}</span>
                    <span style='color:#334155;font-size:0.65rem;'>Live · refreshes every 60s</span>
                </div>
                <div style='font-size:2rem;font-weight:900;color:#f1f5f9;line-height:1;'>
                    ₹{_snap["price"]:,.2f}
                    <span style='font-size:1rem;color:{_chg_color};margin-left:8px;'>{_arrow} {_snap["chg"]:+.2f}%</span>
                </div>
                <div style='margin-top:8px;font-size:0.72rem;color:#475569;'>
                    Today H: <b style='color:#10b981;'>₹{_snap["today_high"]:,.2f}</b> &nbsp;|&nbsp;
                    Today L: <b style='color:#ef4444;'>₹{_snap["today_low"]:,.2f}</b> &nbsp;|&nbsp;
                    Prev Close: <b style='color:#94a3b8;'>₹{_snap["prev_close"]:,.2f}</b> &nbsp;|&nbsp;
                    Vol: <b style='color:#94a3b8;'>{_snap["volume"]:,}</b>
                </div>
                <div style='margin-top:6px;'>
                    {''.join(f'<span class="factor-pos">✅ {f}</span>' for f in _factors["positive"][:2])}
                </div>
            </div>""", unsafe_allow_html=True)
        _sc2.metric("LTP", f"₹{_snap['price']:,.2f}", f"{_snap['chg']:+.2f}%")
        _sc3.metric("Prev Close", f"₹{_snap['prev_close']:,.2f}")
        _sc4.metric("Today High", f"₹{_snap['today_high']:,.2f}")
        _sc5.metric("Today Low", f"₹{_snap['today_low']:,.2f}")
        _sc6.metric("Volume", f"{_snap['volume']//1000}K")

        # Mini sparkline chart
        if not _snap["df"].empty:
            _fig_mini = go.Figure(go.Scatter(
                x=_snap["df"].index, y=_snap["df"]["Close"], mode="lines",
                line=dict(color=_chg_color, width=2),
                fill="tozeroy", fillcolor=f"rgba({'16,185,129' if _snap['chg']>=0 else '239,68,68'},0.06)",
            ))
            _fig_mini.update_layout(
                **_chart_layout(height=120),
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=True, showgrid=False),
                margin=dict(t=4, b=4, l=40, r=10),
            )
            st.plotly_chart(_fig_mini, use_container_width=True)

        # Deep dive link
        st.markdown(
            f'<div style="text-align:right;margin-top:-8px;">'
            f'<span style="color:#334155;font-size:0.72rem;">→ For full analysis open the </span>'
            f'<b style="color:#3b82f6;">🔍 Deep Dive</b> tab and select <b style="color:#f59e0b;">{_snap["symbol"]}</b>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── TRIGGER SCAN ──────────────────────────────────────────────────────────────
if scan_btn or best_btn:
    trades, mkt_label, mkt_bias = run_scan(universe, capital, max_price, min_conf, penny_only)
    st.session_state["trades"] = trades
    # Store ALL trades — buys re-filtered live on every rerun from current min_conf
    st.session_state["scan_time"] = _now_ist().strftime("%I:%M:%S %p IST")
    st.session_state["scan_date"] = scan_date.strftime("%d %b %Y")
    st.session_state["mkt_label"] = mkt_label
    st.session_state["mkt_bias"] = mkt_bias
    st.session_state["_last_scan_ts"] = _time.time()  # reset auto-refresh countdown

# ── Re-filter buys live on every rerun (slider changes take effect immediately) ─
if "trades" in st.session_state:
    _all_trades = st.session_state["trades"]
    # Re-label signal based on current min_conf slider value
    buys_live = [t for t in _all_trades if t["confidence"] >= min_conf]
    st.session_state["buys"] = buys_live
    st.session_state["pennies"] = [t for t in _all_trades if t["is_penny"] and t["confidence"] >= min_conf]

# ── MAIN TABS ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📡 Live Signals",
    "🪙 Penny Stocks",
    "🔮 Next-Day Picks",
    "📈 Forecast",
    "🔎 Stock Explorer",
    "📋 Reports",
    "🔍 Deep Dive",
    "📊 Market Context",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE INTRADAY SIGNALS
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if "trades" not in st.session_state:
        # Welcome screen — no scan run yet
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
        buys = st.session_state.get("buys", [])
        all_t = st.session_state.get("trades", [])
        scan_time = st.session_state.get("scan_time", "")

        # ── Live price refresh (no re-scan — just updates LTP in existing results) ──
        _rp_col1, _rp_col2, _rp_col3 = st.columns([1, 1, 4])
        with _rp_col1:
            _refresh_prices_btn = st.button("🔄 Refresh Prices", key="refresh_prices_btn", use_container_width=True)
        with _rp_col2:
            _price_age = st.session_state.get("_price_refresh_time", scan_time)
            st.markdown(f'<div style="padding-top:10px;color:#334155;font-size:0.72rem;">Prices as of: <b style="color:#3b82f6;">{_price_age}</b></div>', unsafe_allow_html=True)

        if _refresh_prices_btn and all_t:
            # Re-fetch live LTP for each trade and patch price/entry/targets in-place
            DataService.clear_intraday_cache()
            clear_live_cache()
            _updated = 0
            for t in all_t:
                try:
                    tick = get_validated_tick(t["yf_symbol"])
                    if tick and tick.get("is_valid") and tick["price"] > 0:
                        new_price = tick["price"]
                        old_price = t["price"]
                        if abs(new_price - old_price) / max(old_price, 1) > 0.0001:
                            # Recalculate levels from new price
                            atr = max(t.get("atr_raw", new_price * 0.01), new_price * 0.005)
                            t["price"] = round(new_price, 2)
                            t["target_1"] = round(new_price + 2.0 * atr, 2)
                            t["target_2"] = round(new_price + 3.0 * atr, 2)
                            t["stop_loss"] = round(max(new_price - 1.5 * atr, new_price * 0.93), 2)
                            t["invested"] = round(t["qty"] * new_price, 2)
                            t["data_quality"] = "✅ Live"
                            _updated += 1
                except Exception:
                    pass
            st.session_state["trades"] = all_t
            st.session_state["_price_refresh_time"] = _now_ist().strftime("%I:%M:%S %p IST")
            # Re-filter buys with updated prices
            st.session_state["buys"] = [t for t in all_t if t["confidence"] >= min_conf]
            buys = st.session_state["buys"]
            if _updated:
                st.success(f"✅ Updated prices for {_updated} stocks")

        # Stats row — always show even if buys is empty after filtering
        s1,s2,s3,s4,s5,s6 = st.columns(6)
        s1.metric("Scanned", len(all_t))
        s2.metric(f"≥{min_conf:.0%} Conf", len(buys))
        s3.metric("Below Threshold", len(all_t) - len(buys))
        penny_ct = len([t for t in all_t if t["is_penny"]])
        s4.metric("Penny Stocks", penny_ct)
        if buys:
            s5.metric("Best Confidence", f"{buys[0]['confidence']:.0%}", buys[0]["symbol"])
            s6.metric("Best R:R", f"{buys[0]['risk_reward']}x", buys[0]["sector"])
        st.markdown("---")

        if not buys:
            # Show all trades sorted by confidence so user can see what's available
            st.warning(f"No signals at ≥{min_conf:.0%} confidence. Lower the **Min Confidence** slider to see results.")
            if all_t:
                best_available = sorted(all_t, key=lambda x: x["confidence"], reverse=True)[:5]
                st.markdown(f"**Best available signals (top 5 of {len(all_t)} scanned):**")
                _ba_cols = st.columns(min(5, len(best_available)))
                for _col, t in zip(_ba_cols, best_available):
                    _c = t["confidence"]
                    _cc = "#10b981" if _c >= 0.55 else "#f59e0b"
                    with _col:
                        st.markdown(f"""
                        <div class="stat-card" style="text-align:center;">
                            <div style="font-size:0.65rem;color:#334155;">{t["sector"]}</div>
                            <div style="font-size:1rem;font-weight:900;color:#f1f5f9;">{t["symbol"]}</div>
                            <div style="font-size:1.1rem;font-weight:800;color:{_cc};">{_c:.0%}</div>
                            <div style="font-size:0.72rem;color:#475569;">₹{t["price"]:,.2f}</div>
                        </div>""", unsafe_allow_html=True)
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
                        <span style='margin-left:8px;font-size:0.68rem;color:#475569;'>{best.get("data_quality","")}</span>
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
                "Data": t.get("data_quality", ""),
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

    # ── Auto-refresh: non-blocking timestamp-based re-scan ───────────────────
    if auto_refresh:
        _last_scan_ts = st.session_state.get("_last_scan_ts", 0)
        _now_ts = _time.time()
        _elapsed = _now_ts - _last_scan_ts
        _remaining = max(0, refresh_sec - int(_elapsed))

        # Show countdown so user knows when next refresh fires
        st.markdown(
            f'<div style="color:#334155;font-size:0.72rem;margin-top:8px;">'
            f'🔄 Auto-refresh in <b style="color:#3b82f6;">{_remaining}s</b> '
            f'(every {refresh_sec//60}m) — re-runs full scan with fresh prices</div>',
            unsafe_allow_html=True,
        )

        if _elapsed >= refresh_sec:
            # Time to re-scan — clear caches and re-run
            clear_live_cache()
            DataService.clear_intraday_cache()
            trades, mkt_label, mkt_bias = run_scan(universe, capital, max_price, min_conf, penny_only)
            st.session_state["trades"] = trades
            st.session_state["scan_time"] = _now_ist().strftime("%I:%M:%S %p IST")
            st.session_state["scan_date"] = scan_date.strftime("%d %b %Y")
            st.session_state["mkt_label"] = mkt_label
            st.session_state["mkt_bias"] = mkt_bias
            st.session_state["_last_scan_ts"] = _time.time()
            st.rerun()
        else:
            # Not time yet — schedule a lightweight rerun after remaining seconds
            # Use st.empty + fragment-style polling (no blocking sleep)
            _time.sleep(min(5, _remaining))   # max 5s sleep so UI stays responsive
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
        penny_trades, _, _ = run_scan(penny_universe, capital, penny_max_price_filter, penny_min_conf, penny_only=True)
        st.session_state["penny_trades"] = penny_trades
        st.session_state["penny_min_conf"] = penny_min_conf

    # Re-filter penny trades live when slider changes
    if "penny_trades" in st.session_state:
        _cur_penny_conf = penny_min_conf
        pt = [t for t in st.session_state["penny_trades"] if t["confidence"] >= _cur_penny_conf]
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
        nd_min_conf = st.slider("Min Confidence", 0.30, 0.90, 0.45, 0.05, key="nd_conf")
    with nd_c4:
        nd_btn = st.button("🔮 Run Predictions", type="primary", use_container_width=True)

    if nd_btn or st.session_state.pop("_trigger_predict", False):
        if nd_stocks:
            nd_universe = [f"{s}.NS" for s in nd_stocks]
        elif nd_sectors:
            nd_universe = []
            for s in nd_sectors:
                nd_universe.extend(SECTOR_GROUPS.get(s, []))
            nd_universe = list(dict.fromkeys(nd_universe))
        else:
            # Use full universe — no arbitrary 120-stock cap
            nd_universe = INTRADAY_STOCKS[:]

        prog2 = st.progress(0, text="🔮 Running ML predictions...")
        nd_results = []
        nd_processed = 0
        nd_skipped = 0
        total2 = len(nd_universe)

        done2 = 0
        all_preds = []   # keep ALL results for debug, filter after
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = {ex.submit(_predict_one, sym): sym for sym in nd_universe}
            for fut in as_completed(futs):
                done2 += 1
                prog2.progress(done2 / total2, text=f"🔮 {done2}/{total2} — scanning…")
                res = fut.result()
                if res:
                    all_preds.append(res)

        prog2.empty()

        # Show ALL UP predictions above threshold
        nd_results = [r for r in all_preds if r["direction"] == "UP" and r["confidence"] >= nd_min_conf]
        nd_results.sort(key=lambda x: (x["confidence"], x["return_pct"]), reverse=True)

        # Debug info stored so user can see what happened
        st.session_state["nd_results"] = nd_results
        st.session_state["nd_all_preds"] = all_preds

    if "nd_all_preds" in st.session_state:
        all_preds_debug = st.session_state["nd_all_preds"]

        # Re-filter live every time slider changes — no stale results
        nd_results = [
            r for r in all_preds_debug
            if r["direction"] == "UP" and r["confidence"] >= nd_min_conf
        ]
        nd_results.sort(key=lambda x: (x["confidence"], x["return_pct"]), reverse=True)
        st.session_state["nd_results"] = nd_results

        up_ct   = sum(1 for r in all_preds_debug if r["direction"] == "UP")
        down_ct = sum(1 for r in all_preds_debug if r["direction"] == "DOWN")
        neut_ct = sum(1 for r in all_preds_debug if r["direction"] == "NEUTRAL")
        total_processed = len(all_preds_debug)

        # Always show scan summary
        sa, sb, sc, sd = st.columns(4)
        sa.metric("Scanned", total_processed)
        sb.metric("UP", up_ct, delta="bullish")
        sc.metric("DOWN", down_ct)
        sd.metric("NEUTRAL", neut_ct)
        st.markdown("---")

        if not nd_results:
            st.warning(
                f"No UP predictions above **{nd_min_conf:.0%}** confidence. "
                f"Found **{up_ct}** UP signals total — lower the slider to see them."
            )
            # Always show best UP picks regardless of threshold
            best_ups = sorted(
                [r for r in all_preds_debug if r["direction"] == "UP"],
                key=lambda x: x["confidence"], reverse=True
            )[:10]
            if best_ups:
                st.markdown("#### Best UP signals (below your threshold):")
                _cols = st.columns(min(5, len(best_ups)))
                for _col, r in zip(_cols, best_ups[:5]):
                    ret_color = "#10b981" if r["return_pct"] > 0 else "#ef4444"
                    with _col:
                        st.markdown(f"""
                        <div class="stat-card" style="text-align:center;">
                            <div style="font-size:0.65rem;color:#334155;">{r["sector"]}</div>
                            <div style="font-size:1rem;font-weight:900;color:#f1f5f9;">{r["symbol"]}</div>
                            <div style="font-size:1.2rem;color:{ret_color};font-weight:800;">{r["return_pct"]:+.2f}%</div>
                            <div style="font-size:0.75rem;color:#3b82f6;">Conf: {r["confidence"]:.0%}</div>
                            <div style="font-size:0.68rem;color:#475569;">₹{r["price"]:,.2f}</div>
                        </div>""", unsafe_allow_html=True)
            elif total_processed == 0:
                st.error("No stocks were processed. This usually means yfinance rate-limiting or a network issue. Try again in 30 seconds.")
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
        # Clear stale result if symbol changed
        if st.session_state.get("_mp_last_sym") != mp_symbol:
            st.session_state.pop("mp_result", None)
        st.session_state["_mp_last_sym"] = mp_symbol
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
        # Clear stale result if symbol changed
        if st.session_state.get("_ex_last_sym") != ex_sym:
            st.session_state.pop("ex_result", None)
        st.session_state["_ex_last_sym"] = ex_sym
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
        # Clear stale result if symbol changed
        if st.session_state.get("_rp_last_sym") != rp_sym:
            st.session_state.pop("rp_result", None)
        st.session_state["_rp_last_sym"] = rp_sym
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
                    <div style='color:#94a3b8;font-size:0.82rem;'>Volatility: <b style='color:#f59e0b;'>{r["vol"]*100:.1f}%</b></div>
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

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 8 — MARKET CONTEXT (20 INDICATORS)
# ═══════════════════════════════════════════════════════════════════════════════
with tab8:
    st.markdown("### 📊 Market Context — 20 Indicators")
    st.markdown('<div style="color:#334155;font-size:0.82rem;margin-bottom:16px;">Real-time composite of 10 Indian + 10 Global indicators. Drives position sizing and signal filtering across all tabs.</div>', unsafe_allow_html=True)

    mc_col1, mc_col2 = st.columns([1, 5])
    with mc_col1:
        mc_refresh_btn = st.button("🔄 Refresh", type="primary", use_container_width=True, key="mc_refresh")
    with mc_col2:
        st.markdown('<div style="color:#334155;font-size:0.75rem;padding-top:10px;">Auto-cached 15 min · Used by Live Scan to scale position sizes and filter signals</div>', unsafe_allow_html=True)

    with st.spinner("Loading 20 market indicators…"):
        ctx = get_market_context(force_refresh=mc_refresh_btn)

    bias = ctx["market_bias_score"]
    label = ctx["market_bias_label"]
    psm = ctx["position_size_mult"]
    bias_color = "#10b981" if bias >= 0.58 else ("#ef4444" if bias <= 0.42 else "#f59e0b")

    # ── Top summary strip ─────────────────────────────────────────────────────
    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    mc1.markdown(f"""
    <div class="stat-card" style="text-align:center;">
        <div style="font-size:0.65rem;color:#475569;text-transform:uppercase;font-weight:700;">Market Bias</div>
        <div style="font-size:1.4rem;font-weight:900;color:{bias_color};">{label}</div>
        <div style="font-size:0.75rem;color:#94a3b8;">{bias:.0%} score</div>
    </div>""", unsafe_allow_html=True)
    mc2.metric("Position Size", f"{psm:.0%}", "of capital")
    mc3.metric("India Score", f"{ctx['indian_score']:.0%}")
    mc4.metric("Global Score", f"{ctx['global_score']:.0%}")
    mc5.metric("Intraday Safe", "✅ Yes" if ctx["intraday_filter"] else "⚠️ Caution")
    mc6.metric("Swing Safe", "✅ Yes" if ctx["swing_filter"] else "⚠️ Caution")

    # Bias bar
    bar_pct = int(bias * 100)
    bar_color = "#10b981" if bias >= 0.58 else ("#ef4444" if bias <= 0.42 else "#f59e0b")
    st.markdown(f"""
    <div style="margin:12px 0 4px;font-size:0.72rem;color:#475569;font-weight:700;">COMPOSITE MARKET BIAS</div>
    <div style="background:#0f2040;border-radius:8px;height:14px;overflow:hidden;position:relative;">
        <div style="background:{bar_color};width:{bar_pct}%;height:100%;border-radius:8px;transition:width 0.5s;"></div>
        <div style="position:absolute;top:0;left:50%;width:2px;height:100%;background:#334155;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.65rem;color:#334155;margin-top:2px;">
        <span>STRONG BEAR</span><span>NEUTRAL</span><span>STRONG BULL</span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="color:#334155;font-size:0.68rem;margin-top:4px;">Last updated: {ctx["fetched_at"]}</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ── Strategy rules based on current context ───────────────────────────────
    st.markdown("### 🧠 Strategy Rules (Auto-Generated)")
    rules = []
    if bias >= 0.65:
        rules += ["✅ Strong bull market — full position sizing allowed",
                  "✅ Aggressive longs on breakouts with volume",
                  "✅ Sector leaders: follow FII inflows"]
    elif bias >= 0.55:
        rules += ["✅ Moderate bull — 85% position sizing",
                  "✅ Trade with trend, avoid counter-trend",
                  "⚠️ Watch for sector rotation signals"]
    elif bias >= 0.45:
        rules += ["⚠️ Neutral market — 65% position sizing",
                  "⚠️ Trade only high-conviction setups",
                  "⚠️ Tighten stop-losses, reduce targets"]
    elif bias >= 0.35:
        rules += ["🔴 Bearish bias — 40% position sizing",
                  "🔴 Avoid aggressive longs",
                  "🔴 Focus on defensive sectors (FMCG, Pharma)"]
    else:
        rules += ["🔴 Strong bear — 25% position sizing only",
                  "🔴 Cash is a position — wait for reversal",
                  "🔴 High VIX: avoid intraday scalping"]

    # Add VIX-specific rule
    vix_ind = next((i for i in ctx["indicators"] if i["key"] == "india_vix"), None)
    if vix_ind and vix_ind["score"] < 0.40:
        rules.append("⚠️ India VIX elevated — reduce intraday exposure, widen stops")

    for rule in rules:
        st.markdown(f'<div style="padding:6px 12px;margin:3px 0;background:#0a1628;border-left:3px solid {bias_color};border-radius:0 6px 6px 0;font-size:0.82rem;color:#e2e8f0;">{rule}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Indian indicators ─────────────────────────────────────────────────────
    st.markdown("### 🇮🇳 Indian Market Indicators")
    indian_inds = [i for i in ctx["indicators"] if "Indian" in i["category"]]
    _render_indicator_grid(indian_inds)

    st.markdown("---")

    # ── Global indicators ─────────────────────────────────────────────────────
    st.markdown("### 🌍 Global Market Indicators")
    global_inds = [i for i in ctx["indicators"] if "Global" in i["category"]]
    _render_indicator_grid(global_inds)

    st.markdown("---")

    # ── Full indicator table ──────────────────────────────────────────────────
    st.markdown("### 📋 Full Indicator Table")
    ind_tbl = pd.DataFrame([{
        "Indicator": i["name"],
        "Category": i["category"],
        "Score": f"{i['score']:.0%}",
        "Signal": i["signal"],
        "Trend": i["trend"],
        "Impact": i["impact"],
        "Weight": f"{i['weight']:.0%}",
        "Note": i["note"],
    } for i in ctx["indicators"]])
    st.dataframe(ind_tbl, use_container_width=True, hide_index=True)

    # ── Pro trader rules reference ────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 Pro Trader Rules Reference"):
        st.markdown("""
**INTRADAY STRATEGY:**
- Nifty trend + VWAP + Volume spike → confirm with Bank Nifty + VIX
- Trade only when 3+ indicators align

**SWING STRATEGY:**
- Sector rotation + Relative strength + FII inflow support
- Global filter: check S&P 500, Crude Oil, DXY first

**POSITION SIZING RULES:**
- Strong Bull (>65%): Full capital
- Bull (55-65%): 85% capital
- Neutral (45-55%): 65% capital
- Bear (35-45%): 40% capital
- Strong Bear (<35%): 25% capital — cash is a position

**NEVER:**
- Trade against India VIX > 25
- Ignore FII selling pressure
- Use full position in high VIX environment
        """)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style='display:flex;justify-content:space-between;align-items:center;color:#1e3a5f;font-size:0.68rem;padding:4px 0;flex-wrap:wrap;gap:8px;'>
    <span>⚡ QuantSignal India v5.0 &nbsp;|&nbsp; {len(ALL_SYMBOLS_CLEAN)} stocks &nbsp;|&nbsp; {len(SECTOR_UNIVERSE)} sectors &nbsp;|&nbsp; 8 tabs</span>    <span>Data: Yahoo Finance (fallback) · Plug in Kite/Upstox for real-time &nbsp;|&nbsp; Not financial advice</span>
    <span>{_now_ist().strftime("%d %b %Y %I:%M %p")} IST</span>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 7 — DEEP DIVE STOCK ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab7:
    from backend.engines.stock_analysis_engine import load_analysis_bundle, load_peer_data, compute_trade_setup

    st.markdown("""
    <div style='margin-bottom:16px;'>
        <span style='font-size:1.4rem;font-weight:900;color:#f1f5f9;'>🔍 Deep Dive — Stock Analysis</span>
        <div style='color:#334155;font-size:0.8rem;margin-top:4px;'>Full technical + ML + risk + Monte Carlo analysis for any NSE stock</div>
    </div>""", unsafe_allow_html=True)

    dd_col1, dd_col2, dd_col3 = st.columns([2.5, 1, 1])
    with dd_col1:
        dd_symbol = st.selectbox("Select Stock", ALL_SYMBOLS_CLEAN, key="dd_sym",
                                  index=ALL_SYMBOLS_CLEAN.index("RELIANCE") if "RELIANCE" in ALL_SYMBOLS_CLEAN else 0)
    with dd_col2:
        dd_capital = st.number_input("Capital (₹)", value=capital, min_value=1000, step=1000, key="dd_cap")
    with dd_col3:
        dd_risk = st.slider("Risk %", 0.5, 5.0, 2.0, 0.5, key="dd_risk") / 100

    analyse_btn = st.button("🔍 ANALYSE", type="primary", use_container_width=False, key="dd_btn")

    if analyse_btn:
        cache_key = f"sap_{dd_symbol}"
        with st.spinner(f"Loading full analysis for {dd_symbol}..."):
            try:
                bundle = load_analysis_bundle(dd_symbol, dd_capital, dd_risk)
                st.session_state[cache_key] = bundle
            except Exception as e:
                st.error(f"Could not load data for {dd_symbol}: {e}")
                bundle = None
    else:
        bundle = st.session_state.get(f"sap_{dd_symbol}")

    if bundle:
        price = bundle.current_price
        chg = bundle.price_change_1d
        chg_color = "#10b981" if chg >= 0 else "#ef4444"
        chg_arrow = "▲" if chg >= 0 else "▼"
        cap_class = bundle.metadata_report.get("price_class", "MID")
        risk_lvl = bundle.metadata_report.get("risk_level", "🟡 MEDIUM")
        sector = bundle.sector

        # ── SECTION 1: PRICE OVERVIEW ─────────────────────────────────────────
        st.markdown("---")
        ov1, ov2, ov3, ov4, ov5 = st.columns(5)
        ov1.metric("Current Price", f"₹{price:,.2f}", f"{chg_arrow} {chg:+.2f}%")
        ov2.metric("52W High", f"₹{bundle.price_52w_high:,.2f}")
        ov3.metric("52W Low", f"₹{bundle.price_52w_low:,.2f}")
        ov4.metric("Cap Class", cap_class)
        ov5.metric("Risk Level", risk_lvl.split()[-1] if risk_lvl else "—")

        st.markdown(f"""
        <div style='margin:8px 0 16px;'>
            <span class="badge-sector">{sector}</span>
            <span style='background:rgba(16,185,129,0.1);color:#6ee7b7;padding:3px 10px;border-radius:6px;font-size:0.72rem;border:1px solid rgba(16,185,129,0.2);margin-right:4px;'>{cap_class}</span>
            <span style='color:{chg_color};font-weight:700;font-size:0.85rem;'>{chg_arrow} {chg:+.2f}% today</span>
        </div>""", unsafe_allow_html=True)

        # ── SECTION 2: TECHNICAL CHART ────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Technical Chart")
        tf_choice = st.radio("Timeframe", ["Daily", "Intraday 5m"], horizontal=True, key="dd_tf")

        df_chart = bundle.df_intra if tf_choice == "Intraday 5m" and not bundle.df_intra.empty else bundle.df_daily
        df_feat = bundle.df_features if tf_choice == "Daily" else bundle.df_intra

        if not df_chart.empty:
            dc = df_chart.tail(252 if tf_choice == "Daily" else 120)
            fig = make_subplots(rows=4, cols=1, row_heights=[0.55, 0.15, 0.15, 0.15],
                                shared_xaxes=True, vertical_spacing=0.02)
            fig.add_trace(go.Candlestick(
                x=dc.index, open=dc["Open"], high=dc["High"], low=dc["Low"], close=dc["Close"],
                name="Price", increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
                decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
            ), row=1, col=1)
            overlays = [("EMA9","#6366f1","solid"),("EMA21","#ec4899","solid"),
                        ("VWAP","#f59e0b","dot"),("BB_Upper","#475569","dash"),("BB_Lower","#475569","dash")]
            for col_name, color, dash in overlays:
                if col_name in dc.columns:
                    fig.add_trace(go.Scatter(x=dc.index, y=dc[col_name], mode="lines",
                        line=dict(color=color, width=1, dash=dash), name=col_name, showlegend=True), row=1, col=1)
            ts = bundle.trade_setup
            fig.add_hline(y=ts.entry, line_color="#6366f1", line_width=1.5, annotation_text="ENTRY", row=1, col=1)
            fig.add_hline(y=ts.target_1, line_color="#10b981", line_dash="dash", annotation_text="T1", row=1, col=1)
            fig.add_hline(y=ts.stop_loss, line_color="#ef4444", line_dash="dash", annotation_text="SL", row=1, col=1)
            if "RSI" in dc.columns:
                fig.add_trace(go.Scatter(x=dc.index, y=dc["RSI"], mode="lines",
                    line=dict(color="#a78bfa", width=1.5), name="RSI"), row=2, col=1)
                fig.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=0.8, row=2, col=1)
                fig.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=0.8, row=2, col=1)
            macd_col = "MACD_Hist" if "MACD_Hist" in dc.columns else None
            if macd_col:
                hist = dc[macd_col]
                fig.add_trace(go.Bar(x=dc.index, y=hist,
                    marker_color=["#10b981" if v >= 0 else "#ef4444" for v in hist],
                    name="MACD Hist"), row=3, col=1)
            vcols = ["#10b981" if c >= o else "#ef4444" for c, o in zip(dc["Close"], dc["Open"])]
            fig.add_trace(go.Bar(x=dc.index, y=dc["Volume"], marker_color=vcols, name="Volume", opacity=0.8), row=4, col=1)
            layout = _chart_layout(height=620)
            layout["xaxis_rangeslider_visible"] = False
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

        # ── SECTION 3: MULTI-TIMEFRAME SIGNALS ───────────────────────────────
        st.markdown("---")
        st.markdown("### ⏱ Multi-Timeframe Signals")
        tf_cols = st.columns(3)
        timeframes = [
            ("5m Intraday", bundle.intra_score.get("score", 0.5), bundle.intra_score.get("rsi", 50), bundle.intra_score.get("supertrend", "—"), bundle.intra_score.get("reasons", [])),
        ]
        # 1d signal from multi-analyzer
        ma_score_1d = bundle.ma_result.get("combined_score", 0.5)
        ma_signal_1d = bundle.ma_result.get("signal", "NEUTRAL")
        ma_reasons_1d = bundle.ma_result.get("reasoning", [])
        feat_last = bundle.df_features.iloc[-1] if not bundle.df_features.empty else {}
        rsi_1d = float(feat_last.get("RSI", 50)) if hasattr(feat_last, "get") else 50

        for i, (tf_label, score, rsi_val, st_val, reasons) in enumerate(timeframes):
            sig = "BUY" if score >= 0.55 else ("SELL" if score < 0.35 else "NEUTRAL")
            sig_color = "#10b981" if sig == "BUY" else ("#ef4444" if sig == "SELL" else "#f59e0b")
            with tf_cols[i]:
                st.markdown(f"""
                <div class="stat-card" style="text-align:left;">
                    <div style='color:#475569;font-size:0.65rem;text-transform:uppercase;font-weight:700;'>{tf_label}</div>
                    <div style='color:{sig_color};font-size:1.3rem;font-weight:900;margin:6px 0;'>{sig}</div>
                    <div style='color:#94a3b8;font-size:0.75rem;'>Score: <b style='color:#f1f5f9;'>{score:.0%}</b></div>
                    <div style='color:#94a3b8;font-size:0.75rem;'>RSI: <b style='color:#a78bfa;'>{rsi_val:.0f}</b></div>
                    <div style='color:#94a3b8;font-size:0.75rem;'>Supertrend: <b style='color:{"#10b981" if st_val=="BUY" else "#ef4444"};'>{st_val}</b></div>
                    <div style='margin-top:6px;'>{''.join(f"<span class='reason-chip'>{r}</span>" for r in reasons[:3])}</div>
                </div>""", unsafe_allow_html=True)

        with tf_cols[1]:
            sig_1d = "BUY" if ma_score_1d >= 0.55 else ("SELL" if ma_score_1d < 0.35 else "NEUTRAL")
            sig_color_1d = "#10b981" if sig_1d == "BUY" else ("#ef4444" if sig_1d == "SELL" else "#f59e0b")
            st.markdown(f"""
            <div class="stat-card" style="text-align:left;">
                <div style='color:#475569;font-size:0.65rem;text-transform:uppercase;font-weight:700;'>1d Swing</div>
                <div style='color:{sig_color_1d};font-size:1.3rem;font-weight:900;margin:6px 0;'>{sig_1d}</div>
                <div style='color:#94a3b8;font-size:0.75rem;'>Score: <b style='color:#f1f5f9;'>{ma_score_1d:.0%}</b></div>
                <div style='color:#94a3b8;font-size:0.75rem;'>RSI: <b style='color:#a78bfa;'>{rsi_1d:.0f}</b></div>
                <div style='margin-top:6px;'>{''.join(f"<span class='reason-chip'>{r}</span>" for r in ma_reasons_1d[:3])}</div>
            </div>""", unsafe_allow_html=True)

        with tf_cols[2]:
            pred_nd = bundle.pred_next_day
            nd_dir = pred_nd.get("direction", "NEUTRAL")
            nd_conf = pred_nd.get("confidence", 0.5)
            nd_color = "#10b981" if nd_dir == "UP" else ("#ef4444" if nd_dir == "DOWN" else "#f59e0b")
            st.markdown(f"""
            <div class="stat-card" style="text-align:left;">
                <div style='color:#475569;font-size:0.65rem;text-transform:uppercase;font-weight:700;'>ML Next-Day</div>
                <div style='color:{nd_color};font-size:1.3rem;font-weight:900;margin:6px 0;'>{nd_dir}</div>
                <div style='color:#94a3b8;font-size:0.75rem;'>Confidence: <b style='color:#f1f5f9;'>{nd_conf:.0%}</b></div>
                <div style='color:#94a3b8;font-size:0.75rem;'>Predicted: <b style='color:#10b981;'>₹{pred_nd.get("predicted_price", price):,.2f}</b></div>
                <div style='color:#94a3b8;font-size:0.75rem;'>Return: <b style='color:{nd_color};'>{pred_nd.get("predicted_return", 0):+.2f}%</b></div>
            </div>""", unsafe_allow_html=True)

        # ── SECTION 4: ML PREDICTIONS ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔮 ML Predictions")
        pred_multi = bundle.pred_multi
        pred_labels = [("Next Day", bundle.pred_next_day), ("10 Days", pred_multi.get("10_days", {})),
                       ("30 Days", pred_multi.get("30_days", {})), ("3 Months", pred_multi.get("3_months", {}))]
        pred_cols = st.columns(4)
        for i, (lbl, p) in enumerate(pred_labels):
            if not p or "error" in p:
                pred_cols[i].metric(lbl, "N/A")
                continue
            d = p.get("direction", "NEUTRAL")
            c = p.get("confidence", 0.5)
            pp = p.get("predicted_price", price)
            ret = p.get("predicted_return", 0)
            d_color = "#10b981" if d == "UP" else ("#ef4444" if d == "DOWN" else "#f59e0b")
            pred_cols[i].markdown(f"""
            <div class="stat-card">
                <div style='color:#475569;font-size:0.65rem;text-transform:uppercase;font-weight:700;'>{lbl}</div>
                <div style='color:{d_color};font-size:1.1rem;font-weight:900;margin:4px 0;'>{"▲" if d=="UP" else "▼" if d=="DOWN" else "→"} {d}</div>
                <div style='color:#f1f5f9;font-weight:700;'>₹{pp:,.2f}</div>
                <div style='color:{d_color};font-size:0.78rem;font-weight:700;'>{ret:+.2f}%</div>
                <div style='color:#475569;font-size:0.7rem;'>Conf: {c:.0%}</div>
                <div style='color:#334155;font-size:0.68rem;'>₹{p.get("price_low",price):,.0f} – ₹{p.get("price_high",price):,.0f}</div>
            </div>""", unsafe_allow_html=True)

        conds = bundle.pred_next_day.get("market_conditions", [])
        if conds:
            st.markdown('<div style="margin-top:8px;">' + " &nbsp;·&nbsp; ".join(f'<span class="reason-chip">{c}</span>' for c in conds) + '</div>', unsafe_allow_html=True)

        # ── SECTION 5: RISK METRICS ───────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🛡️ Risk Metrics")
        rm = bundle.risk_metrics
        if rm and "error" not in rm:
            r1, r2, r3, r4, r5, r6, r7 = st.columns(7)
            sharpe = rm.get("sharpe_ratio", 0)
            sharpe_color = "#10b981" if sharpe > 1 else ("#f59e0b" if sharpe > 0 else "#ef4444")
            r1.metric("Sharpe", f"{sharpe:.2f}")
            r2.metric("Sortino", f"{rm.get('sortino_ratio', 0):.2f}")
            r3.metric("VaR 95%", f"{rm.get('var_historical', {}).get('var', 0)*100:.2f}%")
            r4.metric("CVaR 95%", f"{rm.get('cvar', {}).get('cvar', 0)*100:.2f}%")
            r5.metric("Max DD", f"{rm.get('max_drawdown', {}).get('max_drawdown', 0)*100:.1f}%")
            r6.metric("Annual Vol", f"{rm.get('volatility_annual', 0)*100:.1f}%")
            r7.metric("Total Return", f"{rm.get('total_return', 0)*100:.1f}%")

            try:
                stress = RiskEngine.stress_test(bundle.df_daily["Close"].pct_change().dropna())
                if stress:
                    st.markdown("**Stress Test Scenarios**")
                    stress_df = pd.DataFrame(stress)
                    st.dataframe(stress_df, use_container_width=True, hide_index=True)
            except Exception:
                pass
        else:
            st.info("Risk metrics unavailable — insufficient data.")

        # ── SECTION 6: MONTE CARLO ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🎲 Monte Carlo Simulation (30-day)")
        mc = bundle.mc_result
        if mc and "error" not in mc:
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Prob Profit", f"{mc.get('prob_profit', 0):.0%}")
            mc2.metric("Expected Price", f"₹{mc.get('expected_price', price):,.2f}")
            mc3.metric("p5 (Worst)", f"₹{mc.get('p5', price):,.2f}")
            mc4.metric("p95 (Best)", f"₹{mc.get('p95', price):,.2f}")

            paths = mc.get("sample_paths", [])
            if paths:
                fig_mc = go.Figure()
                days_x = list(range(len(paths[0])))
                for path in paths[:40]:
                    fig_mc.add_trace(go.Scatter(x=days_x, y=path, mode="lines",
                        line=dict(color="rgba(99,102,241,0.12)", width=1), showlegend=False))
                # Percentile bands
                import numpy as _np
                paths_arr = _np.array(paths)
                for pct, color, name in [(5,"#ef4444","p5"),(25,"#f59e0b","p25"),(50,"#10b981","Median"),(75,"#f59e0b","p75"),(95,"#ef4444","p95")]:
                    band = _np.percentile(paths_arr, pct, axis=0)
                    fig_mc.add_trace(go.Scatter(x=days_x, y=band, mode="lines",
                        line=dict(color=color, width=2, dash="dash" if pct in [5,95] else "solid"),
                        name=name))
                fig_mc.add_hline(y=price, line_color="#6366f1", line_dash="dot", annotation_text="Current")
                fig_mc.update_layout(**_chart_layout("Monte Carlo — 30-Day Price Simulation", height=380))
                st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.info("Monte Carlo unavailable — insufficient data.")

        # ── SECTION 7: MULTI-ANALYZER SCORECARD ──────────────────────────────
        st.markdown("---")
        st.markdown("### 🔬 Multi-Analyzer Scorecard")
        ma = bundle.ma_result
        if ma and "analyzers" in ma:
            combined = ma.get("combined_score", 0.5)
            ma_signal = ma.get("signal", "NEUTRAL")
            sig_color_ma = "#10b981" if "BUY" in ma_signal else ("#ef4444" if "SELL" in ma_signal else "#f59e0b")
            st.markdown(f'<div style="margin-bottom:12px;">Combined Score: <b style="color:#f1f5f9;font-size:1.2rem;">{combined:.0%}</b> &nbsp; Signal: <span style="color:{sig_color_ma};font-weight:800;">{ma_signal}</span></div>', unsafe_allow_html=True)

            breakdown = ma["analyzers"]
            an_cols = st.columns(len(breakdown))
            for i, (name, data) in enumerate(breakdown.items()):
                sc = data["score"]
                sig = data["signal"]
                sig_icon = "🟢" if sig == "BULLISH" else ("🔴" if sig == "BEARISH" else "🟡")
                an_cols[i].markdown(f"""
                <div class="stat-card">
                    <div style='color:#475569;font-size:0.62rem;text-transform:uppercase;font-weight:700;'>{name}</div>
                    <div style='color:#f1f5f9;font-size:1.3rem;font-weight:900;'>{sc:.0%}</div>
                    <div style='font-size:0.8rem;'>{sig_icon} {sig}</div>
                    <div style='color:#334155;font-size:0.65rem;'>wt {data["weight"]:.0%}</div>
                </div>""", unsafe_allow_html=True)

            # Radar chart
            names_r = list(breakdown.keys())
            scores_r = [breakdown[n]["score"] for n in names_r]
            fig_radar = go.Figure(go.Scatterpolar(
                r=scores_r + [scores_r[0]], theta=names_r + [names_r[0]],
                fill="toself", fillcolor="rgba(99,102,241,0.15)",
                line=dict(color="#6366f1", width=2), name="Scores",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1], tickformat=".0%",
                    gridcolor="#1e3a5f", linecolor="#1e3a5f"),
                    angularaxis=dict(gridcolor="#1e3a5f"), bgcolor="#080f1e"),
                template="plotly_dark", paper_bgcolor="#04080f", height=380,
                margin=dict(t=20, b=20),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            reasons_ma = ma.get("reasoning", [])
            if reasons_ma:
                st.markdown("**Signal Reasoning**")
                for r in reasons_ma:
                    icon = "✅" if any(w in r.lower() for w in ["bullish","rising","positive","strong","above","low vix"]) else "⚠️"
                    st.markdown(f"{icon} {r}")

        # ── SECTION 8: GLOBAL FACTOR IMPACT ──────────────────────────────────
        st.markdown("---")
        st.markdown("### 🌍 Global Factor Impact")
        gf = bundle.metadata_report.get("global_factors", {})
        theme = gf.get("theme", "NEUTRAL")
        theme_color = "#10b981" if "STRONG" in theme else ("#f59e0b" if "MODERATE" in theme else "#94a3b8")
        st.markdown(f'<div style="margin-bottom:10px;">Theme: <span style="color:{theme_color};font-weight:800;">{theme}</span></div>', unsafe_allow_html=True)
        gf_col1, gf_col2 = st.columns(2)
        with gf_col1:
            st.markdown("**Positive Factors**")
            for f in gf.get("positive", []):
                st.markdown(f'<span class="factor-pos">✅ {f}</span>', unsafe_allow_html=True)
        with gf_col2:
            st.markdown("**Risk Factors**")
            for f in gf.get("negative", []):
                st.markdown(f'<span class="factor-neg">⚠️ {f}</span>', unsafe_allow_html=True)

        # ── SECTION 9: TRADE SETUP GENERATOR ─────────────────────────────────
        st.markdown("---")
        st.markdown("### 💼 Trade Setup Generator")
        ts = bundle.trade_setup
        sig_color_ts = "#10b981" if ts.signal == "BUY" else ("#ef4444" if ts.signal == "AVOID" else "#f59e0b")

        ts_c1, ts_c2, ts_c3, ts_c4 = st.columns(4)
        ts_c1.metric("Entry", f"₹{ts.entry:,.2f}")
        ts_c2.metric("Target 1", f"₹{ts.target_1:,.2f}", f"+₹{ts.max_profit:,.0f}")
        ts_c3.metric("Stop Loss", f"₹{ts.stop_loss:,.2f}", f"-₹{ts.max_loss:,.0f}", delta_color="inverse")
        ts_c4.metric("R:R Ratio", f"{ts.risk_reward}x")

        ts_c5, ts_c6, ts_c7, ts_c8 = st.columns(4)
        ts_c5.metric("Qty", ts.qty)
        ts_c6.metric("Invested", f"₹{ts.invested:,.0f}")
        ts_c7.metric("ATR", f"₹{ts.atr:,.2f}")
        ts_c8.metric("Signal", ts.signal)

        st.markdown(f"""
        <div class="trade-card" style="margin-top:12px;">
            <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>
                <span style='font-size:1.1rem;font-weight:900;color:#f1f5f9;'>{dd_symbol}</span>
                <span class="badge-sector">{sector}</span>
                <span style='color:{sig_color_ts};font-weight:800;font-size:1rem;'>{ts.signal}</span>
                <span style='color:#475569;font-size:0.78rem;'>Confidence: <b style='color:#f1f5f9;'>{ts.confidence:.0%}</b></span>
            </div>
            <div style='margin-top:10px;display:grid;grid-template-columns:repeat(4,1fr);gap:10px;'>
                <div><div style='color:#475569;font-size:0.65rem;text-transform:uppercase;'>Entry</div><div style='color:#f1f5f9;font-weight:700;'>₹{ts.entry:,.2f}</div></div>
                <div><div style='color:#475569;font-size:0.65rem;text-transform:uppercase;'>Target 1</div><div style='color:#10b981;font-weight:700;'>₹{ts.target_1:,.2f}</div></div>
                <div><div style='color:#475569;font-size:0.65rem;text-transform:uppercase;'>Target 2</div><div style='color:#06b6d4;font-weight:700;'>₹{ts.target_2:,.2f}</div></div>
                <div><div style='color:#475569;font-size:0.65rem;text-transform:uppercase;'>Stop Loss</div><div style='color:#ef4444;font-weight:700;'>₹{ts.stop_loss:,.2f}</div></div>
            </div>
            <div style='margin-top:8px;color:#334155;font-size:0.75rem;'>
                Qty: <b style='color:#3b82f6;'>{ts.qty}</b> &nbsp;·&nbsp;
                Invested: <b style='color:#3b82f6;'>₹{ts.invested:,.0f}</b> &nbsp;·&nbsp;
                Max Profit: <b style='color:#10b981;'>+₹{ts.max_profit:,.0f}</b> &nbsp;·&nbsp;
                Max Loss: <b style='color:#ef4444;'>-₹{ts.max_loss:,.0f}</b>
            </div>
        </div>""", unsafe_allow_html=True)

        # ── SECTION 10: PEER COMPARISON ───────────────────────────────────────
        st.markdown("---")
        with st.expander("📊 Peer Comparison (click to expand)", expanded=False):
            with st.spinner("Loading peer data..."):
                try:
                    peers = load_peer_data(dd_symbol, sector)
                    if peers:
                        # Normalised price chart
                        fig_peer = go.Figure()
                        colors_p = ["#6366f1","#10b981","#f59e0b","#06b6d4","#ec4899"]
                        for i, p in enumerate(peers):
                            df_p = p.get("df", pd.DataFrame())
                            if not df_p.empty:
                                norm = (df_p["Close"] / df_p["Close"].iloc[0]) * 100
                                lw = 2.5 if p["is_target"] else 1.5
                                fig_peer.add_trace(go.Scatter(
                                    x=df_p.index, y=norm, mode="lines",
                                    line=dict(color=colors_p[i % len(colors_p)], width=lw),
                                    name=f"★ {p['symbol']}" if p["is_target"] else p["symbol"],
                                ))
                        fig_peer.add_hline(y=100, line_color="#475569", line_dash="dot")
                        fig_peer.update_layout(**_chart_layout("Normalised 6-Month Performance (Base=100)", height=350))
                        st.plotly_chart(fig_peer, use_container_width=True)

                        # Metrics table
                        peer_tbl = pd.DataFrame([{
                            "Stock": ("★ " if p["is_target"] else "") + p["symbol"],
                            "Price": f"₹{p['price']:,.2f}",
                            "1M %": f"{p['return_1m']:+.1f}%",
                            "3M %": f"{p['return_3m']:+.1f}%",
                            "6M %": f"{p['return_6m']:+.1f}%",
                            "RSI": p["rsi"],
                            "Vol %": f"{p['volatility']:.1f}%",
                            "Sharpe": p["sharpe"],
                        } for p in peers])
                        st.dataframe(peer_tbl, use_container_width=True, hide_index=True)
                    else:
                        st.info("No peer data available for this sector.")
                except Exception as e:
                    st.warning(f"Peer comparison unavailable: {e}")

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;'>
            <div style='font-size:3rem;margin-bottom:12px;'>🔍</div>
            <div style='font-size:1.4rem;font-weight:900;color:#f1f5f9;margin-bottom:8px;'>Stock Deep Dive</div>
            <div style='color:#334155;font-size:0.9rem;'>Select a stock above and click <b style='color:#3b82f6;'>ANALYSE</b> to see the full breakdown</div>
            <div style='color:#1e3a5f;font-size:0.78rem;margin-top:12px;'>
                Technical Chart · ML Predictions · Risk Metrics · Monte Carlo · Multi-Analyzer · Trade Setup · Peer Comparison
            </div>
        </div>""", unsafe_allow_html=True)

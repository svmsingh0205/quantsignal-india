"""QuantSignal India — Live Trading + Prediction Dashboard"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time as dtime
import sys, os, time as _time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.engines.intraday_engine import IntradayEngine, get_market_status
from backend.engines.data_service import DataService
from backend.engines.prediction_engine import PredictionEngine
from backend.intraday_config import INTRADAY_STOCKS, SECTOR_GROUPS

st.set_page_config(page_title="QuantSignal India", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif !important; }
.stApp { background: #060b18; }
.block-container { padding: 1rem 2rem; max-width: 1600px; }
section[data-testid="stSidebar"] { background: #0d1526; border-right: 1px solid #1e2d4a; }
section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
div[data-testid="stMetric"] { background: #0f1e35; border: 1px solid #1e3a5f; border-radius: 12px; padding: 14px 18px; }
div[data-testid="stMetricLabel"] { color: #64748b !important; font-size: 0.7rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
div[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.4rem !important; font-weight: 700 !important; }
.stButton > button { background: linear-gradient(135deg,#1d4ed8,#2563eb) !important; color: white !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; box-shadow: 0 4px 15px rgba(37,99,235,0.35) !important; }
.stButton > button:hover { transform: translateY(-1px) !important; }
.stProgress > div > div { background: linear-gradient(90deg,#1d4ed8,#06b6d4) !important; }
h1,h2,h3 { color: #f1f5f9 !important; }
hr { border-color: #1e2d4a !important; }
.stTabs [data-baseweb="tab-list"] { background: #0d1526; border-radius: 10px; padding: 4px; gap: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #64748b; border-radius: 8px; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #1d4ed8 !important; color: white !important; }
.badge-buy { display:inline-block; background:linear-gradient(135deg,#059669,#10b981); color:white; padding:5px 16px; border-radius:8px; font-weight:700; }
.badge-up { display:inline-block; background:linear-gradient(135deg,#059669,#10b981); color:white; padding:4px 12px; border-radius:6px; font-weight:700; font-size:0.85rem; }
.badge-down { display:inline-block; background:linear-gradient(135deg,#dc2626,#ef4444); color:white; padding:4px 12px; border-radius:6px; font-weight:700; font-size:0.85rem; }
.badge-neutral { display:inline-block; background:linear-gradient(135deg,#d97706,#f59e0b); color:white; padding:4px 12px; border-radius:6px; font-weight:700; font-size:0.85rem; }
.badge-sector { display:inline-block; background:rgba(99,102,241,0.15); color:#a5b4fc; padding:3px 10px; border-radius:6px; font-size:0.72rem; border:1px solid rgba(99,102,241,0.3); margin-right:4px; }
.reason-chip { display:inline-block; background:rgba(6,182,212,0.1); color:#67e8f9; padding:3px 10px; border-radius:6px; margin:2px; font-size:0.72rem; border:1px solid rgba(6,182,212,0.25); }
.pred-card { background:#0f1e35; border:1px solid #1e3a5f; border-radius:14px; padding:20px; margin-bottom:12px; }
.live-dot { display:inline-block; width:8px; height:8px; background:#10b981; border-radius:50%; animation:pulse 1.5s infinite; margin-right:6px; vertical-align:middle; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.5;transform:scale(1.3)} }
</style>
""", unsafe_allow_html=True)

# ── helpers ───────────────────────────────────────────────────────────────────
SYMBOL_TO_SECTOR = {}
for _sec, _syms in SECTOR_GROUPS.items():
    for _s in _syms:
        SYMBOL_TO_SECTOR[_s.replace(".NS", "")] = _sec

def market_open():
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    return dtime(9, 15) <= now.time() <= dtime(15, 30)

def time_to_close():
    now = datetime.now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if now >= close:
        return "Market Closed"
    diff = close - now
    h, m = divmod(diff.seconds // 60, 60)
    return f"{h}h {m}m to close" if h else f"{m}m to close"

def get_universe():
    sf = st.session_state.get("sector_filter", [])
    if not sf:
        return INTRADAY_STOCKS
    out = []
    for s in sf:
        out.extend(SECTOR_GROUPS.get(s, []))
    return list(dict.fromkeys(out))

# ── parallel intraday scan ────────────────────────────────────────────────────
def _scan_one(args):
    sym, capital, max_price, min_conf = args
    try:
        engine = IntradayEngine(capital=capital)
        df = engine.fetch_intraday(sym, period="5d", interval="5m")
        if df.empty or len(df) < 30:
            return None
        price = float(df["Close"].iloc[-1])
        if price > max_price:
            return None
        df = engine.add_indicators(df)
        sc = engine.score_stock(df)
        atr = sc["atr"] or price * 0.01
        entry = round(price, 2)
        sl = round(price - 1.5 * atr, 2)
        t1 = round(price + 2.0 * atr, 2)
        t2 = round(price + 3.0 * atr, 2)
        risk = entry - sl
        rr = round((t1 - entry) / risk, 2) if risk > 0 else 0
        qty = max(1, int(capital // price))
        clean = sym.replace(".NS", "")
        return {
            "symbol": clean, "yf_symbol": sym,
            "sector": SYMBOL_TO_SECTOR.get(clean, "Other"),
            "price": entry, "qty": qty,
            "invested": round(qty * price, 2),
            "target_1": t1, "target_2": t2, "stop_loss": sl,
            "confidence": sc["score"], "risk_reward": rr,
            "profit": round(qty * (t1 - entry), 2),
            "loss": round(qty * (entry - sl), 2),
            "rsi": sc["rsi"], "vwap": sc["vwap"],
            "vol_ratio": sc["vol_ratio"], "supertrend": sc["supertrend"],
            "reasons": sc["reasons"],
            "signal": "BUY" if sc["score"] >= min_conf else "WATCH",
            "df": df,
        }
    except Exception:
        return None

def run_scan(universe, capital, max_price, min_conf):
    prog = st.progress(0, text="🚀 Parallel scanning...")
    results = []
    total = len(universe)
    done = 0
    with ThreadPoolExecutor(max_workers=15) as ex:
        futures = {ex.submit(_scan_one, (sym, capital, max_price, min_conf)): sym for sym in universe}
        for fut in as_completed(futures):
            done += 1
            prog.progress(done / total, text=f"📡 {done}/{total} — {futures[fut].replace('.NS','')}")
            res = fut.result()
            if res:
                results.append(res)
    prog.empty()
    results.sort(key=lambda x: (x["confidence"], x["risk_reward"]), reverse=True)
    return results

# ── sidebar ───────────────────────────────────────────────────────────────────
is_open = market_open()
mkt = get_market_status()

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:8px 0 16px 0;'>
        <div style='font-size:1.8rem;'>⚡</div>
        <div style='font-size:1.2rem;font-weight:800;color:#f1f5f9;'>QuantSignal</div>
        <div style='font-size:0.68rem;color:#475569;letter-spacing:0.1em;'>INDIA TRADING ENGINE</div>
    </div>""", unsafe_allow_html=True)

    dot = '<span class="live-dot"></span>' if is_open else ""
    cls = "color:#10b981" if is_open else "color:#ef4444"
    st.markdown(f'<div style="{cls};font-weight:600;font-size:0.85rem;">{dot}{mkt}</div>', unsafe_allow_html=True)
    if is_open:
        st.markdown(f'<div style="color:#64748b;font-size:0.75rem;">⏱ {time_to_close()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div style="color:#334155;font-size:0.68rem;">{datetime.now().strftime("%d %b %Y, %I:%M %p")}</div>', unsafe_allow_html=True)
    st.markdown("---")

    capital = st.number_input("Capital (₹)", value=5000, min_value=500, max_value=10_000_000, step=500)
    min_conf = st.slider("Min Confidence", 0.30, 0.90, 0.50, 0.05)
    max_price = st.number_input("Max Price (₹)", value=int(capital * 0.95), min_value=10, max_value=5_000_000)
    st.markdown("---")
    st.multiselect("Sector Filter (empty=all)", list(SECTOR_GROUPS.keys()), default=[], key="sector_filter")
    st.markdown("---")
    auto_refresh = st.checkbox("🔄 Auto-refresh", value=False)
    refresh_sec = 300
    if auto_refresh:
        refresh_sec = st.select_slider("Interval", [60, 120, 300, 600], value=300,
                                        format_func=lambda x: f"{x//60}m")
    st.markdown("---")
    st.markdown(f'<div style="color:#475569;font-size:0.68rem;">Universe: <b style="color:#3b82f6;">{len(INTRADAY_STOCKS)}</b> stocks | 14 sectors</div>', unsafe_allow_html=True)
    st.caption("Not financial advice. Paper trade first.")

# ── header ────────────────────────────────────────────────────────────────────
universe = get_universe()
col_t, col_b1, col_b2 = st.columns([4, 1, 1])
with col_t:
    live_badge = '<span style="background:#059669;color:white;padding:2px 8px;border-radius:5px;font-size:0.68rem;font-weight:700;">LIVE</span>' if is_open else '<span style="background:#dc2626;color:white;padding:2px 8px;border-radius:5px;font-size:0.68rem;font-weight:700;">CLOSED</span>'
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:2px;'>
        <span style='font-size:1.7rem;font-weight:800;color:#f8fafc;'>⚡ QuantSignal India</span>{live_badge}
    </div>
    <div style='color:#475569;font-size:0.82rem;'>
        <b style='color:#3b82f6;'>{len(universe)}</b> stocks •
        <b style='color:#3b82f6;'>14 sectors</b> •
        {datetime.now().strftime("%d %b %Y %I:%M %p")}
    </div>""", unsafe_allow_html=True)
with col_b1:
    scan_btn = st.button("🔍 SCAN NOW", type="primary", use_container_width=True)
with col_b2:
    best_btn = st.button("⚡ BEST TRADE", use_container_width=True)

st.markdown("---")

# ── trigger scan ──────────────────────────────────────────────────────────────
if scan_btn or best_btn:
    trades = run_scan(universe, capital, max_price, min_conf)
    st.session_state["trades"] = trades
    st.session_state["buys"] = [t for t in trades if t["signal"] == "BUY"]
    st.session_state["scan_time"] = datetime.now().strftime("%I:%M:%S %p")
    st.session_state["scan_date"] = datetime.now().strftime("%d %b %Y")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📡 Live Intraday Signals",
    "🔮 Next-Day Predictions",
    "📈 Multi-Period Forecast",
    "📊 Stock Deep Dive",
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE INTRADAY
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    if "buys" in st.session_state:
        buys = st.session_state["buys"]
        all_t = st.session_state["trades"]
        scan_time = st.session_state.get("scan_time", "")

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("Scanned", len(all_t))
        s2.metric("BUY Signals", len(buys), scan_time)
        s3.metric("WATCH", len(all_t) - len(buys))
        if buys:
            s4.metric("Best Confidence", f"{buys[0]['confidence']:.0%}", buys[0]["symbol"])
            s5.metric("Best R:R", f"{buys[0]['risk_reward']}x", buys[0]["sector"])
        st.markdown("---")

        if not buys:
            st.info("No strong BUY signals. Lower Min Confidence or remove sector filter.")
        else:
            best = buys[0]
            st.markdown(f"### 🏆 #1 Best Trade")
            cl, cm, cr = st.columns([2.5, 2, 1.2])
            with cl:
                st.markdown(f"""
                <div style='background:#0f1e35;border:1px solid #1e3a5f;border-radius:14px;padding:22px;'>
                    <span class="badge-sector">{best["sector"]}</span>
                    <div style='font-size:1.5rem;font-weight:800;color:#f1f5f9;margin-top:8px;'>{best["symbol"]}</div>
                    <div style='font-size:2.5rem;font-weight:800;color:#10b981;'>₹{best["price"]:,.2f}</div>
                    <div style='margin-top:12px;'><span class="badge-buy">BUY {best["confidence"]:.0%}</span></div>
                    <div style='margin-top:10px;'>{''.join(f'<span class="reason-chip">{r}</span>' for r in best["reasons"])}</div>
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

            if "df" in best and not best["df"].empty:
                dc = best["df"].tail(100)
                fig = make_subplots(rows=3, cols=1, row_heights=[0.60, 0.20, 0.20],
                                    shared_xaxes=True, vertical_spacing=0.02)
                fig.add_trace(go.Candlestick(
                    x=dc.index, open=dc["Open"], high=dc["High"],
                    low=dc["Low"], close=dc["Close"], name="Price",
                    increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
                    decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
                ), row=1, col=1)
                for cn, col, dash, lbl in [("VWAP","#f59e0b","dot","VWAP"),("EMA9","#6366f1","solid","EMA9"),("EMA21","#ec4899","solid","EMA21")]:
                    if cn in dc.columns:
                        fig.add_trace(go.Scatter(x=dc.index, y=dc[cn], mode="lines",
                            line=dict(color=col, width=1.2, dash=dash), name=lbl), row=1, col=1)
                fig.add_hline(y=best["price"], line_color="#6366f1", line_width=1.5, annotation_text="ENTRY", row=1, col=1)
                fig.add_hline(y=best["target_1"], line_color="#10b981", line_dash="dash", annotation_text="T1", row=1, col=1)
                fig.add_hline(y=best["stop_loss"], line_color="#ef4444", line_dash="dash", annotation_text="SL", row=1, col=1)
                if "RSI" in dc.columns:
                    fig.add_trace(go.Scatter(x=dc.index, y=dc["RSI"], mode="lines",
                        line=dict(color="#a78bfa", width=1.5), name="RSI",
                        fill="tozeroy", fillcolor="rgba(167,139,250,0.05)"), row=2, col=1)
                    fig.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=0.8, row=2, col=1)
                    fig.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=0.8, row=2, col=1)
                vcols = ["#10b981" if c >= o else "#ef4444" for c, o in zip(dc["Close"], dc["Open"])]
                fig.add_trace(go.Bar(x=dc.index, y=dc["Volume"], marker_color=vcols, name="Vol", opacity=0.8), row=3, col=1)
                fig.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
                    height=500, margin=dict(t=10,b=10,l=50,r=20), showlegend=True,
                    legend=dict(orientation="h", y=1.02, font=dict(size=10)),
                    xaxis_rangeslider_visible=False)
                fig.update_yaxes(gridcolor="#0f1e35"); fig.update_xaxes(gridcolor="#0f1e35")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")
            # sector chart
            col_sec, col_conf = st.columns(2)
            with col_sec:
                st.markdown("### 📊 By Sector")
                sc_cnt = {}
                for t in buys:
                    sc_cnt[t["sector"]] = sc_cnt.get(t["sector"], 0) + 1
                sc_df = pd.DataFrame(sorted(sc_cnt.items(), key=lambda x: x[1], reverse=True), columns=["Sector","Count"])
                fig_s = go.Figure(go.Bar(x=sc_df["Count"], y=sc_df["Sector"], orientation="h",
                    marker=dict(color=sc_df["Count"], colorscale=[[0,"#1e3a5f"],[1,"#06b6d4"]]),
                    text=sc_df["Count"], textposition="outside"))
                fig_s.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
                    height=280, margin=dict(t=10,b=10,l=10,r=40), yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_s, use_container_width=True)
            with col_conf:
                st.markdown("### 🎯 Top Confidence")
                top15 = buys[:15]
                fig_c = go.Figure(go.Bar(x=[t["symbol"] for t in top15], y=[t["confidence"] for t in top15],
                    marker=dict(color=[t["confidence"] for t in top15],
                                colorscale=[[0,"#dc2626"],[0.5,"#f59e0b"],[1,"#10b981"]], cmin=0, cmax=1),
                    text=[f"{t['confidence']:.0%}" for t in top15], textposition="outside"))
                fig_c.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
                    height=280, margin=dict(t=10,b=10,l=10,r=10), yaxis=dict(range=[0,1.15], tickformat=".0%"))
                st.plotly_chart(fig_c, use_container_width=True)

            st.markdown("---")
            st.markdown(f"### 📋 All BUY Signals ({len(buys)})")
            tbl = pd.DataFrame([{
                "#": i+1, "Stock": t["symbol"], "Sector": t["sector"],
                "Price": f"₹{t['price']:,.2f}", "Target": f"₹{t['target_1']:,.2f}",
                "SL": f"₹{t['stop_loss']:,.2f}", "Conf": f"{t['confidence']:.0%}",
                "R:R": f"{t['risk_reward']}x", "Qty": t["qty"],
                "Invest": f"₹{t['invested']:,.0f}", "Profit": f"+₹{t['profit']:,.2f}",
                "Loss": f"-₹{t['loss']:,.2f}", "RSI": t["rsi"],
                "Vol": f"{t['vol_ratio']:.1f}x", "ST": t["supertrend"],
            } for i, t in enumerate(buys)])
            st.dataframe(tbl, use_container_width=True, hide_index=True, height=min(600, 55+38*len(tbl)))

            st.markdown("---")
            st.markdown("### 🔎 Stock Details (Top 15)")
            for t in buys[:15]:
                icon = "🟢" if t["confidence"] >= 0.65 else "🟡"
                with st.expander(f"{icon} {t['symbol']} [{t['sector']}] ₹{t['price']:,.2f} • {t['confidence']:.0%} • R:R {t['risk_reward']}x"):
                    d1,d2,d3,d4,d5 = st.columns(5)
                    d1.metric("Entry", f"₹{t['price']:,.2f}")
                    d2.metric("Target 1", f"₹{t['target_1']:,.2f}", f"+₹{t['profit']:,.2f}")
                    d3.metric("Stop Loss", f"₹{t['stop_loss']:,.2f}", f"-₹{t['loss']:,.2f}", delta_color="inverse")
                    d4.metric("R:R", f"{t['risk_reward']}x")
                    d5.metric("Qty", t["qty"])
                    st.markdown("".join(f'<span class="reason-chip">{r}</span>' for r in t["reasons"]), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px;background:#0f1e35;border:1px solid #1e3a5f;border-radius:16px;'>
            <div style='font-size:3rem;'>⚡</div>
            <div style='font-size:1.2rem;color:#94a3b8;margin-top:12px;'>Click SCAN NOW to find live BUY signals</div>
            <div style='color:#475569;margin-top:8px;'>Scans {len(INTRADAY_STOCKS)} NSE stocks in parallel</div>
        </div>""", unsafe_allow_html=True)

    if auto_refresh and "buys" in st.session_state:
        _time.sleep(refresh_sec)
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — NEXT-DAY PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 🔮 Next-Day Stock Predictions")
    st.markdown('<div style="color:#64748b;font-size:0.85rem;margin-bottom:16px;">ML ensemble predicts tomorrow\'s direction for every stock. Best for swing entry planning.</div>', unsafe_allow_html=True)

    col_nd1, col_nd2, col_nd3 = st.columns([2, 1, 1])
    with col_nd1:
        nd_sectors = st.multiselect("Filter sectors", list(SECTOR_GROUPS.keys()), default=[], key="nd_sectors")
    with col_nd2:
        nd_min_conf = st.slider("Min Confidence", 0.50, 0.90, 0.60, 0.05, key="nd_conf")
    with col_nd3:
        nd_btn = st.button("🔮 Run Next-Day Scan", type="primary", use_container_width=True)

    if nd_btn:
        nd_universe = []
        if nd_sectors:
            for s in nd_sectors:
                nd_universe.extend(SECTOR_GROUPS.get(s, []))
            nd_universe = list(dict.fromkeys(nd_universe))
        else:
            nd_universe = INTRADAY_STOCKS[:150]  # top 150 for speed

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
                return {
                    "symbol": clean,
                    "sector": SYMBOL_TO_SECTOR.get(clean, "Other"),
                    "price": pred["current_price"],
                    "predicted": pred["predicted_price"],
                    "return_pct": pred["predicted_return"],
                    "direction": pred["direction"],
                    "confidence": pred["confidence"],
                    "rsi": pred["rsi"],
                    "volatility": pred["volatility"],
                    "conditions": pred["market_conditions"],
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
            st.info("No strong next-day UP predictions found. Try lowering confidence.")
        else:
            st.markdown(f"**{len(nd_results)} stocks predicted UP for tomorrow**")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("UP Predictions", len(nd_results))
            m2.metric("Best Return", f"{nd_results[0]['return_pct']:+.2f}%", nd_results[0]["symbol"])
            m3.metric("Avg Confidence", f"{np.mean([r['confidence'] for r in nd_results]):.0%}")
            m4.metric("Avg Expected Return", f"{np.mean([r['return_pct'] for r in nd_results]):+.2f}%")

            st.markdown("---")
            # Top picks cards
            st.markdown("### 🏆 Top 5 Next-Day Picks")
            cols_nd = st.columns(min(5, len(nd_results)))
            for i, (col, r) in enumerate(zip(cols_nd, nd_results[:5])):
                with col:
                    ret_color = "#10b981" if r["return_pct"] > 0 else "#ef4444"
                    st.markdown(f"""
                    <div class="pred-card" style="text-align:center;">
                        <div style="font-size:0.68rem;color:#64748b;">{r["sector"]}</div>
                        <div style="font-size:1.1rem;font-weight:800;color:#f1f5f9;margin:4px 0;">{r["symbol"]}</div>
                        <div style="font-size:1.4rem;font-weight:700;color:#94a3b8;">₹{r["price"]:,.2f}</div>
                        <div style="font-size:1.1rem;font-weight:700;color:{ret_color};">{r["return_pct"]:+.2f}%</div>
                        <div style="font-size:0.75rem;color:#64748b;">→ ₹{r["predicted"]:,.2f}</div>
                        <div style="margin-top:8px;"><span class="badge-up">UP {r["confidence"]:.0%}</span></div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("---")
            # Full table
            nd_tbl = pd.DataFrame([{
                "#": i+1, "Stock": r["symbol"], "Sector": r["sector"],
                "Current ₹": f"₹{r['price']:,.2f}",
                "Predicted ₹": f"₹{r['predicted']:,.2f}",
                "Expected Return": f"{r['return_pct']:+.2f}%",
                "Direction": r["direction"],
                "Confidence": f"{r['confidence']:.0%}",
                "RSI": r["rsi"],
                "Volatility": f"{r['volatility']:.1f}%",
            } for i, r in enumerate(nd_results)])
            st.dataframe(nd_tbl, use_container_width=True, hide_index=True, height=min(500, 55+38*len(nd_tbl)))

            # Scatter: confidence vs expected return
            fig_nd = go.Figure(go.Scatter(
                x=[r["return_pct"] for r in nd_results],
                y=[r["confidence"] for r in nd_results],
                mode="markers+text",
                text=[r["symbol"] for r in nd_results],
                textposition="top center",
                marker=dict(
                    size=[max(8, r["confidence"]*20) for r in nd_results],
                    color=[r["return_pct"] for r in nd_results],
                    colorscale=[[0,"#dc2626"],[0.5,"#f59e0b"],[1,"#10b981"]],
                    showscale=True, colorbar=dict(title="Return %"),
                ),
            ))
            fig_nd.add_vline(x=0, line_color="#475569", line_dash="dash")
            fig_nd.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
                title="Next-Day: Expected Return vs Confidence",
                xaxis_title="Expected Return (%)", yaxis_title="Confidence",
                height=400, margin=dict(t=40,b=40))
            st.plotly_chart(fig_nd, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MULTI-PERIOD FORECAST (10/20/30 days, 3/6 months)
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 📈 Multi-Period Price Forecast")
    st.markdown('<div style="color:#64748b;font-size:0.85rem;margin-bottom:16px;">ML-powered forecasts for 10, 20, 30 days and 3, 6 months with price range bands.</div>', unsafe_allow_html=True)

    all_syms = sorted(set(s.replace(".NS","") for s in INTRADAY_STOCKS))
    col_mp1, col_mp2 = st.columns([2, 1])
    with col_mp1:
        mp_symbol = st.selectbox("Select Stock", all_syms,
                                  index=all_syms.index("RELIANCE") if "RELIANCE" in all_syms else 0,
                                  key="mp_sym")
    with col_mp2:
        mp_btn = st.button("📈 Generate Forecast", type="primary", use_container_width=True)

    if mp_btn:
        with st.spinner(f"Running ML forecast for {mp_symbol}..."):
            df_mp = DataService.fetch_ohlcv(f"{mp_symbol}.NS", period="2y")
            if df_mp.empty:
                st.error(f"No data for {mp_symbol}")
            else:
                nd_pred = PredictionEngine.predict_next_day(df_mp)
                multi_pred = PredictionEngine.predict_multi_horizon(df_mp)
                st.session_state["mp_result"] = {
                    "symbol": mp_symbol,
                    "next_day": nd_pred,
                    "multi": multi_pred,
                    "df": df_mp,
                }

    if "mp_result" in st.session_state:
        r = st.session_state["mp_result"]
        sym = r["symbol"]
        nd = r["next_day"]
        multi = r["multi"]
        df_hist = r["df"]

        if "error" not in nd:
            # Next day highlight
            dir_badge = f'<span class="badge-{"up" if nd["direction"]=="UP" else "down" if nd["direction"]=="DOWN" else "neutral"}">{nd["direction"]}</span>'
            st.markdown(f"""
            <div style='background:#0f1e35;border:1px solid #1e3a5f;border-radius:14px;padding:20px;margin-bottom:16px;'>
                <div style='display:flex;align-items:center;gap:12px;flex-wrap:wrap;'>
                    <span style='font-size:1.3rem;font-weight:800;color:#f1f5f9;'>{sym}</span>
                    {dir_badge}
                    <span style='color:#94a3b8;'>Current: <b style='color:#f1f5f9;'>₹{nd["current_price"]:,.2f}</b></span>
                    <span style='color:#94a3b8;'>Tomorrow: <b style='color:{"#10b981" if nd["predicted_return"]>0 else "#ef4444"};'>₹{nd["predicted_price"]:,.2f} ({nd["predicted_return"]:+.2f}%)</b></span>
                    <span style='color:#94a3b8;'>Confidence: <b style='color:#3b82f6;'>{nd["confidence"]:.0%}</b></span>
                </div>
                <div style='margin-top:10px;color:#64748b;font-size:0.8rem;'>
                    {'&nbsp;•&nbsp;'.join(nd.get("market_conditions", []))}
                </div>
            </div>""", unsafe_allow_html=True)

        # Multi-horizon cards
        horizon_labels = {
            "10_days": ("10 Days", "📅"),
            "20_days": ("20 Days", "📅"),
            "30_days": ("30 Days", "📅"),
            "3_months": ("3 Months", "📆"),
            "6_months": ("6 Months", "📆"),
        }
        cols_h = st.columns(5)
        for col, (key, (label, icon)) in zip(cols_h, horizon_labels.items()):
            pred = multi.get(key, {})
            if pred and "error" not in pred:
                ret = pred["predicted_return"]
                ret_color = "#10b981" if ret > 0 else "#ef4444"
                dir_icon = "↑" if pred["direction"] == "UP" else ("↓" if pred["direction"] == "DOWN" else "→")
                with col:
                    st.markdown(f"""
                    <div class="pred-card" style="text-align:center;">
                        <div style="font-size:0.72rem;color:#64748b;">{icon} {label}</div>
                        <div style="font-size:1.3rem;font-weight:800;color:{ret_color};margin:6px 0;">{dir_icon} {ret:+.1f}%</div>
                        <div style="font-size:0.85rem;color:#94a3b8;">₹{pred["predicted_price"]:,.0f}</div>
                        <div style="font-size:0.72rem;color:#475569;">₹{pred["price_low"]:,.0f} – ₹{pred["price_high"]:,.0f}</div>
                        <div style="font-size:0.72rem;color:#3b82f6;margin-top:4px;">Conf: {pred["confidence"]:.0%}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("---")

        # Forecast chart
        current_price = float(df_hist["Close"].iloc[-1])
        horizons_plot = []
        for key, (label, _) in horizon_labels.items():
            pred = multi.get(key, {})
            if pred and "error" not in pred:
                horizons_plot.append({
                    "label": label,
                    "days": pred["horizon_days"],
                    "price": pred["predicted_price"],
                    "low": pred["price_low"],
                    "high": pred["price_high"],
                    "direction": pred["direction"],
                })

        if horizons_plot:
            fig_mp = go.Figure()
            # Historical price (last 60 days)
            hist_60 = df_hist["Close"].tail(60)
            x_hist = list(range(-len(hist_60), 0))
            fig_mp.add_trace(go.Scatter(x=x_hist, y=hist_60.values, mode="lines",
                line=dict(color="#6366f1", width=2), name="Historical"))
            # Forecast points
            x_fore = [h["days"] for h in horizons_plot]
            y_fore = [h["price"] for h in horizons_plot]
            y_low = [h["low"] for h in horizons_plot]
            y_high = [h["high"] for h in horizons_plot]
            colors = ["#10b981" if h["direction"]=="UP" else "#ef4444" for h in horizons_plot]
            # Confidence band
            fig_mp.add_trace(go.Scatter(x=x_fore+x_fore[::-1], y=y_high+y_low[::-1],
                fill="toself", fillcolor="rgba(99,102,241,0.1)", line=dict(color="rgba(0,0,0,0)"),
                name="Price Range"))
            fig_mp.add_trace(go.Scatter(x=x_fore, y=y_fore, mode="markers+lines+text",
                text=[h["label"] for h in horizons_plot],
                textposition="top center",
                marker=dict(size=12, color=colors, line=dict(color="white", width=2)),
                line=dict(color="#94a3b8", width=1.5, dash="dash"),
                name="Forecast"))
            fig_mp.add_hline(y=current_price, line_color="#f59e0b", line_dash="dot",
                              annotation_text=f"Current ₹{current_price:,.2f}")
            fig_mp.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
                title=f"{sym} — Price Forecast", xaxis_title="Days from Today",
                yaxis_title="Price (₹)", height=420, margin=dict(t=40,b=40))
            fig_mp.update_yaxes(gridcolor="#0f1e35"); fig_mp.update_xaxes(gridcolor="#0f1e35")
            st.plotly_chart(fig_mp, use_container_width=True)

        # Historical chart with volume
        st.markdown(f"### 📊 {sym} — Historical Price (1 Year)")
        fig_hist = make_subplots(rows=2, cols=1, row_heights=[0.75, 0.25], shared_xaxes=True, vertical_spacing=0.03)
        fig_hist.add_trace(go.Candlestick(
            x=df_hist.index, open=df_hist["Open"], high=df_hist["High"],
            low=df_hist["Low"], close=df_hist["Close"], name="Price",
            increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
            decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
        ), row=1, col=1)
        # MA lines
        for w, color in [(20,"#f59e0b"),(50,"#6366f1"),(200,"#ec4899")]:
            if len(df_hist) >= w:
                ma = df_hist["Close"].rolling(w).mean()
                fig_hist.add_trace(go.Scatter(x=df_hist.index, y=ma, mode="lines",
                    line=dict(color=color, width=1), name=f"MA{w}"), row=1, col=1)
        vcols2 = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df_hist["Close"], df_hist["Open"])]
        fig_hist.add_trace(go.Bar(x=df_hist.index, y=df_hist["Volume"], marker_color=vcols2, name="Vol", opacity=0.7), row=2, col=1)
        fig_hist.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
            height=480, margin=dict(t=10,b=10,l=50,r=20), showlegend=True,
            legend=dict(orientation="h", y=1.02, font=dict(size=10)),
            xaxis_rangeslider_visible=False)
        fig_hist.update_yaxes(gridcolor="#0f1e35"); fig_hist.update_xaxes(gridcolor="#0f1e35")
        st.plotly_chart(fig_hist, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — STOCK DEEP DIVE
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 📊 Stock Deep Dive")
    col_dd1, col_dd2, col_dd3 = st.columns([2, 1, 1])
    with col_dd1:
        dd_sym = st.selectbox("Select Stock", all_syms,
                               index=all_syms.index("RELIANCE") if "RELIANCE" in all_syms else 0,
                               key="dd_sym")
    with col_dd2:
        dd_period = st.selectbox("Period", ["1mo","3mo","6mo","1y","2y","5y"], index=3, key="dd_period")
    with col_dd3:
        dd_btn = st.button("🔍 Analyse", type="primary", use_container_width=True)

    if dd_btn:
        with st.spinner(f"Loading {dd_sym}..."):
            df_dd = DataService.fetch_ohlcv(f"{dd_sym}.NS", period=dd_period)
            st.session_state["dd_result"] = {"symbol": dd_sym, "df": df_dd, "period": dd_period}

    if "dd_result" in st.session_state:
        r = st.session_state["dd_result"]
        df_dd = r["df"]
        sym_dd = r["symbol"]

        if df_dd.empty:
            st.error(f"No data for {sym_dd}")
        else:
            # Key stats
            price = float(df_dd["Close"].iloc[-1])
            price_start = float(df_dd["Close"].iloc[0])
            total_ret = (price / price_start - 1) * 100
            high_52 = float(df_dd["High"].max())
            low_52 = float(df_dd["Low"].min())
            avg_vol = float(df_dd["Volume"].mean())
            returns = df_dd["Close"].pct_change().dropna()
            vol_ann = float(returns.std() * np.sqrt(252) * 100)

            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Current Price", f"₹{price:,.2f}")
            m2.metric("Period Return", f"{total_ret:+.1f}%")
            m3.metric("52W High", f"₹{high_52:,.2f}")
            m4.metric("52W Low", f"₹{low_52:,.2f}")
            m5.metric("Avg Volume", f"{avg_vol/1e6:.1f}M")
            m6.metric("Ann. Volatility", f"{vol_ann:.1f}%")

            st.markdown("---")

            # Full chart with indicators
            fig_dd = make_subplots(rows=4, cols=1, row_heights=[0.50,0.17,0.17,0.16],
                                    shared_xaxes=True, vertical_spacing=0.02,
                                    subplot_titles=["Price + MAs", "RSI", "MACD", "Volume"])
            fig_dd.add_trace(go.Candlestick(
                x=df_dd.index, open=df_dd["Open"], high=df_dd["High"],
                low=df_dd["Low"], close=df_dd["Close"], name="Price",
                increasing=dict(line=dict(color="#10b981"), fillcolor="#10b981"),
                decreasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
            ), row=1, col=1)
            for w, color in [(20,"#f59e0b"),(50,"#6366f1"),(200,"#ec4899")]:
                if len(df_dd) >= w:
                    ma = df_dd["Close"].rolling(w).mean()
                    fig_dd.add_trace(go.Scatter(x=df_dd.index, y=ma, mode="lines",
                        line=dict(color=color, width=1), name=f"MA{w}"), row=1, col=1)
            # Bollinger
            ma20 = df_dd["Close"].rolling(20).mean()
            std20 = df_dd["Close"].rolling(20).std()
            fig_dd.add_trace(go.Scatter(x=df_dd.index, y=ma20+2*std20, mode="lines",
                line=dict(color="#334155", width=0.8, dash="dash"), name="BB Upper", showlegend=False), row=1, col=1)
            fig_dd.add_trace(go.Scatter(x=df_dd.index, y=ma20-2*std20, mode="lines",
                line=dict(color="#334155", width=0.8, dash="dash"), name="BB Lower",
                fill="tonexty", fillcolor="rgba(51,65,85,0.1)", showlegend=False), row=1, col=1)
            # RSI
            delta = df_dd["Close"].diff()
            gain = delta.where(delta>0,0).rolling(14).mean()
            loss = (-delta.where(delta<0,0)).rolling(14).mean()
            rsi = 100 - (100/(1+gain/loss.replace(0,np.nan)))
            fig_dd.add_trace(go.Scatter(x=df_dd.index, y=rsi, mode="lines",
                line=dict(color="#a78bfa", width=1.5), name="RSI"), row=2, col=1)
            fig_dd.add_hline(y=70, line_color="#ef4444", line_dash="dot", line_width=0.8, row=2, col=1)
            fig_dd.add_hline(y=30, line_color="#10b981", line_dash="dot", line_width=0.8, row=2, col=1)
            # MACD
            ema12 = df_dd["Close"].ewm(span=12).mean()
            ema26 = df_dd["Close"].ewm(span=26).mean()
            macd = ema12 - ema26
            macd_sig = macd.ewm(span=9).mean()
            macd_hist_vals = macd - macd_sig
            fig_dd.add_trace(go.Scatter(x=df_dd.index, y=macd, mode="lines",
                line=dict(color="#06b6d4", width=1.2), name="MACD"), row=3, col=1)
            fig_dd.add_trace(go.Scatter(x=df_dd.index, y=macd_sig, mode="lines",
                line=dict(color="#f59e0b", width=1.2), name="Signal"), row=3, col=1)
            hist_colors = ["#10b981" if v >= 0 else "#ef4444" for v in macd_hist_vals]
            fig_dd.add_trace(go.Bar(x=df_dd.index, y=macd_hist_vals, marker_color=hist_colors,
                name="Histogram", opacity=0.7), row=3, col=1)
            # Volume
            vcols3 = ["#10b981" if c >= o else "#ef4444" for c, o in zip(df_dd["Close"], df_dd["Open"])]
            fig_dd.add_trace(go.Bar(x=df_dd.index, y=df_dd["Volume"], marker_color=vcols3,
                name="Volume", opacity=0.8), row=4, col=1)
            fig_dd.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
                height=700, margin=dict(t=30,b=10,l=50,r=20), showlegend=True,
                legend=dict(orientation="h", y=1.01, font=dict(size=10)),
                xaxis_rangeslider_visible=False)
            fig_dd.update_yaxes(gridcolor="#0f1e35"); fig_dd.update_xaxes(gridcolor="#0f1e35")
            fig_dd.update_yaxes(title_text="Price ₹", row=1, col=1)
            fig_dd.update_yaxes(title_text="RSI", row=2, col=1, range=[0,100])
            fig_dd.update_yaxes(title_text="MACD", row=3, col=1)
            fig_dd.update_yaxes(title_text="Vol", row=4, col=1)
            st.plotly_chart(fig_dd, use_container_width=True)

            # Returns distribution
            st.markdown("### 📊 Returns Distribution")
            fig_ret = go.Figure(go.Histogram(x=returns*100, nbinsx=60,
                marker_color="rgba(99,102,241,0.6)", marker_line_color="#6366f1", marker_line_width=0.5))
            fig_ret.add_vline(x=0, line_color="#f59e0b", line_dash="dash")
            fig_ret.update_layout(template="plotly_dark", paper_bgcolor="#060b18", plot_bgcolor="#0a1020",
                xaxis_title="Daily Return (%)", yaxis_title="Frequency",
                height=300, margin=dict(t=10,b=40))
            st.plotly_chart(fig_ret, use_container_width=True)

# ── footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div style="display:flex;justify-content:space-between;color:#334155;font-size:0.68rem;padding:4px 0;"><span>⚡ QuantSignal India v4.0 | {len(INTRADAY_STOCKS)} stocks | 14 sectors</span><span>Data: Yahoo Finance (15-min delayed) | Not financial advice</span><span>{datetime.now().strftime("%d %b %Y %I:%M %p")}</span></div>', unsafe_allow_html=True)

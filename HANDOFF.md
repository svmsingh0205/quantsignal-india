# QuantSignal India — Handoff Document
**Version:** 3.0  
**Last updated:** April 2026  
**Purpose:** Resume this project from any new Kiro account / session without losing progress.

---

## What This Project Is

A live intraday trading signal engine for NSE (India) stocks.  
Scans ~200 stocks across 13 sectors, applies technical + ML analysis,  
outputs ranked BUY signals with entry / target / stop-loss.

**Two apps:**
1. `live_trader.py` — Live intraday scanner (main app)
2. `frontend/app.py` — Analytics dashboard (Monte Carlo, Portfolio, Options, Risk)

---

## Current Project State (what's done)

- [x] `backend/config.py` — ~200 stock universe across 13 sectors + geopolitical themes
- [x] `backend/intraday_config.py` — Intraday universe with sector groups
- [x] `backend/engines/data_service.py` — yfinance data fetcher with cache
- [x] `backend/engines/feature_engine.py` — RSI, MA, MACD, Bollinger, momentum
- [x] `backend/engines/ml_engine.py` — Ensemble ML (LogReg + RF + GBM)
- [x] `backend/engines/signal_engine.py` — Swing signal generator
- [x] `backend/engines/intraday_engine.py` — Live intraday scanner (VWAP/EMA/Supertrend)
- [x] `backend/engines/monte_carlo.py` — GBM simulation, VaR
- [x] `backend/engines/risk_engine.py` — VaR, CVaR, Sharpe, drawdown
- [x] `backend/engines/portfolio_engine.py` — Efficient frontier, max Sharpe
- [x] `backend/engines/options_engine.py` — Black-Scholes + Greeks
- [x] `live_trader.py` — Streamlit intraday dashboard (sector filter, charts)
- [x] `requirements.txt` — All dependencies
- [x] `frontend/app.py` — Analytics dashboard (Monte Carlo, Portfolio, Options, Risk)
- [x] `backend/main.py` — FastAPI with 8 endpoints

---

## What To Do Next (continue from here)

### Immediate next steps:
1. Add news/sentiment feed integration (geopolitical news scoring)
2. Add pre-market gap scanner
3. Add options chain live scanner
4. Deploy to Streamlit Cloud
5. Add backtesting engine

### Pending improvements:
- Add `PAYTM.NS`, `JIOFIN.NS` to watchlist for fintech theme
- Add geopolitical news sentiment scoring
- Add pre-market gap scanner
- Add options chain live scanner

---

## How To Run Locally

```bash
# Install dependencies
pip install streamlit plotly yfinance pandas numpy scikit-learn scipy fastapi uvicorn

# Run intraday scanner (main app)
streamlit run live_trader.py --server.port 3000

# Run analytics dashboard
streamlit run frontend/app.py --server.port 3001

# Run FastAPI backend
uvicorn backend.main:app --port 8000
```

---

## Project Structure

```
quantsignal/
├── live_trader.py              ← MAIN APP (intraday scanner)
├── requirements.txt
├── HANDOFF.md                  ← THIS FILE
├── backend/
│   ├── __init__.py
│   ├── config.py               ← ~200 stock universe, 13 sectors
│   ├── intraday_config.py      ← intraday params + sector groups
│   ├── requirements.txt
│   └── engines/
│       ├── __init__.py
│       ├── data_service.py
│       ├── feature_engine.py
│       ├── ml_engine.py
│       ├── signal_engine.py
│       ├── intraday_engine.py
│       ├── monte_carlo.py
│       ├── risk_engine.py
│       ├── portfolio_engine.py
│       └── options_engine.py
└── frontend/
    ├── __init__.py
    └── app.py                  ← analytics dashboard (TODO)
```

---

## Stock Universe Summary (~200 stocks, 13 sectors)

| Sector | Count | Key Stocks | Geopolitical Theme |
|--------|-------|------------|-------------------|
| 🛡️ Defence | 12 | HAL, BEL, BDL, GRSE, MTAR | India-US 10yr defence deal |
| 🏦 PSU Banks | 12 | Canara, Indian Bank, BOI | +31-68% YTD outperformance |
| 🏗️ Infra/Rail | 14 | RVNL, IRFC, IRCON, KEC | ₹11L cr budget allocation |
| ⚡ Energy | 17 | NTPC, Suzlon, SJVN, BPCL | Energy security + renewables |
| 💻 IT/Tech | 15 | TCS, Infosys, Persistent | Mission 500 India-US trade |
| 💊 Pharma | 13 | Sun, Dr Reddy, Laurus | China+1 API sourcing |
| ⚙️ Metals | 9 | Tata Steel, NMDC, Vedanta | Infra demand |
| 🚗 Auto/EV | 10 | Tata Motors, M&M, Olectra | PLI scheme, EV transition |
| 🛒 FMCG | 12 | HUL, ITC, Dabur | Rural consumption |
| 💰 Finance | 14 | HDFC, ICICI, Bajaj Finance | Credit growth |
| 🧪 Chemicals | 13 | SRF, Deepak Nitrite, Navin | China+1 shift |
| 🏠 Realty/Cement | 11 | DLF, Lodha, Ultratech | Infra supercycle |
| 📡 Telecom | 8 | Airtel, Indus Towers, HFCL | 5G rollout |

---

## Key Design Decisions (don't change these)

1. **`ALL_SYMBOLS`** in `config.py` is the master list — always deduplicated
2. **`INTRADAY_STOCKS`** in `intraday_config.py` is the intraday scan list
3. **`SECTOR_GROUPS`** dict maps sector names → symbol lists (used for UI filter)
4. **`SYMBOL_TO_SECTOR`** reverse map is built in `live_trader.py` at startup
5. Intraday engine uses 5m candles, 5d period (yfinance limit)
6. ML engine uses TimeSeriesSplit (not random split) — critical for finance
7. Confidence score = 50% ML + 30% entry timing + 20% risk score

---

## Prompt To Give New Kiro Session

Copy-paste this exactly:

```
I have a QuantSignal India trading project. The HANDOFF.md file in the 
workspace root describes the full project state. All backend engines are 
complete. Continue from where it left off:

1. Read HANDOFF.md first
2. Build frontend/app.py (Streamlit analytics dashboard with Monte Carlo, 
   Portfolio Optimization, Options Pricing, Risk Analysis pages)
3. Build backend/main.py (FastAPI with 8 endpoints)
4. The stock universe is ~200 stocks across 13 sectors in backend/config.py
5. Keep all geopolitical themes (defence, PSU banks, infra, energy, IT, 
   chemicals, EV) in mind when building signals

Run command: streamlit run live_trader.py --server.port 3000
```

---

## Dependencies

```
streamlit>=1.30.0
plotly>=5.18.0
yfinance>=0.2.31
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
```

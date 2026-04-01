"""
Configuration — QuantSignal India
Expanded stock universe: ~200 stocks across NIFTY50, NIFTY Next50,
NIFTY Midcap150, and high-conviction thematic picks.

Geopolitical/macro themes baked in (as of 2025-2026):
  - India-US 10-year defence framework → HAL, BEL, BDL, GRSE, MDL, MTAR, DPSL
  - Make in India / PLI schemes → electronics, pharma, chemicals, EV
  - PSU Bank outperformance → Canara, Indian Bank, BOI, UCO, BOB
  - Energy security + renewables → NTPC, Adani Green, Tata Power, CESC, SJVN
  - Infrastructure supercycle → L&T, RVNL, IRFC, IRCON, KEC, Kalpataru
  - India-US Mission 500 trade deal → IT, pharma exports, chemicals
  - China+1 strategy → textiles, chemicals, electronics manufacturing
  - India-Middle East corridor → logistics, ports, cement
"""

# ── NIFTY 50 ────────────────────────────────────────────────────────────────
NIFTY50_SYMBOLS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS", "LT.NS",
    "ITC.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "ULTRACEMCO.NS", "WIPRO.NS",
    "NESTLEIND.NS", "BAJAJFINSV.NS", "TATAMOTORS.NS", "POWERGRID.NS", "NTPC.NS",
    "ONGC.NS", "JSWSTEEL.NS", "M&M.NS", "TATASTEEL.NS", "ADANIENT.NS",
    "ADANIPORTS.NS", "COALINDIA.NS", "GRASIM.NS", "TECHM.NS", "INDUSINDBK.NS",
    "HINDALCO.NS", "DRREDDY.NS", "DIVISLAB.NS", "CIPLA.NS", "BPCL.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "APOLLOHOSP.NS", "TATACONSUM.NS", "HEROMOTOCO.NS",
    "SBILIFE.NS", "HDFCLIFE.NS", "BAJAJ-AUTO.NS", "UPL.NS", "SHRIRAMFIN.NS",
]

# ── NIFTY Next 50 ───────────────────────────────────────────────────────────
NIFTY_NEXT50_SYMBOLS = [
    "ADANIGREEN.NS", "ADANITRANS.NS", "AMBUJACEM.NS", "AUROPHARMA.NS",
    "BAJAJHLDNG.NS", "BANKBARODA.NS", "BERGEPAINT.NS", "BOSCHLTD.NS",
    "CANBK.NS", "CHOLAFIN.NS", "COLPAL.NS", "DABUR.NS", "DLF.NS",
    "GAIL.NS", "GODREJCP.NS", "HAVELLS.NS", "ICICIPRULI.NS", "INDHOTEL.NS",
    "INDUSTOWER.NS", "IOC.NS", "IRCTC.NS", "JINDALSTEL.NS", "LICI.NS",
    "LODHA.NS", "LUPIN.NS", "MARICO.NS", "MCDOWELL-N.NS", "MUTHOOTFIN.NS",
    "NAUKRI.NS", "NHPC.NS", "NMDC.NS", "OBEROIRLTY.NS", "OFSS.NS",
    "PAGEIND.NS", "PIDILITIND.NS", "PIIND.NS", "PNB.NS", "RECLTD.NS",
    "SAIL.NS", "SIEMENS.NS", "SRF.NS", "TATAPOWER.NS", "TORNTPHARM.NS",
    "TRENT.NS", "UNIONBANK.NS", "VBL.NS", "VEDL.NS", "ZOMATO.NS",
    "ZYDUSLIFE.NS", "PFC.NS",
]

# ── DEFENCE & AEROSPACE (Geopolitical theme: India-US defence deal, Make in India) ──
DEFENCE_SYMBOLS = [
    "HAL.NS",          # Hindustan Aeronautics — fighter jets, helicopters
    "BEL.NS",          # Bharat Electronics — radar, defence electronics
    "BDL.NS",          # Bharat Dynamics — missiles
    "GRSE.NS",         # Garden Reach Shipbuilders — naval vessels
    "MAZDOCK.NS",      # Mazagon Dock — submarines, destroyers
    "MTAR.NS",         # MTAR Technologies — precision defence components
    "DPSL.NS",         # Data Patterns — defence electronics
    "PARAS.NS",        # Paras Defence — optics, space
    "BEML.NS",         # BEML — defence vehicles, metro rail
    "COCHINSHIP.NS",   # Cochin Shipyard — naval + commercial
    "SOLARA.NS",       # Solar Industries — explosives, ammunition
    "ASTRA.NS",        # Astra Microwave — defence microwave components
    "ZENTEC.NS",       # Zen Technologies — defence training simulators
]

# ── PSU BANKS (Outperforming: +31-68% in 2025) ──────────────────────────────
PSU_BANK_SYMBOLS = [
    "SBIN.NS", "PNB.NS", "BANKBARODA.NS", "CANBK.NS", "UNIONBANK.NS",
    "INDIANB.NS",      # Indian Bank — top PSU bank performer
    "BANKINDIA.NS",    # Bank of India
    "CENTRALBK.NS",    # Central Bank of India
    "UCOBANK.NS",      # UCO Bank
    "MAHABANK.NS",     # Bank of Maharashtra
    "PSB.NS",          # Punjab & Sind Bank
    "JKBANK.NS",       # J&K Bank
]

# ── INFRASTRUCTURE & RAILWAYS (India infra supercycle) ──────────────────────
INFRA_SYMBOLS = [
    "LT.NS", "RVNL.NS", "IRFC.NS", "IRCON.NS", "IRCTC.NS",
    "KECL.NS",         # KEC International — power transmission towers
    "KALPATPOWR.NS",   # Kalpataru Power
    "NBCC.NS",         # NBCC — govt construction
    "NCC.NS",          # NCC Ltd — civil construction
    "PNCINFRA.NS",     # PNC Infratech — highways
    "HGINFRA.NS",      # H.G. Infra Engineering
    "GPPL.NS",         # Gujarat Pipavav Port
    "ADANIPORTS.NS",   # Adani Ports
    "CONCOR.NS",       # Container Corp — logistics
    "TIINDIA.NS",      # Tube Investments — engineering
    "APLAPOLLO.NS",    # APL Apollo Tubes — steel tubes
    "JINDALSAW.NS",    # Jindal SAW — pipes
    "WELSPUNIND.NS",   # Welspun India
]

# ── ENERGY & POWER (Energy security theme) ──────────────────────────────────
ENERGY_SYMBOLS = [
    "NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS",
    "NHPC.NS", "SJVN.NS",      # SJVN — hydro power
    "CESC.NS",                  # CESC — power utility
    "TORNTPOWER.NS",            # Torrent Power
    "JSWENERGY.NS",             # JSW Energy
    "GREENPANEL.NS",            # Green Panel
    "SUZLON.NS",                # Suzlon Energy — wind energy
    "INOXWIND.NS",              # Inox Wind
    "BPCL.NS", "IOC.NS", "ONGC.NS", "GAIL.NS",
    "OIL.NS",                   # Oil India
    "MRPL.NS",                  # MRPL — refinery
    "PETRONET.NS",              # Petronet LNG
    "IGL.NS",                   # Indraprastha Gas
    "MGL.NS",                   # Mahanagar Gas
    "GUJGASLTD.NS",             # Gujarat Gas
]

# ── IT & TECH (India-US Mission 500 trade deal beneficiary) ─────────────────
IT_SYMBOLS = [
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS",
    "LTIM.NS",          # LTIMindtree
    "MPHASIS.NS",       # Mphasis
    "PERSISTENT.NS",    # Persistent Systems
    "COFORGE.NS",       # Coforge
    "KPITTECH.NS",      # KPIT Technologies — auto software
    "TATAELXSI.NS",     # Tata Elxsi — design + tech
    "ZENSARTECH.NS",    # Zensar Technologies
    "MASTEK.NS",        # Mastek
    "NIITLTD.NS",       # NIIT
    "RATEGAIN.NS",      # RateGain — travel tech
    "NAUKRI.NS",        # Info Edge (Naukri)
    "POLICYBZR.NS",     # PB Fintech (PolicyBazaar)
    "PAYTM.NS",         # One97 Communications
    "ZOMATO.NS",        # Zomato
    "JIOFIN.NS",        # Jio Financial Services
]

# ── PHARMA & HEALTHCARE (US trade deal + China+1 API sourcing) ──────────────
PHARMA_SYMBOLS = [
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "AUROPHARMA.NS", "LUPIN.NS", "TORNTPHARM.NS", "ZYDUSLIFE.NS",
    "ALKEM.NS",         # Alkem Laboratories
    "IPCALAB.NS",       # IPCA Laboratories
    "GLENMARK.NS",      # Glenmark Pharma
    "NATCOPHARM.NS",    # Natco Pharma
    "LAURUSLABS.NS",    # Laurus Labs — API manufacturer
    "GRANULES.NS",      # Granules India
    "SUVEN.NS",         # Suven Pharma
    "APOLLOHOSP.NS",    # Apollo Hospitals
    "FORTIS.NS",        # Fortis Healthcare
    "MAXHEALTH.NS",     # Max Healthcare
    "METROPOLIS.NS",    # Metropolis Healthcare
    "THYROCARE.NS",     # Thyrocare
]

# ── METALS & MINING (Infrastructure demand + China+1) ───────────────────────
METALS_SYMBOLS = [
    "TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS",
    "SAIL.NS", "NMDC.NS", "COALINDIA.NS",
    "NATIONALUM.NS",    # National Aluminium
    "HINDCOPPER.NS",    # Hindustan Copper
    "MOIL.NS",          # MOIL — manganese
    "RATNAMANI.NS",     # Ratnamani Metals — stainless steel tubes
    "WELCORP.NS",       # Welspun Corp — pipes
    "JINDALSAW.NS",     # Jindal SAW
    "JSWINFRA.NS",      # JSW Infrastructure
    "APLAPOLLO.NS",     # APL Apollo Tubes
]

# ── AUTO & EV (PLI scheme, EV transition) ───────────────────────────────────
AUTO_SYMBOLS = [
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS",
    "HEROMOTOCO.NS", "EICHERMOT.NS",
    "ASHOKLEY.NS",      # Ashok Leyland — commercial vehicles
    "TVSMOTOR.NS",      # TVS Motor
    "BALKRISIND.NS",    # Balkrishna Industries — tyres
    "MRF.NS",           # MRF — tyres
    "APOLLOTYRE.NS",    # Apollo Tyres
    "MOTHERSON.NS",     # Samvardhana Motherson — auto components
    "BOSCHLTD.NS",      # Bosch — auto tech
    "BHARATFORG.NS",    # Bharat Forge — forgings
    "SUNDRMFAST.NS",    # Sundram Fasteners
    "EXIDEIND.NS",      # Exide Industries — batteries
    "AMARARAJA.NS",     # Amara Raja — batteries
    "OLECTRA.NS",       # Olectra Greentech — electric buses
    "TATAELXSI.NS",     # Tata Elxsi — EV software
    "KPITTECH.NS",      # KPIT — EV software
]

# ── FMCG & CONSUMER ─────────────────────────────────────────────────────────
FMCG_SYMBOLS = [
    "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "BRITANNIA.NS",
    "DABUR.NS", "MARICO.NS", "COLPAL.NS", "GODREJCP.NS",
    "TATACONSUM.NS", "VBL.NS",
    "EMAMILTD.NS",      # Emami
    "JYOTHYLAB.NS",     # Jyothy Labs
    "RADICO.NS",        # Radico Khaitan — spirits
    "MCDOWELL-N.NS",    # United Spirits
    "UNITDSPR.NS",      # United Breweries
    "PATANJALI.NS",     # Patanjali Foods
    "BIKAJI.NS",        # Bikaji Foods
    "DEVYANI.NS",       # Devyani International (KFC/Pizza Hut)
    "WESTLIFE.NS",      # Westlife Foodworld (McDonald's)
    "JUBLFOOD.NS",      # Jubilant Foodworks (Domino's)
]

# ── FINANCIALS & NBFC ────────────────────────────────────────────────────────
FINANCE_SYMBOLS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS",
    "INDUSINDBK.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "SHRIRAMFIN.NS",
    "CHOLAFIN.NS", "MUTHOOTFIN.NS",
    "MANAPPURAM.NS",    # Manappuram Finance — gold loans
    "LICHSGFIN.NS",     # LIC Housing Finance
    "PNBHOUSING.NS",    # PNB Housing Finance
    "CANFINHOME.NS",    # Can Fin Homes
    "AAVAS.NS",         # Aavas Financiers
    "HOMEFIRST.NS",     # Home First Finance
    "CREDITACC.NS",     # CreditAccess Grameen — microfinance
    "UJJIVANSFB.NS",    # Ujjivan Small Finance Bank
    "EQUITASBNK.NS",    # Equitas Small Finance Bank
    "SBICARD.NS",       # SBI Cards
    "HDFCAMC.NS",       # HDFC AMC
    "NIPPONLIFE.NS",    # Nippon India AMC
    "ANGELONE.NS",      # Angel One — broking
    "ICICIPRULI.NS",    # ICICI Prudential Life
    "SBILIFE.NS",       # SBI Life
    "HDFCLIFE.NS",      # HDFC Life
    "LICI.NS",          # LIC India
]

# ── CHEMICALS (China+1 beneficiary) ─────────────────────────────────────────
CHEMICALS_SYMBOLS = [
    "PIDILITIND.NS",    # Pidilite — adhesives
    "SRF.NS",           # SRF — specialty chemicals
    "PIIND.NS",         # PI Industries — agrochemicals
    "UPL.NS",           # UPL — agrochemicals
    "ATUL.NS",          # Atul Ltd — specialty chemicals
    "DEEPAKNTR.NS",     # Deepak Nitrite
    "NAVINFLUOR.NS",    # Navin Fluorine — fluorochemicals
    "FLUOROCHEM.NS",    # Gujarat Fluorochemicals
    "CLEAN.NS",         # Clean Science
    "FINEORG.NS",       # Fine Organic Industries
    "TATACHEM.NS",      # Tata Chemicals
    "GNFC.NS",          # GNFC — fertilizers + chemicals
    "COROMANDEL.NS",    # Coromandel International — fertilizers
    "CHAMBLFERT.NS",    # Chambal Fertilizers
    "GSFC.NS",          # GSFC
]

# ── REAL ESTATE & CEMENT ─────────────────────────────────────────────────────
REALTY_CEMENT_SYMBOLS = [
    "DLF.NS", "LODHA.NS", "OBEROIRLTY.NS",
    "GODREJPROP.NS",    # Godrej Properties
    "PRESTIGE.NS",      # Prestige Estates
    "BRIGADE.NS",       # Brigade Enterprises
    "SOBHA.NS",         # Sobha Ltd
    "PHOENIXLTD.NS",    # Phoenix Mills — malls
    "ULTRACEMCO.NS", "AMBUJACEM.NS", "GRASIM.NS",
    "SHREECEM.NS",      # Shree Cement
    "JKCEMENT.NS",      # JK Cement
    "RAMCOCEM.NS",      # Ramco Cements
    "HEIDELBERG.NS",    # HeidelbergCement India
    "NUVOCO.NS",        # Nuvoco Vistas
]

# ── TELECOM & MEDIA ──────────────────────────────────────────────────────────
TELECOM_SYMBOLS = [
    "BHARTIARTL.NS",    # Airtel — 5G rollout
    "INDUSTOWER.NS",    # Indus Towers — tower infra
    "IDEA.NS",          # Vodafone Idea — turnaround play
    "TATACOMM.NS",      # Tata Communications
    "HFCL.NS",          # HFCL — fibre optics
    "STLTECH.NS",       # STL — optical fibre
    "RAILTEL.NS",       # RailTel — govt telecom
    "ROUTE.NS",         # Route Mobile — CPaaS
    "TANLA.NS",         # Tanla Platforms — messaging
]

# ── FULL MASTER UNIVERSE (deduplicated) ─────────────────────────────────────
_all = (
    NIFTY50_SYMBOLS + NIFTY_NEXT50_SYMBOLS + DEFENCE_SYMBOLS +
    PSU_BANK_SYMBOLS + INFRA_SYMBOLS + ENERGY_SYMBOLS + IT_SYMBOLS +
    PHARMA_SYMBOLS + METALS_SYMBOLS + AUTO_SYMBOLS + FMCG_SYMBOLS +
    FINANCE_SYMBOLS + CHEMICALS_SYMBOLS + REALTY_CEMENT_SYMBOLS +
    TELECOM_SYMBOLS
)
ALL_SYMBOLS = list(dict.fromkeys(_all))   # preserve order, remove dupes

NIFTY_INDEX = "^NSEI"
INDIA_VIX = "^INDIAVIX"

# ── Default parameters ───────────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_SHORT_WINDOW = 50
DEFAULT_LONG_WINDOW = 200
RSI_PERIOD = 14
VOLATILITY_WINDOW = 20
MOMENTUM_WINDOW = 200

# ── Signal thresholds ────────────────────────────────────────────────────────
MIN_CONFIDENCE = 0.65
MIN_RISK_REWARD = 1.8
MAX_SIGNALS = 20
MIN_SIGNALS = 5

# ── Monte Carlo ──────────────────────────────────────────────────────────────
MC_SIMULATIONS = 10000
MC_DAYS = 30

# ── Portfolio optimization ───────────────────────────────────────────────────
PORTFOLIO_SIMULATIONS = 50000
RISK_FREE_RATE = 0.065   # India 10Y ~6.5%

# ── Market timing weights ────────────────────────────────────────────────────
MOMENTUM_WEIGHT = 0.40
VOLATILITY_WEIGHT = 0.35
DRAWDOWN_WEIGHT = 0.25

# ── Final signal score weights ───────────────────────────────────────────────
ML_WEIGHT = 0.50
ENTRY_SCORE_WEIGHT = 0.30
RISK_WEIGHT = 0.20

# ── Thematic sector map (for sector-aware scanning) ──────────────────────────
SECTOR_MAP = {
    "defence":    DEFENCE_SYMBOLS,
    "psu_banks":  PSU_BANK_SYMBOLS,
    "infra":      INFRA_SYMBOLS,
    "energy":     ENERGY_SYMBOLS,
    "it":         IT_SYMBOLS,
    "pharma":     PHARMA_SYMBOLS,
    "metals":     METALS_SYMBOLS,
    "auto":       AUTO_SYMBOLS,
    "fmcg":       FMCG_SYMBOLS,
    "finance":    FINANCE_SYMBOLS,
    "chemicals":  CHEMICALS_SYMBOLS,
    "realty":     REALTY_CEMENT_SYMBOLS,
    "telecom":    TELECOM_SYMBOLS,
}

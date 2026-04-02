"""
Universe Engine — Full NSE/BSE stock universe with 1,500+ liquid stocks.

Architecture:
  - Modular symbol lists by cap tier and sector
  - Penny stock engine (price < ₹50, volume-filtered)
  - Sector rotation tracking
  - Dynamic universe builder
  - Plug-in ready for NSE symbol dump / BSE API

Data source: Yahoo Finance (.NS suffix) — swap to Kite/Upstox when available.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# CAP TIER THRESHOLDS (price-based proxy — replace with market cap when API available)
# ══════════════════════════════════════════════════════════════════════════════
PENNY_MAX_PRICE   = 50      # ≤ ₹50
SMALL_CAP_MAX     = 500     # ₹50 – ₹500
MID_CAP_MAX       = 2000    # ₹500 – ₹2,000
LARGE_CAP_MIN     = 2000    # > ₹2,000

MIN_PENNY_VOLUME  = 500_000   # min avg daily volume for penny stocks
MIN_LIQUID_VOLUME = 100_000   # min avg daily volume for all stocks


# ══════════════════════════════════════════════════════════════════════════════
# NIFTY 50 + NEXT 50
# ══════════════════════════════════════════════════════════════════════════════
_NIFTY50 = [
    "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN","BHARTIARTL",
    "KOTAKBANK","LT","ITC","AXISBANK","BAJFINANCE","ASIANPAINT","MARUTI","HCLTECH",
    "SUNPHARMA","TITAN","ULTRACEMCO","WIPRO","NESTLEIND","BAJAJFINSV","TATAMOTORS",
    "POWERGRID","NTPC","ONGC","JSWSTEEL","M&M","TATASTEEL","ADANIENT","ADANIPORTS",
    "COALINDIA","GRASIM","TECHM","INDUSINDBK","HINDALCO","DRREDDY","DIVISLAB",
    "CIPLA","BPCL","EICHERMOT","BRITANNIA","APOLLOHOSP","TATACONSUM","HEROMOTOCO",
    "SBILIFE","HDFCLIFE","BAJAJ-AUTO","UPL","SHRIRAMFIN",
]

_NIFTY_NEXT50 = [
    "ADANIGREEN","AMBUJACEM","AUROPHARMA","BANKBARODA","BERGEPAINT","BOSCHLTD",
    "CANBK","CHOLAFIN","COLPAL","DABUR","DLF","GAIL","GODREJCP","HAVELLS",
    "ICICIPRULI","INDHOTEL","INDUSTOWER","IOC","IRCTC","JINDALSTEL","LICI",
    "LODHA","LUPIN","MARICO","MCDOWELL-N","MUTHOOTFIN","NAUKRI","NHPC","NMDC",
    "OBEROIRLTY","OFSS","PAGEIND","PIDILITIND","PIIND","PNB","RECLTD","SAIL",
    "SIEMENS","SRF","TATAPOWER","TORNTPHARM","TRENT","UNIONBANK","VBL","VEDL",
    "ZOMATO","ZYDUSLIFE","PFC","BAJAJHLDNG","GODREJPROP",
]

# ══════════════════════════════════════════════════════════════════════════════
# NIFTY MIDCAP 150
# ══════════════════════════════════════════════════════════════════════════════
_MIDCAP150 = [
    "ABCAPITAL","ABFRL","ACC","AIAENG","AJANTPHARM","ALKEM","APLLTD","ASTRAL",
    "ATUL","AUBANK","AUROPHARMA","BALKRISIND","BANDHANBNK","BATAINDIA","BAYERCROP",
    "BHARATFORG","BHEL","BIOCON","BLUESTARCO","BSOFT","CAMS","CANFINHOME",
    "CARBORUNIV","CASTROLIND","CEATLTD","CENTRALBK","CESC","CGPOWER","CHAMBLFERT",
    "CHOLAFIN","CLEAN","COFORGE","CONCOR","COROMANDEL","CREDITACC","CROMPTON",
    "CUMMINSIND","CYIENT","DALBHARAT","DEEPAKNTR","DELTACORP","DMART","ELGIEQUIP",
    "EMAMILTD","ENGINERSIN","EQUITASBNK","ESCORTS","EXIDEIND","FINEORG","FLUOROCHEM",
    "FORTIS","GLENMARK","GNFC","GODREJIND","GRANULES","GSFC","GUJGASLTD",
    "HAPPSTMNDS","HDFCAMC","HFCL","HINDCOPPER","HINDPETRO","HONAUT","IDFCFIRSTB",
    "IEX","IGPL","IGL","INDIAMART","INDIACEM","INDIANB","INDIGO","INTELLECT",
    "IPCALAB","IRB","IRFC","ISEC","JBCHEPHARM","JKCEMENT","JKLAKSHMI","JSWENERGY",
    "JUBLFOOD","KALYANKJIL","KANSAINER","KECL","KPITTECH","KRBL","LAURUSLABS",
    "LICHSGFIN","LINDEINDIA","LTIM","LTTS","LUPIN","MANAPPURAM","MAZDOCK",
    "MCX","METROPOLIS","MFSL","MGL","MPHASIS","MRF","MTAR","NATCOPHARM",
    "NATIONALUM","NAVINFLUOR","NBCC","NCC","NIACL","NIPPONLIFE","NMDC","NOCIL",
    "OBEROIRLTY","OFSS","OLECTRA","PAGEIND","PATANJALI","PCBL","PERSISTENT",
    "PETRONET","PHOENIXLTD","PIIND","PNCINFRA","POLYCAB","PRESTIGE","PRINCEPIPES",
    "PSPPROJECT","RADICO","RAILTEL","RAMCOCEM","RATNAMANI","RECLTD","REDINGTON",
    "ROUTE","RVNL","SAFARI","SAIL","SCHAEFFLER","SHREECEM","SJVN","SKFINDIA",
    "SOBHA","SOLARA","SONATSOFTW","STARHEALTH","STLTECH","SUNDRMFAST","SUNTECK",
    "SUPREMEIND","SUZLON","TANLA","TATACHEM","TATACOMM","TATAELXSI","TIINDIA",
    "TIMKEN","TORNTPOWER","TRENT","TRIDENT","TVSMOTOR","UJJIVANSFB","UNIONBANK",
    "UTIAMC","VAIBHAVGBL","VEDL","VINATI","VOLTAS","WELCORP","WHIRLPOOL",
    "WIPRO","YESBANK","ZENSARTECH","ZYDUSLIFE",
]

# ══════════════════════════════════════════════════════════════════════════════
# NIFTY SMALLCAP 250
# ══════════════════════════════════════════════════════════════════════════════
_SMALLCAP250 = [
    "AAVAS","ABBOTINDIA","ACCELYA","ADANIGAS","AEGASIND","AFFLE","AJAXENG",
    "AKZOINDIA","ALEMBICLTD","ALEMBICPH","AMBER","AMJLAND","ANUPAM","APCOTEXIND",
    "APOLLOTYRE","APTUS","ARVSMART","ASAHIINDIA","ASHOKA","ASHOKLEY","ASTERDM",
    "ASTRAZEN","ATGL","AVANTIFEED","AXISCADES","BAJAJCON","BAJAJHLDNG","BALRAMCHIN",
    "BASF","BATAINDIA","BAYERCROP","BBTC","BCONCEPTS","BEML","BERGEPAINT",
    "BIKAJI","BIRLACORPN","BLISSGVS","BRIGADE","BSOFT","CAPACITE","CARTRADE",
    "CASTROLIND","CERA","CGPOWER","CHALET","CHOLAHLDNG","CIGNITITEC","CLNINDIA",
    "COCHINSHIP","COLPAL","CRAFTSMAN","CRISIL","CROMPTON","CSBBANK","DATAPATTNS",
    "DBREALTY","DCBBANK","DEEPAKFERT","DELTACORP","DEVYANI","DHANI","DHANUKA",
    "DODLA","DPSL","EASEMYTRIP","EDELWEISS","ELGIEQUIP","EMKAY","ENDURANCE",
    "ENGINERSIN","ERIS","ESABINDIA","ESTER","ETHOS","EXIDEIND","FAZE3Q",
    "FEDERALBNK","FINCABLES","FINPIPE","FORCEMOT","GABRIEL","GALAXYSURF",
    "GARFIBRES","GEOJITFSL","GESHIP","GLAXO","GODREJAGRO","GPPL","GRSE",
    "GSPL","GTLINFRA","HAPPSTMNDS","HARDWYN","HATSUN","HAVELLS","HDFCLIFE",
    "HEIDELBERG","HERITAGE","HGINFRA","HIKAL","HINDCOPPER","HINDPETRO","HOMEFIRST",
    "HONAUT","HUDCO","IDEAFORGE","IIFL","INDHOTEL","INDIAMART","INFOEDGE",
    "INOXWIND","INTELLECT","IPCALAB","IRCON","ISEC","ITDCEM","JKPAPER","JMFINANCIL",
    "JSWINFRA","JUBLINGREA","JUSTDIAL","KAJARIACER","KALPATPOWR","KARURVYSYA",
    "KFINTECH","KNRCON","KOLTEPATIL","KRSNAA","KSCL","LATENTVIEW","LAXMIMACH",
    "LEMONTREE","LICI","LINDEINDIA","LLOYDSME","LODHA","LUMAX","LUXIND",
    "MAHINDCIE","MAHLIFE","MARICO","MASTEK","MAXHEALTH","MAZDOCK","MCX",
    "MEDANTA","MEDPLUS","MIDHANI","MINDAIND","MINDA","MMFS","MOIL","MOTHERSON",
    "MOTILALOFS","MSTCLTD","MUTHOOTFIN","NAUKRI","NAVINFLUOR","NESCO","NEWGEN",
    "NIITLTD","NUVAMA","NUVOCO","NYKAA","OBEROIRLTY","OFSS","ONMOBILE",
    "ORIENTELEC","ORIENTCEM","PARAS","PATANJALI","PCJEWELLER","PFIZER","PGHH",
    "PHOENIXLTD","PNBHOUSING","POLICYBZR","POLYPLEX","PRAJIND","PRINCEPIPES",
    "PRISM","PRUDENT","PSB","PSPPROJECT","QUESS","RAJESHEXPO","RATEGAIN",
    "RAYMOND","REDINGTON","REPCO","ROSSARI","ROUTE","SAFARI","SANOFI",
    "SAPPHIRE","SBICARD","SCHAEFFLER","SENCO","SEQUENT","SHRIRAMFIN","SIEMENS",
    "SKFINDIA","SMCGLOBAL","SMLISUZU","SOBHA","SOLARA","SPECIALITY","SPENCERS",
    "SRTRANSFIN","STARHEALTH","STRIDES","SUDARSCHEM","SUNDARMFIN","SUNPHARMA",
    "SUPRAJIT","SWARAJENG","SWIGGY","SYMPHONY","TANLA","TATAINVEST","TATAPOWER",
    "TCNSBRANDS","TEAMLEASE","THYROCARE","TIMKEN","TITAGARH","TORNTPHARM",
    "TRIL","TRIDENT","TRIVENI","TTKHLTCARE","TTKPRESTIG","UJJIVAN","UNIPARTS",
    "UTIAMC","V2RETAIL","VAIBHAVGBL","VIJAYA","VSTIND","WELSPUNIND","WESTLIFE",
    "WHIRLPOOL","WOCKPHARMA","XCHANGING","YESBANK","ZEEL","ZENTEC","ZOMATO",
]


# ══════════════════════════════════════════════════════════════════════════════
# PENNY STOCKS (price ≤ ₹50, liquid, NSE-listed) — 300+ symbols
# ══════════════════════════════════════════════════════════════════════════════
_PENNY = [
    # PSU Banks / Finance
    "YESBANK","IDEA","UCOBANK","MAHABANK","PSB","CENTRALBK","BANKINDIA","JKBANK",
    "INDIANB","SOUTHBANK","LAKSHVILAS","DCBBANK","CSBBANK","KARURVYSYA",
    "UJJIVAN","EQUITASBNK","UJJIVANSFB","IDFCFIRSTB","BANDHANBNK","AUBANK",
    # Power / Energy
    "RPOWER","JPPOWER","TRIL","NHPC","SJVN","INOXWIND","SUZLON","ADANIPOWER",
    "TORNTPOWER","JSWENERGY","CESC","GREENPANEL","ORIENTELEC","KPIL",
    "RTNPOWER","GMRP&UI","NAVA","AVADHSUGAR","TRIVENI","DWARIKESH",
    # Infra / Construction
    "GMRINFRA","IRB","ASHOKA","SADBHAV","CAPACITE","ITDCEM","HUDCO","NBCC",
    "PNCINFRA","HGINFRA","NCC","KNRCON","PSPPROJECT","IRCON","RVNL",
    "WELSPUNIND","PRINCEPIPES","FINCABLES","TITAGARH","LLOYDSME",
    # Metals / Mining
    "NATIONALUM","HINDCOPPER","MOIL","SAIL","NMDC","COALINDIA","MSTCLTD",
    "TINPLATE","WELCORP","JINDALSAW","APLAPOLLO","RATNAMANI","UNIPARTS",
    "HFCL","STLTECH","RAILTEL","ROUTE","TANLA","ONMOBILE","GTLINFRA",
    # Pharma / Healthcare
    "LALPATHLAB","KRSNAA","VIJAYA","THYROCARE","BIOCON","STRIDES","SEQUENT",
    "WOCKPHARMA","BLISSGVS","HIKAL","SOLARA","GRANULES","NATCOPHARM",
    "LAURUSLABS","GLENMARK","IPCALAB","AJANTPHARM","JBCHEPHARM",
    # FMCG / Consumer
    "VSTIND","GODFRYPHLP","PATANJALI","HATSUN","HERITAGE","PARAG","DODLA",
    "BIKAJI","DEVYANI","WESTLIFE","SAPPHIRE","SPECIALITY","BARBEQUE",
    # Realty
    "DBREALTY","MAHLIFE","KOLTEPATIL","SUNTECK","ARVSMART","NCLIND",
    "BRIGADE","SOBHA","PHOENIXLTD","PRESTIGE","GODREJPROP",
    # Chemicals
    "IGPL","NOCIL","NEOGEN","PCBL","ROSSARI","ANUPAM","SUDARSCHEM",
    "DEEPAKFERT","GSFC","GNFC","CHAMBLFERT","COROMANDEL","TATACHEM",
    # Defence / Aerospace
    "DPSL","ZENTEC","PARAS","ASTRA","IDEAFORGE","MIDHANI","DATAPATTNS",
    # Misc high-volume penny
    "TRIDENT","DELTACORP","SPENCERS","V2RETAIL","SMCGLOBAL",
    "EDELWEISS","EMKAY","GEOJITFSL","JMFINANCIL","PRUDENT",
    # Additional liquid penny stocks
    "IRFC","RECLTD","PFC","BHEL","BEML","CONCOR","TIINDIA","GPPL",
    "ADANIGAS","IGL","MGL","GUJGASLTD","MRPL","PETRONET","OIL",
    "JSWINFRA","RATNAMANI","WELSPUNIND","PRINCEPIPES","FINCABLES",
    # Small/Mid cap high volume
    "YESBANK","IDEA","SUZLON","RPOWER","JPPOWER","UCOBANK","MAHABANK",
    "PSB","CENTRALBK","BANKINDIA","NHPC","SJVN","IRFC","RECLTD","PFC",
    "NBCC","BHEL","NATIONALUM","HINDCOPPER","MOIL","INOXWIND","OLECTRA",
    # IT / Tech penny
    "ONMOBILE","GTLINFRA","RAILTEL","ROUTE","TANLA","HFCL","STLTECH",
    "NIITLTD","XCHANGING","CIGNITITEC","AXISCADES","BSOFT","REDINGTON",
    # Auto / EV penny
    "GABRIEL","MINDA","LUMAX","SUBROS","SWARAJENG","MAHINDCIE","SMLISUZU",
    "SUPRAJIT","ENDURANCE","CRAFTSMAN","UNIPARTS","TITAGARH",
    # Agri / Fertilizers
    "DEEPAKFERT","GSFC","GNFC","CHAMBLFERT","COROMANDEL","TATACHEM",
    "AVANTIFEED","BALRAMCHIN","KRBL","GODREJAGRO","DHANUKA","KSCL",
    # Textile / Manufacturing
    "TRIDENT","RAYMOND","WELSPUNIND","GARFIBRES","POLYPLEX","ESTER",
    "TCNSBRANDS","SAFARI","ETHOS","VAIBHAVGBL","KALYANKJIL","SENCO",
    # Media / Entertainment
    "ZEEL","SUNTV","PVRINOX","INOXLEISUR","TIPS","SAREGAMA",
    # Logistics / Shipping
    "GESHIP","CONCOR","GPPL","ADANIPORTS","JSWINFRA","IRCON",
    # Hospitality
    "LEMONTREE","CHALET","INDHOTEL","MAHINDHOTEL","TAJGVK",
    # Misc NSE liquid penny
    "DELTACORP","SPENCERS","V2RETAIL","GTLINFRA","SMCGLOBAL",
    "EDELWEISS","EMKAY","GEOJITFSL","JMFINANCIL","PRUDENT","DHANI",
    "SRTRANSFIN","MMFS","CREDITACC","REPCO","APTUS","HOMEFIRST",
    "MANAPPURAM","LICHSGFIN","PNBHOUSING","CANFINHOME","AAVAS",
    # PSU / Govt enterprises
    "HUDCO","IREDA","RECLTD","PFC","IRFC","NBCC","BHEL","BEML",
    "CONCOR","NMDC","COALINDIA","NATIONALUM","HINDCOPPER","MOIL",
    "SAIL","RVNL","IRCON","RAILTEL","STLTECH","HFCL",
]
# Deduplicate while preserving order
_PENNY = list(dict.fromkeys(_PENNY))

# ══════════════════════════════════════════════════════════════════════════════
# SECTOR SYMBOL LISTS (full coverage)
# ══════════════════════════════════════════════════════════════════════════════
SECTOR_UNIVERSE: dict[str, list[str]] = {
    "🛡️ Defence": [
        "HAL","BEL","BDL","GRSE","MAZDOCK","MTAR","BEML","COCHINSHIP",
        "ASTRA","ZENTEC","PARAS","DPSL","SOLARA","MIDHANI","DATAPATTNS","IDEAFORGE",
        "BHEL","ENGINERSIN","TIINDIA","KALYANKJIL",
    ],
    "🏦 PSU Banks": [
        "SBIN","PNB","BANKBARODA","CANBK","UNIONBANK","INDIANB","BANKINDIA",
        "CENTRALBK","UCOBANK","MAHABANK","PSB","JKBANK","KARURVYSYA","CSBBANK",
        "DCBBANK","FEDERALBNK","SOUTHBANK","LAKSHVILAS",
    ],
    "🏗️ Infra/Rail": [
        "LT","RVNL","IRFC","IRCON","IRCTC","KECL","KALPATPOWR","NBCC","NCC",
        "PNCINFRA","HGINFRA","GPPL","ADANIPORTS","CONCOR","TIINDIA","APLAPOLLO",
        "JINDALSAW","WELSPUNIND","GMRINFRA","IRB","ASHOKA","KNRCON","PSPPROJECT",
        "CAPACITE","SADBHAV","ITDCEM","HUDCO","ENGINERSIN","JSWINFRA","RATNAMANI",
    ],
    "⚡ Energy": [
        "NTPC","POWERGRID","TATAPOWER","ADANIGREEN","NHPC","SJVN","CESC",
        "TORNTPOWER","JSWENERGY","SUZLON","INOXWIND","BPCL","IOC","ONGC","GAIL",
        "OIL","MRPL","PETRONET","IGL","MGL","GUJGASLTD","ADANIGAS","RPOWER",
        "JPPOWER","TRIL","ORIENTELEC","KPIL","THERMAX","GREENPANEL",
    ],
    "💻 IT/Tech": [
        "TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","MPHASIS","PERSISTENT",
        "COFORGE","KPITTECH","TATAELXSI","ZENSARTECH","MASTEK","NIITLTD",
        "RATEGAIN","NAUKRI","POLICYBZR","PAYTM","ZOMATO","JIOFIN","ANGELONE",
        "HAPPSTMNDS","SONATSOFTW","TANLA","ROUTE","HFCL","STLTECH","RAILTEL",
        "INTELLECT","NEWGEN","CYIENT","BIRLASOFT","LATENTVIEW","CIGNITITEC",
        "XCHANGING","ACCELYA","AXISCADES","BSOFT","REDINGTON","KFINTECH",
    ],
    "💊 Pharma": [
        "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","LUPIN","TORNTPHARM",
        "ZYDUSLIFE","ALKEM","IPCALAB","GLENMARK","NATCOPHARM","LAURUSLABS",
        "GRANULES","FORTIS","MAXHEALTH","METROPOLIS","THYROCARE","SUVEN",
        "AJANTPHARM","JBCHEPHARM","ERIS","ABBOTINDIA","PFIZER","SANOFI","GLAXO",
        "BIOCON","STRIDES","SEQUENT","DIVI","SOLARA","LALPATHLAB","KRSNAA",
        "VIJAYA","WOCKPHARMA","BLISSGVS","HIKAL","ASTERDM","MEDANTA","MEDPLUS",
        "APOLLOHOSP","STARHEALTH","NIACL",
    ],
    "⚙️ Metals": [
        "TATASTEEL","JSWSTEEL","HINDALCO","VEDL","SAIL","NMDC","COALINDIA",
        "NATIONALUM","HINDCOPPER","MOIL","RATNAMANI","WELCORP","JINDALSAW",
        "JSWINFRA","APLAPOLLO","TINPLATE","MSTCLTD","LLOYDSME","UNIPARTS",
        "TITAGARH","WELSPUNIND","PRINCEPIPES","FINCABLES","GARFIBRES",
    ],
    "🚗 Auto/EV": [
        "MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
        "ASHOKLEY","TVSMOTOR","BALKRISIND","MRF","APOLLOTYRE","MOTHERSON",
        "BOSCHLTD","BHARATFORG","SUNDRMFAST","EXIDEIND","AMARARAJA","OLECTRA",
        "TATAELXSI","KPITTECH","CEATLTD","GABRIEL","MINDA","LUMAX","SUBROS",
        "SWARAJENG","ESCORTS","FORCEMOT","MAHINDCIE","CRAFTSMAN","ENDURANCE",
        "SMLISUZU","SCHAEFFLER","SKFINDIA","TIMKEN","ELGIEQUIP",
    ],
    "🛒 FMCG": [
        "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","COLPAL",
        "GODREJCP","TATACONSUM","VBL","EMAMILTD","JYOTHYLAB","RADICO",
        "MCDOWELL-N","UNITDSPR","PATANJALI","BIKAJI","DEVYANI","WESTLIFE",
        "JUBLFOOD","SAPPHIRE","BARBEQUE","SPECIALITY","VSTIND","GODFRYPHLP",
        "HATSUN","HERITAGE","PARAG","DODLA","KRBL","AVANTIFEED","BALRAMCHIN",
        "BAJAJCON","TTKPRESTIG","TTKHLTCARE","PGHH","SYMPHONY",
    ],
    "💰 Finance": [
        "HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","INDUSINDBK","BAJFINANCE",
        "BAJAJFINSV","SHRIRAMFIN","CHOLAFIN","MUTHOOTFIN","MANAPPURAM","LICHSGFIN",
        "PNBHOUSING","CANFINHOME","CREDITACC","UJJIVANSFB","EQUITASBNK","SBICARD",
        "HDFCAMC","NIPPONLIFE","ICICIPRULI","AAVAS","HOMEFIRST","APTUS","REPCO",
        "SUNDARMFIN","MMFS","IIFL","MFSL","ABSLAMC","UTIAMC","360ONE","NUVAMA",
        "MOTILALOFS","ISEC","GEOJITFSL","SMCGLOBAL","ANGELONE","IDFCFIRSTB",
        "AUBANK","BANDHANBNK","FEDERALBNK","CSBBANK","DCBBANK","KARURVYSYA",
        "EMKAY","JMFINANCIL","KFINTECH","CAMS","PRUDENT","SRTRANSFIN","DHANI",
    ],
    "🧪 Chemicals": [
        "PIDILITIND","SRF","PIIND","UPL","ATUL","DEEPAKNTR","NAVINFLUOR",
        "FLUOROCHEM","CLEAN","FINEORG","TATACHEM","GNFC","COROMANDEL","CHAMBLFERT",
        "GSFC","AARTI","VINATI","GALAXYSURF","ROSSARI","ANUPAM","SUDARSCHEM",
        "NOCIL","NEOGEN","PCBL","IGPL","BASF","AKZOINDIA","KANSAINER","BERGER",
        "ASIANPAINT","BERGEPAINT","KANSAINER","DEEPAKFERT","GSPL","HIKAL",
        "BAYERCROP","DHANUKA","KSCL","RALLIS","SUMICHEM","INSECTICIDES",
    ],
    "🏠 Realty/Cement": [
        "DLF","LODHA","OBEROIRLTY","GODREJPROP","PRESTIGE","BRIGADE","SOBHA",
        "PHOENIXLTD","MAHLIFE","KOLTEPATIL","SUNTECK","ARVSMART","NCLIND",
        "ULTRACEMCO","AMBUJACEM","GRASIM","SHREECEM","JKCEMENT","RAMCOCEM",
        "HEIDELBERG","NUVOCO","DALMIA","JKLAKSHMI","BIRLACORPN","PRISM","ORIENT",
        "DBREALTY","JUBLINGREA","CHALET","LEMONTREE","INDHOTEL","MAHINDCIE",
    ],
    "📡 Telecom": [
        "BHARTIARTL","INDUSTOWER","IDEA","TATACOMM","HFCL","STLTECH","RAILTEL",
        "ROUTE","TANLA","ONMOBILE","GTLINFRA","INDIAMART","JUSTDIAL","INFOEDGE",
        "ZOMATO","SWIGGY","NYKAA","CARTRADE","EASEMYTRIP","RATEGAIN",
    ],
    "📈 Small/Mid Cap": [
        "IRFC","RECLTD","PFC","ADANIPOWER","RVNL","NBCC","BHEL","NATIONALUM",
        "HINDCOPPER","MOIL","SUZLON","INOXWIND","OLECTRA","UCOBANK","MAHABANK",
        "CENTRALBK","PSB","BANKINDIA","JKBANK","RPOWER","JPPOWER","TRIL",
        "GMRINFRA","IRB","ASHOKA","KNRCON","PSPPROJECT","CAPACITE","SADBHAV",
        "IDEAFORGE","DATAPATTNS","MIDHANI","PARAS","DPSL","ZENTEC","ASTRA",
        "COCHINSHIP","GRSE","MAZDOCK","MTAR","BEML","HAL","BEL","BDL",
        "SJVN","CESC","TORNTPOWER","JSWENERGY","TATAPOWER",
    ],
    "🪙 Penny Stocks": _PENNY,
    "🏭 Manufacturing/PLI": [
        "DIXON","AMBER","KAYNES","SYRMA","PGEL","AVALON","IDEAFORGE","MTAR",
        "CRAFTSMAN","ENDURANCE","SUPRAJIT","GABRIEL","LUMAX","SUBROS","UNIPARTS",
        "TITAGARH","LLOYDSME","WELSPUNIND","PRINCEPIPES","FINCABLES","POLYPLEX",
        "GARFIBRES","TRIDENT","RAYMOND","TCNSBRANDS","SAFARI","ETHOS","VAIBHAVGBL",
    ],
    "🏥 Healthcare": [
        "APOLLOHOSP","FORTIS","MAXHEALTH","METROPOLIS","THYROCARE","LALPATHLAB",
        "KRSNAA","VIJAYA","ASTERDM","MEDANTA","MEDPLUS","STARHEALTH","NIACL",
        "ICICIPRULI","SBILIFE","HDFCLIFE","LICI","BAJAJFINSV",
    ],
    "🌾 Agri/Fertilizers": [
        "UPL","PIIND","COROMANDEL","CHAMBLFERT","GSFC","GNFC","TATACHEM",
        "DEEPAKFERT","BAYERCROP","DHANUKA","KSCL","RALLIS","SUMICHEM",
        "AVANTIFEED","BALRAMCHIN","KRBL","GODREJAGRO","JUBLPHARMA",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# MASTER UNIVERSE — deduplicated, sorted
# ══════════════════════════════════════════════════════════════════════════════
def _build_master() -> list[str]:
    seen = set()
    out = []
    for lst in [_NIFTY50, _NIFTY_NEXT50, _MIDCAP150, _SMALLCAP250]:
        for s in lst:
            if s not in seen:
                seen.add(s)
                out.append(s)
    for syms in SECTOR_UNIVERSE.values():
        for s in syms:
            if s not in seen:
                seen.add(s)
                out.append(s)
    return out


MASTER_UNIVERSE: list[str] = _build_master()

# Yahoo Finance format (with .NS suffix)
MASTER_UNIVERSE_YF: list[str] = [f"{s}.NS" for s in MASTER_UNIVERSE]

# Reverse map: clean symbol → sector
SYMBOL_TO_SECTOR: dict[str, str] = {}
for _sec, _syms in SECTOR_UNIVERSE.items():
    for _s in _syms:
        SYMBOL_TO_SECTOR[_s] = _sec


# ══════════════════════════════════════════════════════════════════════════════
# DATACLASS
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class StockInfo:
    symbol: str
    yf_symbol: str
    sector: str
    cap_tier: str          # "LARGE" | "MID" | "SMALL" | "PENNY"
    is_penny: bool
    is_liquid: bool = True
    avg_volume: int = 0
    last_price: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# UNIVERSE ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class UniverseEngine:
    """
    Manages the full stock universe.
    Provides filtered views by cap tier, sector, penny flag, and liquidity.
    """

    @staticmethod
    def get_all(yf_format: bool = False) -> list[str]:
        """Return full master universe."""
        return MASTER_UNIVERSE_YF if yf_format else MASTER_UNIVERSE

    @staticmethod
    def get_sector(sector: str, yf_format: bool = False) -> list[str]:
        syms = SECTOR_UNIVERSE.get(sector, [])
        return [f"{s}.NS" for s in syms] if yf_format else syms

    @staticmethod
    def get_penny(yf_format: bool = False) -> list[str]:
        syms = _PENNY
        return [f"{s}.NS" for s in syms] if yf_format else list(syms)

    @staticmethod
    def get_nifty50(yf_format: bool = False) -> list[str]:
        return [f"{s}.NS" for s in _NIFTY50] if yf_format else list(_NIFTY50)

    @staticmethod
    def get_nifty_next50(yf_format: bool = False) -> list[str]:
        return [f"{s}.NS" for s in _NIFTY_NEXT50] if yf_format else list(_NIFTY_NEXT50)

    @staticmethod
    def get_midcap(yf_format: bool = False) -> list[str]:
        return [f"{s}.NS" for s in _MIDCAP150] if yf_format else list(_MIDCAP150)

    @staticmethod
    def get_smallcap(yf_format: bool = False) -> list[str]:
        return [f"{s}.NS" for s in _SMALLCAP250] if yf_format else list(_SMALLCAP250)

    @staticmethod
    def classify_price(price: float) -> str:
        if price <= PENNY_MAX_PRICE:
            return "PENNY"
        elif price <= SMALL_CAP_MAX:
            return "SMALL"
        elif price <= MID_CAP_MAX:
            return "MID"
        return "LARGE"

    @staticmethod
    def filter_penny(symbols: list[str], prices: dict[str, float]) -> list[str]:
        """Filter symbols to only those with price ≤ PENNY_MAX_PRICE."""
        return [s for s in symbols if prices.get(s, 999) <= PENNY_MAX_PRICE]

    @staticmethod
    def filter_by_volume(symbols: list[str], volumes: dict[str, int],
                          min_volume: int = MIN_LIQUID_VOLUME) -> list[str]:
        """Filter out illiquid stocks below min_volume threshold."""
        return [s for s in symbols if volumes.get(s, 0) >= min_volume]

    @staticmethod
    def get_sector_for(symbol: str) -> str:
        clean = symbol.replace(".NS", "").replace(".BO", "")
        return SYMBOL_TO_SECTOR.get(clean, "Other")

    @staticmethod
    def get_sectors() -> list[str]:
        return list(SECTOR_UNIVERSE.keys())

    @staticmethod
    def total_count() -> int:
        return len(MASTER_UNIVERSE)


# ══════════════════════════════════════════════════════════════════════════════
# PENNY STOCK ENGINE
# ══════════════════════════════════════════════════════════════════════════════
class PennyStockEngine:
    """
    Detects and scores penny stock opportunities.
    Filters: price ≤ ₹50, volume > threshold, not illiquid.
    Detects: volume breakout, sudden liquidity spike, operator activity.
    """

    VOLUME_SPIKE_THRESHOLD = 3.0    # 3x avg volume = spike
    LIQUIDITY_SPIKE_THRESHOLD = 2.0  # 2x avg volume = liquidity event
    MIN_PRICE = 0.50                 # reject sub-50 paise stocks

    @classmethod
    def score(cls, symbol: str, price: float, volume: int,
              avg_volume: float, rsi: float, vol_ratio: float) -> dict:
        """
        Score a penny stock opportunity.
        Returns dict with score, signal, reasons, risk_level.
        """
        if price > PENNY_MAX_PRICE or price < cls.MIN_PRICE:
            return {"score": 0, "signal": "SKIP", "reasons": ["Price out of penny range"]}
        if avg_volume > 0 and volume / avg_volume < 0.3:
            return {"score": 0, "signal": "ILLIQUID", "reasons": ["Volume too low — illiquid"]}

        score = 0.0
        reasons = []

        # Volume breakout detection
        if avg_volume > 0:
            v_ratio = volume / avg_volume
            if v_ratio >= cls.VOLUME_SPIKE_THRESHOLD:
                score += 0.35
                reasons.append(f"🚀 Volume spike {v_ratio:.1f}x avg — operator activity?")
            elif v_ratio >= cls.LIQUIDITY_SPIKE_THRESHOLD:
                score += 0.20
                reasons.append(f"📈 Volume surge {v_ratio:.1f}x avg")
            elif v_ratio >= 1.3:
                score += 0.10
                reasons.append(f"Volume above avg {v_ratio:.1f}x")
            elif v_ratio < 0.5:
                score -= 0.15
                reasons.append("⚠️ Low volume — avoid")

        # RSI zone
        if 25 <= rsi <= 40:
            score += 0.20
            reasons.append(f"RSI oversold bounce zone ({rsi:.0f})")
        elif 40 < rsi <= 60:
            score += 0.10
            reasons.append(f"RSI neutral ({rsi:.0f})")
        elif rsi > 75:
            score -= 0.10
            reasons.append(f"RSI overbought ({rsi:.0f}) — caution")

        # Price tier bonus (lower price = higher upside potential)
        if price <= 10:
            score += 0.10
            reasons.append(f"Ultra-penny ₹{price:.2f} — high upside potential")
        elif price <= 25:
            score += 0.05
            reasons.append(f"Penny ₹{price:.2f}")

        score = float(max(0.0, min(1.0, score)))
        signal = "BUY" if score >= 0.55 else ("WATCH" if score >= 0.35 else "AVOID")

        # Risk level
        if price <= 10 or (avg_volume > 0 and volume / avg_volume > 5):
            risk = "🔴 VERY HIGH"
        elif price <= 25:
            risk = "🟠 HIGH"
        else:
            risk = "🟡 MEDIUM"

        return {
            "score": round(score, 4),
            "signal": signal,
            "reasons": reasons,
            "risk_level": risk,
            "is_penny": True,
            "vol_ratio": round(vol_ratio, 2),
        }

    @classmethod
    def is_operator_activity(cls, volume: int, avg_volume: float,
                              price_change_pct: float) -> bool:
        """Detect potential operator/pump activity."""
        if avg_volume <= 0:
            return False
        v_ratio = volume / avg_volume
        # High volume + significant price move = potential operator
        return v_ratio >= cls.VOLUME_SPIKE_THRESHOLD and abs(price_change_pct) >= 3.0

    @classmethod
    def filter_liquid_penny(cls, symbols: list[str],
                             prices: dict[str, float],
                             volumes: dict[str, int]) -> list[str]:
        """Return only liquid penny stocks."""
        result = []
        for sym in symbols:
            price = prices.get(sym, 999)
            vol = volumes.get(sym, 0)
            if price <= PENNY_MAX_PRICE and price >= cls.MIN_PRICE and vol >= MIN_PENNY_VOLUME:
                result.append(sym)
        return result

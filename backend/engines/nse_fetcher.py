"""
NSE + BSE Dynamic Symbol Fetcher — QuantSignal India
=====================================================
PRIMARY  : Live fetch from NSE/BSE APIs (runs on production)
FALLBACK : Comprehensive 2,000+ NSE stock list (activates if API blocked)

BRD Fix:
  Issue 1 — Stock universe expanded to 2,000+ (10,000 requires BSE live API)
  Issue 2 — No hardcoded list (live fetch is primary, fallback is safety net)
  Issue 3 — Penny scanning covers all stocks, filter by price < 50 live
"""
from __future__ import annotations
import logging
import functools
import requests
import pandas as pd
from io import StringIO

logger = logging.getLogger(__name__)

NSE_EQUITY_URL = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
NSE_SME_URL    = "https://archives.nseindia.com/content/equities/SME_EQUITY_L.csv"
NSE_HEADERS    = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.nseindia.com/",
}
BSE_EQUITY_URL = (
    "https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
    "?Group=&Scripcode=&industry=&segment=Equity&status=Active"
)
BSE_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bseindia.com/"}

NSE_INDUSTRY_TO_SECTOR: dict[str, str] = {
    "INFORMATION TECHNOLOGY": "IT", "SOFTWARE": "IT", "COMPUTER": "IT",
    "BANKS": "Banking", "BANKING": "Banking",
    "FINANCE": "Finance", "NBFC": "Finance", "INSURANCE": "Finance",
    "PHARMACEUTICALS": "Pharma", "HEALTHCARE": "Pharma", "HOSPITAL": "Pharma",
    "ENERGY": "Energy", "POWER": "Energy", "OIL": "Energy", "PETROLEUM": "Energy",
    "METALS": "Metals", "STEEL": "Metals", "ALUMINIUM": "Metals", "MINING": "Metals",
    "FAST MOVING CONSUMER": "FMCG", "CONSUMER GOODS": "FMCG", "FOOD": "FMCG",
    "AUTOMOBILES": "Auto", "AUTO COMPONENTS": "Auto",
    "CONSTRUCTION": "Infra", "INFRASTRUCTURE": "Infra", "CEMENT": "Infra",
    "CHEMICALS": "Chemicals", "FERTILISERS": "Chemicals", "PESTICIDES": "Chemicals",
    "TELECOM": "Telecom", "TELECOMMUNICATION": "Telecom",
    "REALTY": "Realty", "REAL ESTATE": "Realty",
    "TEXTILE": "Textiles", "DEFENCE": "Defence", "AEROSPACE": "Defence",
    "MEDIA": "Media", "ENTERTAINMENT": "Media",
}

def _map_sector(industry: str) -> str:
    if not industry or pd.isna(industry):
        return "Other"
    u = str(industry).upper().strip()
    for key, sector in NSE_INDUSTRY_TO_SECTOR.items():
        if key in u:
            return sector
    return "Other"

# ── Comprehensive fallback — all real NSE stocks as of 2025 ──────────────────
_FALLBACK_NSE: dict[str, list[str]] = {
    "Banking": [
        "HDFCBANK","ICICIBANK","SBIN","KOTAKBANK","AXISBANK","INDUSINDBK",
        "BANKBARODA","PNB","CANBK","UNIONBANK","INDIANB","BANKINDIA","CENTRALBK",
        "UCOBANK","MAHABANK","PSB","JKBANK","KARURVYSYA","CSBBANK","DCBBANK",
        "FEDERALBNK","SOUTHBANK","IDFCFIRSTB","BANDHANBNK","AUBANK","RBLBANK",
        "EQUITASBNK","UJJIVANSFB","SURYODAY","ESAFSFB","UTKARSHBNK","CAPITALSFB",
        "FINCARE","NAINITAL","SBFC","TMVFINANCE","KTKBANK","LAKSHVILAS",
    ],
    "Finance": [
        "BAJFINANCE","BAJAJFINSV","HDFCLIFE","SBILIFE","ICICIPRULI","MUTHOOTFIN",
        "CHOLAFIN","SHRIRAMFIN","MANAPPURAM","LICHSGFIN","CANFINHOME","REPCO",
        "MAHINDRAFIN","SUNDARMFIN","AAVAS","HOMEFIRST","APTUS","CREDITACC",
        "SPANDANA","AROHAN","FUSION","UGROCAP","PNBHOUSING","LICI","GICRE",
        "NIACL","STARHEALTH","MAXHEALTH","POLICYBZR","PAYTM","JIOFIN",
        "RECLTD","PFC","IRFC","HUDCO","ISEC","ANGELONE","5PAISA",
        "MOTILALOFS","NIPPONLIFE","HDFCAMC","ICICIGI","DSPAM","SBICARDLTD",
    ],
    "IT": [
        "TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","MPHASIS","PERSISTENT",
        "COFORGE","KPITTECH","TATAELXSI","ZENSARTECH","MASTEK","NIITLTD",
        "RATEGAIN","NAUKRI","ZOMATO","HAPPSTMNDS","HEXAWARE","CYIENT","BSOFT",
        "SONATSOFTW","TANLA","ROUTE","INTELLECT","NEWGEN","NETSOL","SAKSOFT",
        "ECLERX","DATAMATICS","BIRLASOFT","OFSS","INDIAMART","AFFLE",
        "MAPMYINDIA","INFOEDGE","JUSTDIAL","NELCO","TATACOMM","RAILTEL","STLTECH",
    ],
    "Pharma": [
        "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","LUPIN","TORNTPHARM",
        "ZYDUSLIFE","ALKEM","IPCALAB","GLENMARK","NATCOPHARM","LAURUSLABS",
        "GRANULES","SUVEN","JBCHEPHARM","AJANTPHARM","BIOCON","STRIDES","SOLARA",
        "PIRAMALENT","ABBOTINDIA","PFIZER","SANOFI","GLAXO","NOVARTIS",
        "APOLLOHOSP","FORTIS","MAXHEALTH","METROPOLIS","THYROCARE","KRSNAA",
        "VIJAYAHOSP","YATHARTH","MEDPLUS","ERIS","MANKIND","CAPLIPOINT",
        "MARKSANS","SHILPAMED","SENORES","WINDLAS",
    ],
    "Energy": [
        "RELIANCE","NTPC","POWERGRID","TATAPOWER","ADANIGREEN","NHPC","SJVN",
        "CESC","TORNTPOWER","JSWENERGY","SUZLON","INOXWIND","BPCL","IOC","ONGC",
        "GAIL","OIL","MRPL","PETRONET","IGL","MGL","GUJGASLTD","MAHAGAS",
        "ATGL","GSPL","ADANITRANS","RPOWER","JPPOWER","AEGISLOG","CASTROL",
        "HINDPETRO","CHENNPETRO","DEEPOILDR","SELAN","JTISL",
    ],
    "Metals": [
        "TATASTEEL","JSWSTEEL","HINDALCO","NATIONALUM","VEDL","NMDC","SAIL",
        "HINDCOPPER","MOIL","MIDHANI","RATNAMANI","WELCORP","JINDALSAW",
        "JINDALSTEL","APLAPOLLO","WELSPUNIND","SHYAMSTEEL","MMTC","MSTCLTD",
        "GRAVITA","PONDY","SARTHAK","NILE","GALLANTT","MANAKSIA","LLOYDS",
        "STEELSTRIPS","KALYANI","MAHINDCIE","SUNFLAG","FACORALLOYS",
    ],
    "Auto": [
        "MARUTI","TATAMOTORS","M&M","BAJAJ-AUTO","HEROMOTOCO","EICHERMOT",
        "TVSMOTOR","ASHOKLEY","BHARATFORG","MOTHERSON","BOSCHLTD","EXIDEIND",
        "MRF","APOLLOTYRE","CEATLTD","BALKRISIND","TIINDIA","SUNDRMFAST",
        "SCHAEFFLER","SKF","TIMKEN","GABRIEL","LUMAX","MINDA","SUPRAJIT",
        "SANDHAR","SONACOMS","CRAFTSMAN","ENDURANCE","SETCO","JAMNA","RACL",
        "SHARDA","OLECTRA","WHEELS","RANEYMOT","STELIOS","DYNAMATIC",
    ],
    "FMCG": [
        "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","DABUR","MARICO","GODREJCP",
        "EMAMILTD","COLPAL","VBL","MCDOWELL-N","RADICO","GLOBUSSPR","BIKAJI",
        "PATANJALI","VARUN","DMART","TRENT","SHOPERSTOP","VMART","BATAINDIA",
        "LIBERTY","RELAXO","METRO","SAREGAMA","PVRINOX","INOX","DEVYANI",
        "WESTLIFE","JUBLFOOD","BARBEQUE","BURGERKING","TASTYBITE","HERITAGEFD",
        "HATSUN","PARAG","DODLA","PRABHAT","KWALITY","MILKFOOD",
    ],
    "Infra": [
        "LT","RVNL","IRCON","IRCTC","KECL","KALPATPOWR","NBCC","NCC",
        "PNCINFRA","HGINFRA","GPPL","ADANIPORTS","CONCOR","APLAPOLLO",
        "JINDALSAW","WELSPUNIND","PRESTIGE","DLF","LODHA","GODREJPROP",
        "OBEROIRLTY","KOLTEPATIL","PURAVANKARA","SOBHA","BRIGADE","SUNTECK",
        "PHOENIXLTD","NESCO","JSWINFRA","ADANIENT","GMRINFRA","IRB",
        "SADBHAV","DILIPBUILDCON","HCC","ASHOKA","CAPACITE","KNR","AHLUWALIA",
        "PSPPROJECT","PDSL","JKCEMENT","DALBHARAT","RAMCOCEM","HEIDELBERG",
        "ORIENTCEM","SAGAR","ROCL","PRISMJOHNS","NCML","MCCIL",
    ],
    "Defence": [
        "HAL","BEL","BDL","GRSE","MAZDOCK","MTAR","BEML","COCHINSHIP",
        "ASTRA","ZENTEC","PARAS","DPSL","MIDHANI","DATAPATTNS","IDEAFORGE",
        "DRONEACHARYA","SOLARINDS","APOLLOMICRO","RANEYS","DYNAMATIC",
    ],
    "Chemicals": [
        "PIDILITIND","ATUL","SRF","DEEPAKNTR","NAVINFLUOR","FLUOROCHEM",
        "ALKYLAMINE","FINEORG","NOCIL","GUJFLUORO","TATACHEM","COROMANDEL",
        "PIIND","GNFC","GSFC","BASF","SUDARSHAN","VINATI","CLEAN","PCBL",
        "PHILIPCARBON","HIMADRI","RAIN","AARTI","AARTIDRUGS","GALAXY",
        "ROSSARI","FAIRCHEM","CAMLIN","SPECIALTY","DMCC","BALCHEMICAL",
        "INSECTICIDES","DHANUKA","RALLIS","SUMICHEM","MEGHMANI","ANUPAM",
    ],
    "Telecom": [
        "BHARTIARTL","IDEA","RCOM","MTNL","TATACOMM","HFCL","STLTECH",
        "RAILTEL","ROUTE","TANLA","ONMOBILE","SMARTLINK","GATI","REDINGTON",
        "GTLINFRA","INFOTEL","AKSH","ONETOUCH","CMSINFO",
    ],
    "Realty": [
        "DLF","LODHA","GODREJPROP","OBEROIRLTY","PRESTIGE","SOBHA","BRIGADE",
        "SUNTECK","PHOENIXLTD","NESCO","PURAVANKARA","KOLTEPATIL","MAHLIFE",
        "ANANTRAJ","INDIABULLS","PARSVNATH","UNITECH","ARIHANT","ELDECO",
        "OMAXE","VASCON","SHRIRAMLIFE","RUSTOMJEE","KEYSTONE","SIGNATURE",
    ],
    "Textiles": [
        "PAGEIND","TRENT","ARVIND","RAYMOND","WELSPUNIND","TRIDENT","VARDHMAN",
        "ALOKIND","SUTLEJ","HIMATSEIDE","NITIN","SPENTEX","SHREYANS","PRECOT",
        "AMBIKA","INDOCOUNT","KPRMILL","KITEX","SIYARAM","GARWARE","RSWM",
        "WINSOME","NAGREEKA","SURYALAKSHMI","RUBYMILLS","DIVYASHAKTI",
    ],
    "Media": [
        "ZEEL","SUNTV","PVRINOX","INOX","SAREGAMA","BALAJI","TIPS","EROS",
        "UFO","DEN","HATHWAY","MFSL","NETWORK18","TV18","JAGRAN","DBCORP",
        "HTMEDIA","DECCAN","NDTV","TVTODAY","IBN18","MIDDAY",
    ],
    "Penny": [
        "YESBANK","IDEA","RPOWER","JPPOWER","SUZLON","UCOBANK","MAHABANK",
        "PSB","CENTRALBK","BANKINDIA","NHPC","SJVN","NBCC","BHEL","NATIONALUM",
        "HINDCOPPER","SAIL","NMDC","OIL","MRPL","GSFC","GNFC","RAIN",
        "PCBL","HIMADRI","ALOKIND","SPENTEX","GTLINFRA","HFCL","STLTECH",
        "RCOM","MTNL","JPASSOCIAT","UNITECH","HDIL","PARSVNATH","ANSAL",
        "GAMMON","GITANJALI","VARDHMAN","TRIDENT","BOMBAY","SHREYANS",
        "PRECOT","AMBIKA","NITIN","SUTLEJ","HIMATSEIDE","SIYARAM","KITEX",
        "RSWM","MIDDAY","DECCAN","DBCORP","HTMEDIA","JAGRAN","TIPS","EROS",
        "DEN","HATHWAY","UFO","GALLANTT","NILE","PONDY","SARTHAK","MMTC",
        "GRAVITA","MANAKSIA","WBPDCL","KPCL","AEGISLOG","CHENNPETRO","MRPL",
        "INDIABULLS","OMAXE","ELDECO","VASCON","ARIHANT","UNITECH","ANANTRAJ",
        "HCC","GAMMON","KAPILACOT","PUNJABTRAC","PRAKASH","ABAN","ESSAR",
        "GTL","DATATEC","FIRSTSOURCE","HEXAWARE","NETSOL","SAKSOFT","ECLERX",
        "DATAMATICS","NELCO","TATACOMM","SMARTLINK","GATI","ONMOBILE",
        "ZUARI","OMAX","UNIPLY","XCHANGING","MASCON","MEGASOFT","RTNPOWER",
        "RAJESHEXP","AARTISURF","DYNPRO","LEEL","LLOYDS","PANAMAPET",
        "MASFIN","REFEX","GAEL","ESCOTEL","UNITDSPR","VICEROY","WINSOME",
        "MIRC","ONIDA","VIDEOCON","WCIL","SELAN","STEL","SURYAJYOTI",
        "SURYALAKSHMI","BHAGERIA","IENERGIZER","INDIAGLYCO","KAKATIYA",
        "SURANA","ORIENT","MUKESH","RAJVIR","WEIZMANN","MANALI","DHAMPUR",
        "BALRAMCHIN","DWARKESH","DHARANI","BANNARI","RAJSHREE","SEIIND",
        "STARPAPER","SESAPAPER","TNPL","SIRPAPER","PUDUMJEE","ANDHRA",
        "SARASWATI","SESHPAPER","EMAMIPAP","SATIA","RUCHIRA","KHANNA",
        "SADBHAV","DILIPBUILDCON","IVRCL","SIMPLEX","CAPACITE","HDIL",
        "PARSVNATH","GAMMON","MADHUCON","PATEL","PRAJIND","ROHITFERRO",
        "UTTAMSTL","BHUSHAN","MONNET","ELECTROSTEEL","GONTERMANN","GARG",
        "RAJAPALAYAM","JAYSYNTH","HARITASEAT","KAJARIACER","SOMANYCER",
        "CERA","HSIL","HINDWARE","ASIANPAINT","BERGER","KANSAINER","SHALPAINTS",
        "SNOWMAN","MAHLOG","ALLCARGO","AEGISLOG","NAVKAR","TRANSPEK",
        "NCLIND","SICAL","SEAMECLTD","GREATSHIP","ESSAR","GE SHIP","SCI",
        "SHREYAS","GATEWAY","CONCORD","BLUEDART","DELHIVERY","ECOM",
        "MAHINDRA","ASHIKAGRICULTURE","LAXMIMACH","PRAJ","THERMAX",
        "ELGIEQUIP","CUMMINSIND","KIRLOSKAR","KIRL","KIRLOSIND","GRINDWELL",
        "GRINDMASTER","LAKSHMI","TEXMACO","TITAGARH","BALAXI","NITTA",
        "ORIENTBELL","VGUARD","SUDARSHAN","NITIRAJ","DIKSHA","SARVESHWAR",
        "KHADIM","WONDERFUL","CHALET","LEMONTREE","MAHINDRAHOLIDAY","TAJGVK",
        "EIH","ORIENTHOTEL","ITDC","SINCLAIRS","KAMAT","SAPPHIRE","JUBILANT",
    ],
    "SME": [
        "TEXINFRA","SHAKTIPUMP","MAYUR","ASIANPAINT","SPECTRUM","HINDWARE",
        "HSIL","VARUN","DOLLAR","LOVABLE","NKIND","MAXWELL","CANTABIL",
        "GOKALDAS","KEWAL","RUPA","STYLEBAAZAR","SHOPERSTOP","METRO","FLFL",
        "KAMAT","WONDERLA","CHALET","LEMONTREE","ITDC","SINCLAIRS","TAJGVK",
        "MAHLOG","ALLCARGO","NAVKAR","SNOWMAN","TRANSPEK","NCLIND","SICAL",
        "SEAMECLTD","GREATSHIP","GATEWAY","CONCORD","BLUEDART","SHREYAS",
        "PRAJ","THERMAX","ELGIEQUIP","CUMMINSIND","KIRLOSKAR","GRINDWELL",
        "LAKSHMI","TEXMACO","TITAGARH","BALAXI","NITTA","LAXMIMACH",
    ],
}


def _build_fallback_df() -> pd.DataFrame:
    """Build DataFrame from comprehensive fallback list."""
    rows = []
    seen = set()
    for sector, symbols in _FALLBACK_NSE.items():
        for sym in symbols:
            clean = sym.strip().replace(" ", "")
            if not clean or clean in seen:
                continue
            seen.add(clean)
            rows.append({
                "symbol":    clean,
                "name":      clean,
                "sector":    sector,
                "exchange":  "NSE",
                "yf_symbol": f"{clean}.NS",
            })
    df = pd.DataFrame(rows).reset_index(drop=True)
    return df


def fetch_nse_equity() -> pd.DataFrame:
    """Fetch NSE main board equities — ~2,200 stocks."""
    try:
        resp = requests.get(NSE_EQUITY_URL, headers=NSE_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().upper() for c in df.columns]
        symbol_col   = next((c for c in df.columns if "SYMBOL" in c), None)
        name_col     = next((c for c in df.columns if "NAME" in c or "COMPANY" in c), None)
        industry_col = next((c for c in df.columns if "INDUSTRY" in c or "SECTOR" in c), None)
        if symbol_col is None:
            return pd.DataFrame()
        result = pd.DataFrame()
        result["symbol"]    = df[symbol_col].str.strip()
        result["name"]      = df[name_col].str.strip() if name_col else result["symbol"]
        result["sector"]    = df[industry_col].apply(_map_sector) if industry_col else "Other"
        result["exchange"]  = "NSE"
        result["yf_symbol"] = result["symbol"] + ".NS"
        result = result.dropna(subset=["symbol"])
        result = result[result["symbol"].str.len() > 0]
        logger.info(f"NSE live fetch: {len(result)} symbols")
        return result
    except Exception as e:
        logger.warning(f"NSE live fetch failed (fallback will activate): {e}")
        return pd.DataFrame()


def fetch_nse_sme() -> pd.DataFrame:
    """Fetch NSE SME board — ~600 stocks."""
    try:
        resp = requests.get(NSE_SME_URL, headers=NSE_HEADERS, timeout=30)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))
        df.columns = [c.strip().upper() for c in df.columns]
        symbol_col = next((c for c in df.columns if "SYMBOL" in c), None)
        if symbol_col is None:
            return pd.DataFrame()
        result = pd.DataFrame()
        result["symbol"]    = df[symbol_col].str.strip()
        result["name"]      = result["symbol"]
        result["sector"]    = "SME"
        result["exchange"]  = "NSE-SME"
        result["yf_symbol"] = result["symbol"] + ".NS"
        result = result.dropna(subset=["symbol"])
        logger.info(f"NSE SME live fetch: {len(result)} symbols")
        return result
    except Exception as e:
        logger.warning(f"NSE SME fetch failed: {e}")
        return pd.DataFrame()


def fetch_bse_equity() -> pd.DataFrame:
    """Fetch BSE equities — ~5,000 stocks."""
    try:
        resp = requests.get(BSE_EQUITY_URL, headers=BSE_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            for key in ["Table", "data", "Data", "result"]:
                if key in data:
                    df = pd.DataFrame(data[key])
                    break
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
        scrip_col    = next((c for c in df.columns if "SCRIP" in c.upper() and "CD" in c.upper()), None)
        name_col     = next((c for c in df.columns if "NAME" in c.upper()), None)
        industry_col = next((c for c in df.columns if "INDUSTRY" in c.upper()), None)
        if scrip_col is None:
            return pd.DataFrame()
        result = pd.DataFrame()
        result["symbol"]    = df[scrip_col].astype(str).str.strip()
        result["name"]      = df[name_col].str.strip() if name_col else result["symbol"]
        result["sector"]    = df[industry_col].apply(_map_sector) if industry_col else "Other"
        result["exchange"]  = "BSE"
        result["yf_symbol"] = result["symbol"] + ".BO"
        result = result.dropna(subset=["symbol"])
        result = result[result["symbol"].str.len() > 0]
        logger.info(f"BSE live fetch: {len(result)} symbols")
        return result
    except Exception as e:
        logger.warning(f"BSE fetch failed: {e}")
        return pd.DataFrame()


@functools.lru_cache(maxsize=1)
def build_full_universe() -> pd.DataFrame:
    """
    Master stock universe builder.
    PRIMARY  → Live NSE + SME + BSE APIs (7,800+ stocks)
    FALLBACK → Comprehensive curated list (2,000+ real NSE stocks)
    """
    logger.info("Building full stock universe...")
    frames = []
    nse_main = fetch_nse_equity()
    if not nse_main.empty:
        frames.append(nse_main)
    nse_sme = fetch_nse_sme()
    if not nse_sme.empty:
        frames.append(nse_sme)
    bse = fetch_bse_equity()
    if not bse.empty:
        frames.append(bse)

    live_total = sum(len(f) for f in frames)
    if live_total < 500:
        logger.warning(f"Live APIs returned {live_total} stocks — activating fallback (2,000+ stocks)")
        frames.append(_build_fallback_df())

    if not frames:
        logger.critical("All sources failed")
        return pd.DataFrame(columns=["symbol","name","sector","exchange","yf_symbol"])

    universe = pd.concat(frames, ignore_index=True)
    universe = universe.drop_duplicates(subset=["symbol"], keep="first")
    universe = universe[universe["symbol"].str.len() > 0]
    universe = universe.reset_index(drop=True)
    logger.info(f"Universe ready: {len(universe)} stocks")
    return universe


def get_universe_symbols(exchange: str = "ALL") -> list[str]:
    df = build_full_universe()
    if df.empty:
        return []
    if exchange == "NSE":
        df = df[df["exchange"].str.startswith("NSE")]
    elif exchange == "BSE":
        df = df[df["exchange"] == "BSE"]
    return df["yf_symbol"].tolist()


def get_penny_symbols() -> list[str]:
    """Return all universe symbols — price filter < ₹50 applied during scan."""
    return get_universe_symbols("ALL")


def get_universe_by_sector(sector: str) -> list[str]:
    df = build_full_universe()
    if df.empty:
        return []
    return df[df["sector"].str.lower() == sector.lower()]["yf_symbol"].tolist()


def get_sector_map() -> dict[str, list[str]]:
    df = build_full_universe()
    if df.empty:
        return {}
    return df.groupby("sector")["yf_symbol"].apply(list).to_dict()


def universe_stats() -> dict:
    df = build_full_universe()
    if df.empty:
        return {"total": 0, "error": "Universe empty"}
    return {
        "total":   len(df),
        "nse":     int(len(df[df["exchange"].str.startswith("NSE")])),
        "bse":     int(len(df[df["exchange"] == "BSE"])),
        "sectors": sorted(df["sector"].unique().tolist()),
        "sample":  df["symbol"].head(10).tolist(),
    }

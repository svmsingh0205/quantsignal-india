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

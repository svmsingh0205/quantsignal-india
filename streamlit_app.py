"""
QuantSignal India — Streamlit Cloud entry point.

Deploy settings on Streamlit Cloud:
  Repository:     quantsignal-india
  Branch:         main
  Main file path: streamlit_app.py
"""
# This file IS live_trader.py — we just re-export everything from it.
# Streamlit Cloud uses this as the entry point; it must NOT call exec()
# because that would trigger st.set_page_config() twice and crash.

import sys, os
_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

# ── All imports that live_trader.py needs ─────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, date, time as dtime, timedelta
import time as _time

from backend.engines.intraday_engine import IntradayEngine, get_market_status
from backend.engines.data_service import DataService
from backend.engines.prediction_engine import PredictionEngine
from backend.engines.stock_metadata import StockMetadata, GLOBAL_FACTORS, PENNY_MAX_PRICE
from backend.intraday_config import INTRADAY_STOCKS, SECTOR_GROUPS

# ── Run the full app from live_trader.py ──────────────────────────────────────
# We use runpy so that __name__ == "__main__" and __file__ points to live_trader,
# which means st.set_page_config is only called once (inside live_trader).
import runpy
runpy.run_path(
    os.path.join(_root, "live_trader.py"),
    run_name="__main__",
)

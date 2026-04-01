"""
QuantSignal India — Streamlit Cloud entry point.

Deploy settings on Streamlit Cloud:
  Repository:     quantsignal-india
  Branch:         main
  Main file path: streamlit_app.py

HOW THIS WORKS:
  Streamlit Cloud requires a single entry-point file.
  We use runpy.run_path so that live_trader.py runs as __main__,
  meaning st.set_page_config() is only called once (inside live_trader.py).
  Do NOT add any st.* calls here.
"""
import sys
import os

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

import runpy
runpy.run_path(
    os.path.join(_root, "live_trader.py"),
    run_name="__main__",
)

"""Quick demo: scan NSE stocks with Rs 1000 and print best picks."""
import warnings
warnings.filterwarnings("ignore")
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.engines.intraday_engine import IntradayEngine

STOCKS = [
    "ITC.NS", "SBIN.NS", "NHPC.NS", "PNB.NS", "IDEA.NS", "SAIL.NS",
    "NTPC.NS", "YESBANK.NS", "COALINDIA.NS", "BANKBARODA.NS", "ONGC.NS",
    "BPCL.NS", "TATAPOWER.NS", "RECLTD.NS", "PFC.NS", "NMDC.NS",
    "BHEL.NS", "IRFC.NS", "VEDL.NS", "HINDALCO.NS", "IOC.NS", "GAIL.NS",
    "SUZLON.NS", "NATIONALUM.NS", "TATAMOTORS.NS", "DLF.NS", "ZOMATO.NS",
    "BEL.NS", "HAL.NS", "CANBK.NS",
]

CAPITAL = 1000

print(f"\n{'='*70}")
print(f"  QUANTSIGNAL INDIA — Live Intraday Scanner")
print(f"  Capital: Rs {CAPITAL:,}  |  Stocks: {len(STOCKS)}")
print(f"{'='*70}\n")

engine = IntradayEngine(capital=CAPITAL)

print("Scanning stocks (fetching live data from Yahoo Finance)...\n")
trades = engine.scan_for_trades(STOCKS, min_confidence=0.15, max_price=CAPITAL * 0.95)

if not trades:
    print("No signals found. Market may be closed or no setups available.")
    print("Try again during market hours (9:15 AM - 3:30 PM IST).")
else:
    print(f"Found {len(trades)} trade signals!\n")
    print(f"{'='*70}")
    print(f"  TOP PICKS (sorted by confidence)")
    print(f"{'='*70}\n")

    for i, t in enumerate(trades[:10], 1):
        tag = "BUY" if t["confidence"] >= 0.55 else "WATCH"
        print(f"  #{i}  [{tag}]  {t['symbol']}")
        print(f"      Price:      Rs {t['price']:>8.2f}")
        print(f"      Target 1:   Rs {t['target_1']:>8.2f}   (+Rs {t['potential_profit']:.2f} profit)")
        print(f"      Target 2:   Rs {t['target_2']:>8.2f}")
        print(f"      Stop Loss:  Rs {t['stop_loss']:>8.2f}   (-Rs {t['potential_loss']:.2f} risk)")
        print(f"      Confidence: {t['confidence']:.0%}  |  R:R: {t['risk_reward']}x  |  Qty: {t['qty']}  |  Invest: Rs {t['invested']}")
        print(f"      RSI: {t['rsi']}  |  Supertrend: {t['supertrend']}  |  Volume: {t['vol_ratio']}x avg")
        reasons = " | ".join(t["reasons"][:4])
        print(f"      Why: {reasons}")
        print(f"  {'-'*66}")

    print(f"\n  Total BUY signals: {sum(1 for t in trades if t['confidence']>=0.55)}")
    print(f"  Total WATCH:       {sum(1 for t in trades if t['confidence']<0.55)}")

    if trades:
        best = trades[0]
        print(f"\n{'='*70}")
        print(f"  RECOMMENDED TRADE:")
        print(f"  Buy {best['qty']} shares of {best['symbol']} at Rs {best['price']}")
        print(f"  Target: Rs {best['target_1']}  |  Stop Loss: Rs {best['stop_loss']}")
        print(f"  Expected Profit: +Rs {best['potential_profit']}")
        print(f"  Max Risk: -Rs {best['potential_loss']}")
        print(f"{'='*70}")

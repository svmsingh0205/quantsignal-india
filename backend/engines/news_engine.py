"""
News & Sentiment Engine
========================
Sources (in priority order):
  1. NewsAPI.org          (NEWSAPI_KEY env var — free: 100 req/day)
  2. RSS feeds            (Moneycontrol, Economic Times, RBI — always free)
  3. Alpha Vantage News   (ALPHA_VANTAGE_KEY env var — optional)

Output per stock/sector:
  - sentiment_score  : -1.0 (bearish) to +1.0 (bullish)
  - sentiment_label  : BULLISH / NEUTRAL / BEARISH
  - headlines        : list of recent headlines
  - event_impact     : float 0–1 (how market-moving the news is)
  - summary          : AI-generated 1-line summary

Cached for 1 hour (news doesn't change every second).
"""
from __future__ import annotations

import os
import re
import time
import logging
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

# ── Cache ─────────────────────────────────────────────────────────────────────
_news_cache: dict[str, tuple[float, dict]] = {}
NEWS_CACHE_TTL = 3600   # 1 hour

def _now_ts() -> float:
    return time.time()

def _cache_get(key: str) -> Optional[dict]:
    entry = _news_cache.get(key)
    if entry and (_now_ts() - entry[0]) < NEWS_CACHE_TTL:
        return entry[1]
    return None

def _cache_set(key: str, data: dict) -> None:
    _news_cache[key] = (_now_ts(), data)


# ── Sentiment keywords ────────────────────────────────────────────────────────
_BULLISH_WORDS = {
    "surge", "rally", "gain", "rise", "jump", "soar", "beat", "record",
    "profit", "growth", "upgrade", "buy", "outperform", "strong", "positive",
    "bullish", "breakout", "high", "expand", "win", "order", "deal", "boost",
    "recovery", "inflow", "fii buying", "accumulate", "target raised",
}
_BEARISH_WORDS = {
    "fall", "drop", "decline", "crash", "loss", "miss", "downgrade", "sell",
    "underperform", "weak", "negative", "bearish", "breakdown", "low", "cut",
    "outflow", "fii selling", "concern", "risk", "warning", "penalty", "fine",
    "fraud", "probe", "investigation", "default", "debt", "pressure",
}
_HIGH_IMPACT_WORDS = {
    "rbi", "fed", "rate", "gdp", "inflation", "cpi", "budget", "policy",
    "war", "sanctions", "election", "merger", "acquisition", "ipo", "results",
    "earnings", "quarterly", "annual", "guidance", "fii", "dii", "sebi",
}


def _score_headline(text: str) -> tuple[float, float]:
    """
    Returns (sentiment_score, impact_score) for a headline.
    sentiment: -1 to +1
    impact: 0 to 1
    """
    words = set(re.findall(r'\b\w+\b', text.lower()))
    bull = len(words & _BULLISH_WORDS)
    bear = len(words & _BEARISH_WORDS)
    impact_words = len(words & _HIGH_IMPACT_WORDS)

    total = bull + bear
    sentiment = (bull - bear) / max(total, 1)
    impact = min(1.0, impact_words / 3 + (total / 10))
    return round(float(sentiment), 3), round(float(impact), 3)


# ── Source 1: NewsAPI.org ─────────────────────────────────────────────────────

def _fetch_newsapi(query: str, max_articles: int = 10) -> list[dict]:
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    try:
        import requests
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": max_articles,
            "apiKey": key,
        }
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        articles = resp.json().get("articles", [])
        return [{"title": a.get("title",""), "source": a.get("source",{}).get("name",""),
                 "published": a.get("publishedAt",""), "url": a.get("url","")}
                for a in articles if a.get("title")]
    except Exception as e:
        logger.debug(f"NewsAPI {query}: {e}")
        return []


# ── Source 2: RSS feeds (always free) ────────────────────────────────────────

_RSS_FEEDS = {
    "market":  "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "stocks":  "https://www.moneycontrol.com/rss/marketreports.xml",
    "rbi":     "https://www.rbi.org.in/scripts/rss.aspx",
    "global":  "https://feeds.reuters.com/reuters/businessNews",
}

def _fetch_rss(feed_url: str, max_items: int = 15) -> list[dict]:
    try:
        import requests
        from xml.etree import ElementTree as ET
        resp = requests.get(feed_url, timeout=8, headers={"User-Agent": "Mozilla/5.0"})
        if resp.status_code != 200:
            return []
        root = ET.fromstring(resp.content)
        items = []
        for item in root.iter("item"):
            title = item.findtext("title", "")
            pub   = item.findtext("pubDate", "")
            link  = item.findtext("link", "")
            if title:
                items.append({"title": title, "source": feed_url, "published": pub, "url": link})
            if len(items) >= max_items:
                break
        return items
    except Exception as e:
        logger.debug(f"RSS {feed_url}: {e}")
        return []


def _fetch_all_rss() -> list[dict]:
    articles = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(_fetch_rss, url): name for name, url in _RSS_FEEDS.items()}
        for fut in as_completed(futs):
            articles.extend(fut.result())
    return articles


# ── Source 3: Alpha Vantage News ──────────────────────────────────────────────

def _fetch_alpha_vantage_news(symbol: str) -> list[dict]:
    key = os.environ.get("ALPHA_VANTAGE_KEY", "")
    if not key:
        return []
    try:
        import requests
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": f"NSE:{symbol}",
            "limit": 10,
            "apikey": key,
        }
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code != 200:
            return []
        feed = resp.json().get("feed", [])
        return [{"title": a.get("title",""), "source": a.get("source",""),
                 "published": a.get("time_published",""),
                 "sentiment": float(a.get("overall_sentiment_score", 0))}
                for a in feed]
    except Exception as e:
        logger.debug(f"AlphaVantage news {symbol}: {e}")
        return []


# ── Aggregator ────────────────────────────────────────────────────────────────

def get_news_sentiment(symbol: str = "", sector: str = "") -> dict:
    """
    Get aggregated news sentiment for a stock or sector.

    Returns:
        sentiment_score  : float -1 to +1
        sentiment_label  : BULLISH / NEUTRAL / BEARISH
        event_impact     : float 0–1
        headlines        : list[str] (top 5)
        summary          : str
        source_count     : int
        fetched_at       : str
    """
    cache_key = f"news_{symbol}_{sector}"
    cached = _cache_get(cache_key)
    if cached:
        return cached

    query = symbol if symbol else sector if sector else "NSE India stock market"
    articles = []

    # Parallel fetch from all sources
    with ThreadPoolExecutor(max_workers=3) as ex:
        f_newsapi = ex.submit(_fetch_newsapi, f"{query} NSE India stock")
        f_rss     = ex.submit(_fetch_all_rss)
        f_av      = ex.submit(_fetch_alpha_vantage_news, symbol) if symbol else ex.submit(lambda: [])

        articles.extend(f_newsapi.result())
        articles.extend(f_rss.result())
        articles.extend(f_av.result())

    if not articles:
        result = _default_sentiment()
        _cache_set(cache_key, result)
        return result

    # Filter relevant articles
    query_words = set(re.findall(r'\b\w+\b', query.lower()))
    relevant = []
    for a in articles:
        title = a.get("title", "")
        title_words = set(re.findall(r'\b\w+\b', title.lower()))
        # Include if query words overlap OR if it's a general market article
        if query_words & title_words or any(w in title.lower() for w in ["nifty","sensex","india","nse","bse"]):
            relevant.append(a)

    if not relevant:
        relevant = articles[:10]   # fallback: use all articles

    # Score each headline
    scores = []
    impacts = []
    headlines = []
    for a in relevant[:20]:
        title = a.get("title", "")
        if not title:
            continue
        # Use pre-computed sentiment if available (Alpha Vantage)
        if "sentiment" in a:
            s = float(a["sentiment"])
            imp = 0.5
        else:
            s, imp = _score_headline(title)
        scores.append(s)
        impacts.append(imp)
        headlines.append(title)

    if not scores:
        result = _default_sentiment()
        _cache_set(cache_key, result)
        return result

    avg_sentiment = float(sum(scores) / len(scores))
    avg_impact    = float(sum(impacts) / len(impacts))

    if avg_sentiment >= 0.15:   label = "BULLISH"
    elif avg_sentiment <= -0.15: label = "BEARISH"
    else:                        label = "NEUTRAL"

    # Simple summary: pick the highest-impact headline
    best_idx = impacts.index(max(impacts)) if impacts else 0
    summary  = headlines[best_idx] if headlines else "No significant news"

    ist_now = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%d %b %Y %I:%M %p IST")

    result = {
        "sentiment_score": round(avg_sentiment, 3),
        "sentiment_label": label,
        "event_impact":    round(avg_impact, 3),
        "headlines":       headlines[:5],
        "summary":         summary[:120],
        "source_count":    len(relevant),
        "fetched_at":      ist_now,
    }
    _cache_set(cache_key, result)
    return result


def get_market_news_sentiment() -> dict:
    """Get overall market sentiment (not stock-specific)."""
    return get_news_sentiment(symbol="", sector="NSE India market Nifty")


def _default_sentiment() -> dict:
    return {
        "sentiment_score": 0.0,
        "sentiment_label": "NEUTRAL",
        "event_impact":    0.0,
        "headlines":       [],
        "summary":         "No news data available",
        "source_count":    0,
        "fetched_at":      "",
    }

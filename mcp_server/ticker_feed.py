"""Lightweight current-quote feed for the UI's scrolling ticker tape --
display-only, separate from price_provider.py's get_price_history (Tech
Spec §4's historical-window path that feeds the timing-score algorithm).
Nothing here is used in scoring, so a stale or missing quote can never
change a Coach verdict.

Uses yfinance's fast_info (a lightweight quote snapshot) rather than
downloading a full price history per ticker, since the tape only needs the
latest price and previous close.
"""

from __future__ import annotations

from typing import TypedDict


class LatestQuote(TypedDict):
    ticker: str
    price: float
    change_pct: float


def fetch_latest_quotes(tickers: list[str]) -> list[LatestQuote]:
    import yfinance as yf  # lazy import: keeps this module importable offline

    quotes: list[LatestQuote] = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
            price = info["last_price"]
            prev_close = info["previous_close"]
            if not price or not prev_close:
                continue
            quotes.append(
                LatestQuote(ticker=ticker, price=price, change_pct=100 * (price - prev_close) / prev_close)
            )
        except Exception:
            continue  # one delisted/bad ticker shouldn't blank the whole tape
    return quotes

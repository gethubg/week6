"""News data access, mirroring price_provider.py's shape: a small Protocol
so retrieval/indexing can swap in a canned provider without touching
network/API-key state.

Real path wraps Finnhub's `/company-news` endpoint (Tech Spec §9.1 /
.env.example NEWS_API_KEY). Unlike NewsAPI.org's free tier -- which only
returns articles from the last month -- Finnhub's free tier covers roughly
the last 12 months (verified empirically; Finnhub doesn't document an exact
cutoff). Trade legs older than that still get an empty list rather than an
error, same as before, just with a longer window.
"""

from __future__ import annotations

import os
from datetime import date as date_cls
from datetime import datetime
from typing import Protocol, TypedDict


class NewsArticle(TypedDict):
    date: str
    source: str
    title: str
    summary: str
    url: str


class NewsProvider(Protocol):
    def get_news(self, ticker: str, start: str, end: str) -> list[NewsArticle] | dict:
        ...


def _parse(d: str) -> date_cls:
    return datetime.strptime(d, "%Y-%m-%d").date()


class FinnhubProvider:
    """Wraps Finnhub's /company-news endpoint with an in-memory cache per
    (ticker, start, end) to avoid redundant network calls when multiple legs
    share a ticker within one run. Ticker-native (unlike NewsAPI's keyword
    search on `q=ticker`), so results are already scoped to the company
    rather than any article that happens to mention the ticker string."""

    _BASE_URL = "https://finnhub.io/api/v1/company-news"

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], list[NewsArticle] | dict] = {}

    def get_news(self, ticker: str, start: str, end: str) -> list[NewsArticle] | dict:
        key = (ticker, start, end)
        if key in self._cache:
            return self._cache[key]

        result = self._fetch(ticker, start, end)
        self._cache[key] = result
        return result

    def _fetch(self, ticker: str, start: str, end: str) -> list[NewsArticle] | dict:
        import requests  # lazy import: keeps this module importable offline

        api_key = os.environ.get("NEWS_API_KEY")
        if not api_key:
            return {"error": "NEWS_API_KEY is not set"}

        try:
            response = requests.get(
                self._BASE_URL,
                params={"symbol": ticker, "from": start, "to": end, "token": api_key},
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # network failure, bad ticker, rate limit, etc.
            return {"error": f"failed to fetch news for {ticker}: {exc}"}

        if isinstance(payload, dict) and payload.get("error"):
            return {"error": payload["error"]}

        articles: list[NewsArticle] = []
        for item in payload:
            summary = item.get("summary") or item.get("headline") or ""
            if not summary.strip():
                # Embedders reject empty input downstream; a headline-less,
                # summary-less item has nothing worth indexing anyway.
                continue
            published = item.get("datetime")
            published_date = datetime.fromtimestamp(published).strftime("%Y-%m-%d") if published else start
            articles.append(
                NewsArticle(
                    date=published_date,
                    source=item.get("source") or "finnhub",
                    title=item.get("headline") or "",
                    summary=summary,
                    url=item.get("url") or "",
                )
            )
        return articles


class FakeNewsProvider:
    """Test/offline stand-in: serves canned articles from an in-memory table
    keyed by ticker, filtered to the requested [start, end] range."""

    def __init__(self, articles_by_ticker: dict[str, list[NewsArticle]]) -> None:
        self._articles_by_ticker = articles_by_ticker

    def get_news(self, ticker: str, start: str, end: str) -> list[NewsArticle] | dict:
        articles = self._articles_by_ticker.get(ticker)
        if not articles:
            return {"error": f"no news for {ticker}"}

        start_d, end_d = _parse(start), _parse(end)
        in_range = [a for a in articles if start_d <= _parse(a["date"]) <= end_d]
        if not in_range:
            return {"error": f"no news for {ticker} between {start} and {end}"}
        return in_range

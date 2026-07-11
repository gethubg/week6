"""Price data access (Tech Spec §4). Wraps yfinance behind a small
Protocol so the MCP server -- and anything calling it in tests -- can swap
in a canned provider without touching network/API-key state.
"""

from __future__ import annotations

from datetime import date as date_cls
from datetime import datetime, timedelta
from typing import Protocol, TypedDict


class PriceBar(TypedDict):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class PriceProvider(Protocol):
    def get_price_history(self, ticker: str, start: str, end: str) -> list[PriceBar] | dict:
        ...


def _parse(d: str) -> date_cls:
    return datetime.strptime(d, "%Y-%m-%d").date()


class YFinancePriceProvider:
    """Wraps yfinance.download with an in-memory cache per (ticker, start,
    end) to avoid redundant network calls when multiple legs share a
    ticker within one run (Tech Spec §4)."""

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str, str], list[PriceBar] | dict] = {}

    def get_price_history(self, ticker: str, start: str, end: str) -> list[PriceBar] | dict:
        key = (ticker, start, end)
        if key in self._cache:
            return self._cache[key]

        result = self._fetch(ticker, start, end)
        self._cache[key] = result
        return result

    def _fetch(self, ticker: str, start: str, end: str) -> list[PriceBar] | dict:
        import yfinance as yf  # lazy import: keeps this module importable offline

        try:
            df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        except Exception as exc:  # network failure, bad ticker, etc.
            return {"error": f"failed to fetch price history for {ticker}: {exc}"}

        if df is None or df.empty:
            return {"error": f"no price data returned for {ticker} between {start} and {end}"}

        # Recent yfinance versions return MultiIndex columns (e.g. ("Open", "SPY"))
        # even for a single ticker; flatten to plain column names so row["Open"]
        # yields a scalar instead of a Series.
        if df.columns.nlevels > 1:
            df.columns = df.columns.droplevel(1)

        bars: list[PriceBar] = []
        for idx, row in df.iterrows():
            bars.append(
                PriceBar(
                    date=idx.strftime("%Y-%m-%d"),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            )
        return bars


class FakePriceProvider:
    """Test/offline stand-in: serves canned bars from an in-memory table
    keyed by ticker, filtered to the requested [start, end) range."""

    def __init__(self, bars_by_ticker: dict[str, list[PriceBar]]) -> None:
        self._bars_by_ticker = bars_by_ticker

    def get_price_history(self, ticker: str, start: str, end: str) -> list[PriceBar] | dict:
        bars = self._bars_by_ticker.get(ticker)
        if not bars:
            return {"error": f"no price data for {ticker}"}

        start_d, end_d = _parse(start), _parse(end)
        in_range = [b for b in bars if start_d <= _parse(b["date"]) <= end_d]
        if not in_range:
            return {"error": f"no price data for {ticker} between {start} and {end}"}
        return in_range


def padded_window(date: str, window_days: int) -> tuple[str, str]:
    """[date - window_days, date + window_days] as ISO strings, for feeding
    into get_price_history ahead of a local-extremes computation."""
    d = _parse(date)
    start = d - timedelta(days=window_days)
    end = d + timedelta(days=window_days)
    return start.isoformat(), end.isoformat()

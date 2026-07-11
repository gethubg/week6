import pytest

from agents.analysis_agent import score_trades
from db.db import get_connection, init_db, insert_leg, insert_trade
from db.models import Leg, Trade
from mcp_server.price_provider import FakePriceProvider, padded_window
from mcp_server.tools import compute_local_extremes

AAPL_BARS = [
    {"date": "2023-12-22", "open": 95, "high": 96, "low": 90, "close": 95, "volume": 1},
    {"date": "2024-01-01", "open": 96, "high": 100, "low": 95, "close": 99, "volume": 1},
    {"date": "2024-01-11", "open": 118, "high": 120, "low": 110, "close": 119, "volume": 1},
    {"date": "2024-01-20", "open": 119, "high": 121, "low": 115, "close": 120, "volume": 1},
]


def _make_extremes_fn(bars_by_ticker):
    provider = FakePriceProvider(bars_by_ticker)

    def extremes_fn(ticker: str, date: str, window_days: int) -> dict:
        start, end = padded_window(date, window_days)
        bars = provider.get_price_history(ticker, start, end)
        if isinstance(bars, dict) and "error" in bars:
            return bars
        return compute_local_extremes(bars, date, window_days)

    return extremes_fn


def _sample_trade() -> Trade:
    trade = Trade(
        id="t1",
        ticker="AAPL",
        open_date="2024-01-01",
        close_date="2024-01-11",
        status="closed",
        quantity=10,
        avg_entry_price=100.0,
        avg_exit_price=120.0,
        realized_pnl=200.0,
    )
    trade.legs = [
        Leg(id="l1", trade_id="t1", type="buy", date="2024-01-01", price=100.0, quantity=10),
        Leg(id="l2", trade_id="t1", type="sell", date="2024-01-11", price=120.0, quantity=10),
    ]
    return trade


def _seed(conn, trade: Trade) -> None:
    """Mirrors what the Ingestion Agent would already have written: trade
    + leg rows must exist before timing_score rows can FK-reference them."""
    insert_trade(conn, trade)
    for leg in trade.legs:
        insert_leg(conn, leg)
    conn.commit()


def test_score_trades_scores_buy_and_sell_legs():
    conn = get_connection(":memory:")
    init_db(conn)
    trade = _sample_trade()
    _seed(conn, trade)

    scores, errors = score_trades([trade], conn, extremes_fn=_make_extremes_fn({"AAPL": AAPL_BARS}))

    assert not errors
    assert len(scores) == 2

    buy_score = next(s for s in scores if s.leg_id == "l1")
    sell_score = next(s for s in scores if s.leg_id == "l2")

    # Buy leg's +/-10d window spans bars 1-3: local range [90, 120].
    # Buy at 100 -> percentile ~33.3, middling entry -> neutral.
    assert buy_score.local_low == 90
    assert buy_score.local_high == 120
    assert buy_score.percentile == pytest.approx(33.33, abs=0.1)
    assert buy_score.verdict == "neutral"

    # Sell leg's +/-10d window spans bars 2-4: local range [95, 121].
    # Sell at 120 -> percentile ~96.2, near the local high -> strong.
    assert sell_score.local_low == 95
    assert sell_score.local_high == 121
    assert sell_score.percentile == pytest.approx(96.15, abs=0.1)
    assert sell_score.verdict == "strong"

    rows = conn.execute("SELECT * FROM timing_score").fetchall()
    assert len(rows) == 2


def test_score_trades_skips_dividend_legs():
    conn = get_connection(":memory:")
    init_db(conn)
    trade = _sample_trade()
    trade.legs.append(Leg(id="l3", trade_id="t1", type="dividend", date="2024-01-05", price=0.5, quantity=1))
    _seed(conn, trade)

    scores, errors = score_trades([trade], conn, extremes_fn=_make_extremes_fn({"AAPL": AAPL_BARS}))

    assert {s.leg_id for s in scores} == {"l1", "l2"}
    assert not errors


def test_score_trades_records_error_instead_of_fabricating_score():
    conn = get_connection(":memory:")
    init_db(conn)
    trade = _sample_trade()
    _seed(conn, trade)

    # No price data at all for this ticker.
    scores, errors = score_trades([trade], conn, extremes_fn=_make_extremes_fn({}))

    assert scores == []
    assert len(errors) == 2
    assert all("AAPL" in e for e in errors)

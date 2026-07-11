"""Ingestion Agent (Product Spec §5.1 / Tech Spec §6.1).

Parses a Robinhood CSV export, classifies each row, and runs FIFO lot
matching to produce closed/open/unmatched Trade + Leg records.

Row classification is rule-based on the ``Trans Code`` column for V0 (see
Product Spec's decision to keep the LLM out of deterministic math/parsing).
``classify_row_type`` is the seam for the "ambiguous row" case the tech spec
calls out: if a code doesn't match a known bucket, ``ambiguous_row_classifier``
gets a chance to call an LLM and decide; the default implementation just
returns None (skip + log), which keeps ingestion fully offline-testable.
"""

from __future__ import annotations

import csv
import io
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

from langsmith import traceable

from db.db import insert_leg, insert_trade
from db.models import Leg, Trade
from tools.lot_matching import IdGenerator, NormalizedRow, match_lots

# -- column normalization -----------------------------------------------------

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("Activity Date", "Process Date", "Date"),
    "ticker": ("Instrument", "Symbol", "Ticker"),
    "trans_code": ("Trans Code", "Transaction Code", "Type"),
    "quantity": ("Quantity", "Qty"),
    "price": ("Price",),
}

# Present in standard Robinhood exports but not required: dividend (CDIV)
# rows often leave Quantity/Price blank and only populate Amount.
_OPTIONAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "amount": ("Amount",),
}

_BUY_CODES = {"buy", "btc"}
_SELL_CODES = {"sell", "stc"}
_DIVIDEND_CODES = {"cdiv", "div", "dividend"}

AmbiguousRowClassifier = Callable[[str, str], "str | None"]


def default_ambiguous_classifier(trans_code: str, description: str) -> str | None:
    """Rule-based seam for codes not in the known buckets. Returns None
    (skip) by default; swap in an LLM-backed classifier later without
    touching the rest of the pipeline."""
    return None


@dataclass
class IngestionResult:
    trades: list[Trade]
    legs: list[Leg]
    skipped_rows: list[dict]


def _normalize_header(fieldnames: list[str]) -> dict[str, str]:
    """Map canonical field name -> actual CSV column name present."""
    resolved: dict[str, str] = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in fieldnames:
                resolved[canonical] = alias
                break
    missing = set(_COLUMN_ALIASES) - set(resolved)
    if missing:
        raise ValueError(f"CSV missing required columns for: {sorted(missing)}")

    for canonical, aliases in _OPTIONAL_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in fieldnames:
                resolved[canonical] = alias
                break
    return resolved


def _parse_money(raw: str) -> float:
    cleaned = raw.strip().replace("$", "").replace(",", "")
    negative = cleaned.startswith("(") and cleaned.endswith(")")
    if negative:
        cleaned = cleaned[1:-1]
    value = float(cleaned)
    return -value if negative else value


# Robinhood exports dates as M/D/YYYY; some rows may already be ISO if the
# source has been pre-normalized. Everything downstream (FIFO sort order,
# NormalizedRow.date's ISO contract, padded_window's strptime) assumes ISO,
# so this is the one place raw date strings get normalized.
_DATE_FORMATS = ("%Y-%m-%d", "%m/%d/%Y")


def _normalize_date(raw: str) -> str:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"unrecognized date format: {raw!r}")


def classify_row_type(
    trans_code: str,
    description: str = "",
    ambiguous_classifier: AmbiguousRowClassifier = default_ambiguous_classifier,
) -> str | None:
    code = trans_code.strip().lower()
    if code in _BUY_CODES:
        return "buy"
    if code in _SELL_CODES:
        return "sell"
    if code in _DIVIDEND_CODES:
        return "dividend"
    return ambiguous_classifier(trans_code, description)


def parse_robinhood_csv(
    csv_text: str,
    ambiguous_classifier: AmbiguousRowClassifier = default_ambiguous_classifier,
) -> tuple[list[NormalizedRow], list[dict]]:
    """Returns (normalized_rows_sorted_chronologically, skipped_rows)."""
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        return [], []
    columns = _normalize_header(list(reader.fieldnames))

    rows: list[NormalizedRow] = []
    skipped: list[dict] = []

    for raw_row in reader:
        trans_code = raw_row[columns["trans_code"]] or ""
        row_type = classify_row_type(trans_code, raw_row.get("Description") or "", ambiguous_classifier)
        if row_type is None:
            skipped.append(dict(raw_row))
            continue

        ticker = (raw_row[columns["ticker"]] or "").strip()
        date_raw = (raw_row[columns["date"]] or "").strip()
        quantity_raw = (raw_row[columns["quantity"]] or "").strip()
        price_raw = (raw_row[columns["price"]] or "").strip()

        try:
            date = _normalize_date(date_raw)
            if row_type == "dividend":
                # Dividend rows commonly leave Quantity/Price blank and only
                # populate Amount (the total dividend received).
                quantity = abs(float(quantity_raw.replace(",", ""))) if quantity_raw else 0.0
                if price_raw:
                    price = abs(_parse_money(price_raw))
                elif "amount" in columns and (raw_row[columns["amount"]] or "").strip():
                    price = abs(_parse_money(raw_row[columns["amount"]]))
                else:
                    raise ValueError("dividend row has no price or amount")
            else:
                quantity = abs(float(quantity_raw.replace(",", "")))
                price = abs(_parse_money(price_raw))
        except (ValueError, KeyError):
            skipped.append(dict(raw_row))
            continue

        rows.append(NormalizedRow(ticker=ticker, type=row_type, date=date, price=price, quantity=quantity))

    # Stable sort: chronological overall, original relative order preserved
    # within a ticker/date so FIFO matching is deterministic.
    rows.sort(key=lambda r: r.date)
    return rows, skipped


@traceable(name="ingestion_agent", run_type="chain")
def run_ingestion(
    csv_text: str,
    conn: sqlite3.Connection,
    ambiguous_classifier: AmbiguousRowClassifier = default_ambiguous_classifier,
    id_gen: IdGenerator | None = None,
) -> IngestionResult:
    rows, skipped = parse_robinhood_csv(csv_text, ambiguous_classifier)
    trades, legs = match_lots(rows, id_gen=id_gen)

    for trade in trades:
        insert_trade(conn, trade)
    for leg in legs:
        insert_leg(conn, leg)
    conn.commit()

    return IngestionResult(trades=trades, legs=legs, skipped_rows=skipped)

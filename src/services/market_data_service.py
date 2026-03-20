from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class MarketSnapshot:
    ticker: str
    company_name: str
    sector: str
    industry: str
    current_price: float | None
    currency: str | None
    market_cap: int | None
    recent_prices: pd.DataFrame


class MarketDataError(Exception):
    pass


logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _empty_recent_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])


def _normalize_recent_prices(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return _empty_recent_prices()

    recent = history.tail(7).copy()
    recent = recent.reset_index()
    recent = recent.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "date" in recent.columns:
        recent["date"] = pd.to_datetime(recent["date"]).dt.strftime("%Y-%m-%d")

    for col in ("open", "high", "low", "close"):
        if col in recent.columns:
            recent[col] = pd.to_numeric(recent[col], errors="coerce").round(2)

    if "volume" in recent.columns:
        recent["volume"] = pd.to_numeric(recent["volume"], errors="coerce").fillna(0).astype("int64")

    cols = [c for c in ["date", "open", "high", "low", "close", "volume"] if c in recent.columns]
    return recent[cols]


def fetch_market_snapshot(ticker: str) -> MarketSnapshot:
    symbol = (ticker or "").strip().upper() or "N/A"
    info: dict[str, Any] = {}
    fast_info: dict[str, Any] = {}
    history = pd.DataFrame()

    try:
        tk = yf.Ticker(symbol)
    except Exception:
        logger.exception("yfinance.Ticker initialization failed for %s", symbol)
        tk = None

    if tk is not None:
        try:
            # get_info() is generally more explicit than property access and easier to catch safely.
            info = tk.get_info() or {}
        except Exception:
            logger.exception("Failed to fetch info for %s", symbol)
            info = {}

        try:
            fast_info = dict(tk.fast_info or {})
        except Exception:
            logger.exception("Failed to fetch fast_info for %s", symbol)
            fast_info = {}

        try:
            history = tk.history(period="1mo", interval="1d", auto_adjust=False, actions=False)
        except Exception:
            logger.exception("Ticker.history failed for %s", symbol)
            history = pd.DataFrame()

    if history is None or history.empty:
        try:
            fallback = yf.download(
                symbol,
                period="1mo",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            # yfinance can return MultiIndex columns in some download cases.
            if isinstance(fallback.columns, pd.MultiIndex):
                fallback.columns = fallback.columns.get_level_values(0)
            history = fallback
        except Exception:
            logger.exception("yf.download fallback failed for %s", symbol)
            history = pd.DataFrame()

    recent = _normalize_recent_prices(history)

    current_price = (
        _safe_float(fast_info.get("lastPrice"))
        or _safe_float(fast_info.get("regularMarketPrice"))
        or _safe_float(info.get("currentPrice"))
        or _safe_float(info.get("regularMarketPrice"))
    )
    if current_price is None and "close" in recent.columns and not recent.empty:
        current_price = _safe_float(recent.iloc[-1].get("close"))

    company_name = (
        str(
            info.get("shortName")
            or info.get("longName")
            or info.get("displayName")
            or symbol
        )
        if symbol != "N/A"
        else "N/A"
    )
    sector = str(info.get("sector") or "N/A")
    industry = str(info.get("industry") or "N/A")
    currency = info.get("currency") or fast_info.get("currency")
    market_cap = _safe_int(info.get("marketCap")) or _safe_int(fast_info.get("marketCap"))

    return MarketSnapshot(
        ticker=symbol,
        company_name=company_name,
        sector=sector,
        industry=industry,
        current_price=current_price,
        currency=currency,
        market_cap=market_cap,
        recent_prices=recent,
    )

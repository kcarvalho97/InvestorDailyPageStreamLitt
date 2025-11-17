import io
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

from config import CACHE_TTL_SECONDS, MAX_HISTORY_DAYS


def _stamp_df(df: pd.DataFrame) -> pd.DataFrame:
    """Attach UTC fetch time to a DataFrame via .attrs."""
    df.attrs["fetched_at"] = datetime.utcnow()
    return df


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_fx_data(num_days: int) -> pd.DataFrame:
    """
    FX: historical USD → AUD / EUR / GBP / JPY rates from Frankfurter (no API key).

    Columns:
      - date       (datetime)
      - AUD/EUR/GBP/JPY (float) : quote currency per 1 USD
      - rate       (float)      : alias for AUD (for backwards compatibility)
      - change_pct (float)      : daily % change in USD→AUD
    """
    end_date = datetime.utcnow().date()
    effective_days = max(num_days - 1, 0)
    start_date = end_date - timedelta(days=effective_days)

    symbols = ["AUD", "EUR", "GBP", "JPY"]
    symbols_str = ",".join(symbols)

    url = (
        f"https://api.frankfurter.dev/v1/{start_date}..{end_date}"
        f"?base=USD&symbols={symbols_str}"
    )

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if "rates" not in data:
        raise RuntimeError(f"Unexpected FX API response: {data}")

    rows: list[dict] = []
    for date_str, rate_map in sorted(data["rates"].items()):
        row: dict[str, float | pd.Timestamp] = {"date": pd.to_datetime(date_str)}
        for cur in symbols:
            val = rate_map.get(cur)
            if val is not None:
                row[cur] = float(val)
        rows.append(row)

    if not rows:
        raise RuntimeError("Frankfurter returned no usable FX rates for this period.")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    if "AUD" in df.columns:
        df["rate"] = df["AUD"]
        df["change_pct"] = df["rate"].pct_change() * 100.0
    else:
        df["rate"] = np.nan
        df["change_pct"] = np.nan

    return _stamp_df(df)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_crypto_market_chart(
    coin_id: str,
    days: int = 365,
    vs_currency: str = "usd",
) -> pd.DataFrame:
    """
    Crypto: historical prices from CoinGecko `market_chart` endpoint.

    Returns:
      - date   (datetime)
      - price  (float, in vs_currency)
    """
    days = min(days, MAX_HISTORY_DAYS)

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        "vs_currency": vs_currency,
        "days": days,
        "interval": "daily",
    }
    headers = {
        "User-Agent": "streamlit-markets-dashboard/0.1",
    }

    resp = requests.get(url, params=params, headers=headers, timeout=10)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            raise RuntimeError(
                "CoinGecko returned 401 Unauthorized. They may now require an API key "
                "for this endpoint. Crypto charts are disabled, but the rest of the "
                "dashboard still works."
            ) from e
        raise

    js = resp.json()

    prices = js.get("prices")
    if not prices:
        raise RuntimeError(f"Unexpected CoinGecko response for {coin_id}: {js}")

    df = pd.DataFrame(prices, columns=["timestamp_ms", "price"])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
    df = df[["date", "price"]].sort_values("date").reset_index(drop=True)
    return _stamp_df(df)


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_stooq_ohlc(symbol: str, days: int = 365) -> pd.DataFrame:
    """
    Stocks/ETFs/Metals: daily OHLCV from Stooq via CSV download.

    Returns:
      - date   (datetime)
      - close  (float)
    """
    days = min(days, MAX_HISTORY_DAYS)

    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    text = resp.text

    df = pd.read_csv(io.StringIO(text))
    if not any(str(c).lower() == "date" for c in df.columns):
        df = pd.read_csv(io.StringIO(text), sep=";")

    date_col = next((c for c in df.columns if str(c).lower() == "date"), None)
    close_col = next((c for c in df.columns if str(c).lower() == "close"), None)
    if date_col is None or close_col is None:
        raise RuntimeError(f"Stooq CSV for {symbol} has unexpected columns: {df.columns}")

    date_vals = df[date_col]

    if pd.api.types.is_integer_dtype(date_vals):
        df["date"] = pd.to_datetime(date_vals.astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["date"] = pd.to_datetime(date_vals, errors="coerce")

    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    if not df.empty:
        end = df["date"].max()
        start = end - pd.Timedelta(days=days - 1)
        df = df[df["date"] >= start].copy().reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"Stooq returned no usable data for {symbol}")

    return _stamp_df(df[["date", "close"]])


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_etf_bundle(days: int = MAX_HISTORY_DAYS) -> dict:
    """
    Fetch a small bundle of ETFs from Stooq (keyless).
    Uses US-listed tickers only for reliability.
    """
    days = min(days, MAX_HISTORY_DAYS)

    etf_symbols = {
        "IVV (iShares Core S&P 500)": "IVV.US",
        "SCHG (Schwab US Large-Cap Growth)": "SCHG.US",
        "SPY (SPDR S&P 500)": "SPY.US",
        "ACWX (iShares MSCI ACWI ex US)": "ACWX.US",
    }

    data: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for label, symbol in etf_symbols.items():
        try:
            df = fetch_stooq_ohlc(symbol, days=days)
            data[label] = df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label} ({symbol}): {e}")

    return {"data": data, "errors": errors}


@st.cache_data(show_spinner=False, ttl=CACHE_TTL_SECONDS)
def fetch_metals_bundle(days: int = MAX_HISTORY_DAYS) -> dict:
    """
    Fetch precious metals from Stooq (keyless):
    - Gold (XAUUSD)
    - Silver (XAGUSD)
    - Platinum (XPTUSD)

    Returns: {"data": {label: df, ...}, "errors": [msg, ...]}
    """
    days = min(days, MAX_HISTORY_DAYS)

    metals_symbols = {
        "Gold (XAUUSD)": "XAUUSD",
        "Silver (XAGUSD)": "XAGUSD",
        "Platinum (XPTUSD)": "XPTUSD",
    }

    data: dict[str, pd.DataFrame] = {}
    errors: list[str] = []

    for label, symbol in metals_symbols.items():
        try:
            df = fetch_stooq_ohlc(symbol, days=days)
            data[label] = df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label} ({symbol}): {e}")

    return {"data": data, "errors": errors}


import io
import concurrent.futures
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf  # NEW: Replaces CoinGecko for full history

from config import CACHE_TTL_SECONDS, MAX_HISTORY_DAYS


def _stamp_df(df: pd.DataFrame) -> pd.DataFrame:
    """Attach UTC fetch time to a DataFrame via .attrs."""
    df.attrs["fetched_at"] = datetime.utcnow()
    return df


def fetch_fx_data(num_days: int = MAX_HISTORY_DAYS) -> pd.DataFrame:
    """
    Fetch historical FX rates (blocking).
    ALWAYS fetches MAX_HISTORY_DAYS so we can cache it once and filter later.
    """
    end_date = datetime.utcnow().date()
    effective_days = max(num_days - 1, 0)
    start_date = end_date - timedelta(days=effective_days)

    symbols = ["AUD", "EUR", "GBP", "JPY", "CNY", "INR"]
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

    rows = []
    for date_str, rate_map in sorted(data["rates"].items()):
        row = {"date": pd.to_datetime(date_str)}
        for cur in symbols:
            val = rate_map.get(cur)
            if val is not None:
                row[cur] = float(val)
        rows.append(row)

    if not rows:
        raise RuntimeError("Frankfurter returned no usable FX rates.")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Backwards compatibility cols
    if "AUD" in df.columns:
        df["rate"] = df["AUD"]
        df["change_pct"] = df["rate"].pct_change() * 100.0
    else:
        df["rate"] = np.nan
        df["change_pct"] = np.nan

    return _stamp_df(df)


def fetch_crypto_yfinance() -> dict:
    """
    Fetch FULL history for BTC and ETH using yfinance.
    Returns: {'BTC': df, 'ETH': df}
    """
    # Tickers mapping: Label -> Yahoo Ticker
    tickers_map = {
        "BTC": "BTC-USD",
        "ETH": "ETH-USD"
    }
    
    # Download all in one batch for speed
    # auto_adjust=True gets the Split/Dividend adjusted price (standard for charts)
    raw_data = yf.download(
        list(tickers_map.values()), 
        period="max", 
        interval="1d", 
        auto_adjust=True, 
        progress=False,
        threads=True
    )

    if raw_data.empty:
        raise RuntimeError("Yahoo Finance returned no crypto data.")

    # Standardize results into our dict format
    results = {}
    
    for label, ticker in tickers_map.items():
        # yfinance returns a MultiIndex if >1 ticker: (PriceType, Ticker)
        # We just want the 'Close' for the specific ticker
        try:
            # Check if MultiIndex (multiple tickers downloaded)
            if isinstance(raw_data.columns, pd.MultiIndex):
                price_series = raw_data["Close"][ticker]
            else:
                # Single ticker download structure
                price_series = raw_data["Close"]

            df = price_series.reset_index()
            df.columns = ["date", "price"]
            
            # Clean dates: Remove TimeZone info to match Stooq/Frankfurter
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            
            # Sort and drop NaNs
            df = df.dropna().sort_values("date").reset_index(drop=True)
            
            results[label] = _stamp_df(df)
            
        except KeyError:
            # Ticker failed inside the batch
            continue

    if not results:
        raise RuntimeError("Could not parse yfinance crypto data.")

    return results


def fetch_stooq_ohlc(symbol: str, days: int = MAX_HISTORY_DAYS) -> pd.DataFrame:
    """
    Fetch Stooq data. Always tries to fetch MAX_HISTORY_DAYS.
    """
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
        raise RuntimeError(f"Bad columns in Stooq CSV for {symbol}")

    # Clean dates
    date_vals = df[date_col]
    if pd.api.types.is_integer_dtype(date_vals):
        df["date"] = pd.to_datetime(date_vals.astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["date"] = pd.to_datetime(date_vals, errors="coerce")

    df["close"] = pd.to_numeric(df[close_col], errors="coerce")
    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # Filter to max history requested (global limit)
    if not df.empty:
        end = df["date"].max()
        start = end - pd.Timedelta(days=days - 1)
        df = df[df["date"] >= start].copy().reset_index(drop=True)

    if df.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    return _stamp_df(df[["date", "close"]])


def _fetch_stooq_bundle(symbol_map: dict[str, str], days: int) -> dict:
    """Helper for synchronous bundle fetching (used inside threads)."""
    data = {}
    errors = []
    for label, symbol in symbol_map.items():
        try:
            df = fetch_stooq_ohlc(symbol, days=days)
            data[label] = df
        except Exception as e:
            errors.append(f"{label}: {e}")
    return {"data": data, "errors": errors}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def fetch_all_data_concurrently(max_days: int):
    """
    The Master Loader:
    1. Runs in parallel (Threads).
    2. Fetches MAXIMUM history for everything.
    3. Caches the huge result bundle.
    """
    
    # Define the work to be done
    def get_fx():
        try:
            return fetch_fx_data(max_days), None
        except Exception as e:
            return None, str(e)

    def get_crypto():
        # Uses yfinance now (no key, full history)
        try:
            data = fetch_crypto_yfinance()
            return data, None
        except Exception as e:
            return {}, str(e)

    def get_etfs():
        # UPDATED: A truly diversified "Global Macro" list
        symbols = {
            "VOO (S&P 500)": "VOO.US",         # The Baseline
            "QQQ (Tech/Growth)": "QQQ.US",     # Tech Sector
            "SCHD (Dividend/Value)": "SCHD.US", # Value/Defense
            "IWM (Small Cap)": "IWM.US",       # Risk/Real Economy
            "TLT (20y Treasuries)": "TLT.US",  # Hedge/Rates
            "VXUS (International)": "VXUS.US", # Global Diversification
        }
        res = _fetch_stooq_bundle(symbols, max_days)
        return res, res.get("errors")

    def get_metals():
        symbols = {
            "Gold (XAUUSD)": "XAUUSD",
            "Silver (XAGUSD)": "XAGUSD",
            "Platinum (XPTUSD)": "XPTUSD",
        }
        res = _fetch_stooq_bundle(symbols, max_days)
        return res, res.get("errors")

    # Execute in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_fx = executor.submit(get_fx)
        future_crypto = executor.submit(get_crypto)
        future_etf = executor.submit(get_etfs)
        future_metals = executor.submit(get_metals)

        # Gather results
        fx_data, fx_err = future_fx.result()
        crypto_data, crypto_err = future_crypto.result()
        etf_bundle, etf_err_list = future_etf.result()
        metals_bundle, metals_err_list = future_metals.result()

    # Consolidate errors for the bundle dictionaries
    etf_err = None if not etf_err_list else "; ".join(etf_err_list)
    metals_err = None if not metals_err_list else "; ".join(metals_err_list)

    return {
        "fx": (fx_data, fx_err),
        "crypto": (crypto_data, crypto_err),
        "etf": (etf_bundle, etf_err),
        "metals": (metals_bundle, metals_err)
    }

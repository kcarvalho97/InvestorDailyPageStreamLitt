import numpy as np
import pandas as pd
import streamlit as st

from config import CACHE_TTL_SECONDS, MAX_HISTORY_DAYS
from data_loaders import fetch_all_data_concurrently
from tabs_overview import render_overview, render_fx_tab
from tabs_etf import render_etf_tab


def get_attr_time(df):
    """Helper: pull 'fetched_at' timestamp out of a DataFrame.attrs, if present."""
    if df is None:
        return None
    attrs = getattr(df, "attrs", None)
    if not attrs:
        return None
    return attrs.get("fetched_at")


def filter_by_date(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Efficiently slices the dataframe in memory based on the requested history window.
    Assumes 'date' column exists and is sorted.
    """
    if df is None or df.empty:
        return df

    # Calculate the cutoff date
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].copy()


def filter_bundle(bundle: dict, days: int) -> dict:
    """Helper to filter an entire bundle (ETF/Metals) of dataframes."""
    if not bundle or "data" not in bundle:
        return bundle

    new_data = {}
    for label, df in bundle["data"].items():
        new_data[label] = filter_by_date(df, days)

    return {"data": new_data, "errors": bundle.get("errors", [])}


def main() -> None:
    # ---------- BASIC PAGE SETUP ----------
    st.set_page_config(
        page_title="Markets Dashboard (Keyless APIs)",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Markets Dashboard – Keyless APIs Only")
    st.caption(
        "Interactive Streamlit app using only **no-key** data sources:\n"
        "- FX via Frankfurter | Crypto via **Yahoo Finance** (No Limits) | ETFs & Metals via Stooq\n"
        "- **Optimized:** Loads all data in parallel and caches locally."
    )

    # ---------- SIDEBAR CONTROLS ----------
    st.sidebar.header("Global Options")

    history_days = st.sidebar.slider(
        "History window (days)",
        min_value=30,
        max_value=MAX_HISTORY_DAYS,
        value=365,
        step=30,
        help="Adjusts the display window. Data is cached, so this is instant.",
    )
    assets_years = max(1, int(round(history_days / 365)))

    use_log_scale = st.sidebar.checkbox(
        "Use log scale",
        value=False,
    )

    st.sidebar.markdown("### Quick USD converter")
    usd_amount = st.sidebar.number_input("Amount in USD", value=100.0, step=10.0)
    target_fx = st.sidebar.selectbox("Convert into", ["AUD", "EUR", "GBP", "JPY"], index=0)

    # ---------- LOAD DATA (CONCURRENTLY) ----------
    # We fetch the MAXIMUM amount of data once.
    # The cache key depends only on MAX_HISTORY_DAYS, not the slider value.
    with st.spinner("Fetching all market data (Parallel Mode)..."):
        results = fetch_all_data_concurrently(MAX_HISTORY_DAYS)

    # Unpack results
    raw_fx, fx_error = results["fx"]
    raw_crypto, crypto_error = results["crypto"]
    raw_etf_bundle, etf_error = results["etf"]
    raw_metals_bundle, metals_error = results["metals"]

    # ---------- FILTER DATA (IN MEMORY) ----------
    # Now we slice the raw "10 year" data down to "history_days" for the view
    fx_data = filter_by_date(raw_fx, history_days)

    crypto_data = {}
    if raw_crypto:
        for coin, df in raw_crypto.items():
            crypto_data[coin] = filter_by_date(df, history_days)

    etf_data_bundle = filter_bundle(raw_etf_bundle, history_days)
    metals_data_bundle = filter_bundle(raw_metals_bundle, history_days)

    # ---------- LAST REFRESH TIME ----------
    refresh_candidates = []
    if raw_fx is not None:
        refresh_candidates.append(get_attr_time(raw_fx))
    if raw_crypto:
        for df in raw_crypto.values():
            if df is not None:
                refresh_candidates.append(get_attr_time(df))

    if refresh_candidates:
        last_refresh = max(t for t in refresh_candidates if t is not None)
        st.sidebar.info(
            f"**Data Cache:** {CACHE_TTL_SECONDS // 60} min\n"
            f"**Last Refresh:** {last_refresh.strftime('%H:%M:%S')} UTC"
        )

    # ---------- SIDEBAR FX CONVERTER ----------
    if fx_error is None and fx_data is not None and not fx_data.empty:
        latest_row = fx_data.iloc[-1]
        rate = latest_row.get(target_fx)
        if rate and not np.isnan(rate):
            st.sidebar.metric(f"≈ {target_fx} now", f"{usd_amount * rate:,.2f}")

    st.markdown("---")

    # ---------- TABS ----------
    tab_overview, tab_fx, tab_etf = st.tabs(
        ["🏠 Overview", "💱 FX (Detail)", "📊 ETFs (Stooq)"]
    )

    with tab_overview:
        render_overview(
            fx_data=fx_data,
            fx_error=fx_error,
            crypto_data=crypto_data,
            crypto_error=crypto_error,
            etf_bundle=etf_data_bundle,
            etf_error=etf_error,
            metals_bundle=metals_data_bundle,
            metals_error=metals_error,
            assets_years=assets_years,
            use_log_scale=use_log_scale,
        )

    with tab_fx:
        render_fx_tab(
            fx_data=fx_data,
            fx_error=fx_error,
            assets_years=assets_years,
            use_log_scale=use_log_scale,
        )

    with tab_etf:
        render_etf_tab(
            etf_bundle=etf_data_bundle,
            etf_error=etf_error,
            assets_years=assets_years,
            use_log_scale=use_log_scale,
        )


if __name__ == "__main__":
    main()

import numpy as np
import pandas as pd
import streamlit as st

from config import CACHE_TTL_SECONDS, MAX_HISTORY_DAYS
from data_loaders import fetch_all_data_concurrently
from tabs_overview import render_overview
from tabs_etf import render_etf_tab


def get_attr_time(df):
    """Helper: pull 'fetched_at' timestamp out of a DataFrame.attrs, if present."""
    if df is None:
        return None
    attrs = getattr(df, "attrs", None)
    if not attrs:
        return None
    return attrs.get("fetched_at")


def filter_by_date(df: pd.DataFrame | None, days: int) -> pd.DataFrame | None:
    """Slice a dataframe to the last `days` days, assuming a sorted 'date' column."""
    if df is None or df.empty:
        return df
    cutoff = df["date"].max() - pd.Timedelta(days=days)
    return df[df["date"] >= cutoff].copy()


def filter_bundle(bundle: dict | None, days: int) -> dict | None:
    """Filter an entire bundle (ETF/Metals) of dataframes in-place."""
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
        "Interactive Streamlit app using only **no-key** data sources (Frankfurter, "
        "Stooq, Yahoo Finance). Loads all data in parallel and caches it on the "
        "server for fast reuse."
    )

    # ---------- LOAD DATA (CONCURRENTLY, MAX RANGE) ----------
    with st.spinner("Fetching all market data (parallel mode)…"):
        results = fetch_all_data_concurrently(MAX_HISTORY_DAYS)

    raw_fx, fx_error = results["fx"]
    raw_crypto, crypto_error = results["crypto"]
    raw_etf_bundle, etf_error = results["etf"]
    raw_metals_bundle, metals_error = results["metals"]

    # ---------- VIEW / SCALE OPTIONS (HIDDEN MENU) ----------
    with st.expander("☰ View & scale options", expanded=False):
        history_days = st.slider(
            "History window (days)",
            min_value=30,
            max_value=MAX_HISTORY_DAYS,
            value=365,
            step=30,
            help=(
                "Controls how much of the loaded history is shown. "
                "Data is cached at a larger horizon, so this is instant."
            ),
        )
        use_log_scale = st.checkbox(
            "Use log scale for price charts",
            value=False,
            help="Useful when assets move by very different percentages.",
        )

        st.markdown("---")
        st.markdown("**Quick USD converter**")
        usd_amount = st.number_input("Amount in USD", value=100.0, step=10.0)
        target_fx = st.selectbox(
            "Convert into",
            ["AUD", "EUR", "GBP", "JPY"],
            index=0,
        )

    assets_years = max(1, int(round(history_days / 365)))

    # ---------- FILTER DATA (IN MEMORY) ----------
    fx_data = filter_by_date(raw_fx, history_days)

    crypto_data: dict[str, pd.DataFrame] = {}
    if raw_crypto:
        for coin, df in raw_crypto.items():
            crypto_data[coin] = filter_by_date(df, history_days)

    etf_data_bundle = filter_bundle(raw_etf_bundle, history_days)
    metals_data_bundle = filter_bundle(raw_metals_bundle, history_days)

    # ---------- LAST REFRESH TIME (CAPTION) ----------
    refresh_candidates = []
    if raw_fx is not None:
        refresh_candidates.append(get_attr_time(raw_fx))
    if raw_crypto:
        for df in raw_crypto.values():
            if df is not None:
                refresh_candidates.append(get_attr_time(df))

    if refresh_candidates:
        last_refresh = max(t for t in refresh_candidates if t is not None)
        if last_refresh is not None:
            st.caption(
                f"Data cache: ~{CACHE_TTL_SECONDS // 60} min · "
                f"Last refresh: {last_refresh.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )

    # ---------- QUICK FX CONVERTER (INLINE) ----------
    if (
        fx_error is None
        and fx_data is not None
        and not fx_data.empty
        and "rate" in fx_data.columns
    ):
        latest_row = fx_data.iloc[-1]
        rate = latest_row.get(target_fx)
        if rate is not None and not np.isnan(rate):
            converted = usd_amount * rate
            st.info(
                f"Quick converter: **{usd_amount:,.0f} USD** ≈ "
                f"**{converted:,.2f} {target_fx}** (latest rate)."
            )

    st.markdown("---")

    # ---------- TABS ----------
    tab_overview, tab_etf = st.tabs(["🏠 Overview", "📊 ETFs (Stooq)"])

    with tab_overview:
        render_overview(
            fx_data=fx_data,
            fx_error=fx_error,
            crypto_data=crypto_data,
            crypto_error=crypto_error,
            etf_data_bundle=etf_data_bundle,
            etf_error=etf_error,
            metals_data_bundle=metals_data_bundle,
            metals_error=metals_error,
            assets_years=assets_years,
            use_log_scale=use_log_scale,
        )

    with tab_etf:
        render_etf_tab(
            etf_data_bundle=etf_data_bundle,
            etf_error=etf_error,
            assets_years=assets_years,
            use_log_scale=use_log_scale,
        )


if __name__ == "__main__":
    main()

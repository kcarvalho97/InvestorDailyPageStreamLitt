import numpy as np
import streamlit as st

from config import CACHE_TTL_SECONDS, MAX_HISTORY_DAYS
from data_loaders import (
    fetch_fx_data,
    fetch_crypto_market_chart,
    fetch_etf_bundle,
    fetch_metals_bundle,
)
from tabs_overview import render_overview
from tabs_fx import render_fx_tab
from tabs_crypto import render_crypto_tab
from tabs_etf import render_etf_tab
from tabs_metals import render_metals_tab


def get_attr_time(df):
    """Helper: pull 'fetched_at' timestamp out of a DataFrame.attrs, if present."""
    if df is None:
        return None
    attrs = getattr(df, "attrs", None)
    if not attrs:
        return None
    return attrs.get("fetched_at")


def main() -> None:
    # ---------- BASIC PAGE SETUP ----------
    st.set_page_config(
        page_title="Markets Dashboard (Keyless APIs)",
        page_icon="📈",
        layout="wide",
    )

    st.title("📈 Markets Dashboard – Keyless APIs Only")
    st.caption(
        "Interactive Streamlit app using only **no-key** data sources where possible:\n"
        "- FX (USD → AUD / EUR / GBP / JPY) via [Frankfurter](https://frankfurter.dev/)\n"
        "- Crypto (BTC & ETH) via [CoinGecko](https://www.coingecko.com/en/api/documentation)\n"
        "- ETFs via free **Stooq** CSV endpoints (IVV, SCHG, SPY, ACWX)\n"
        "- Metals (Gold, Silver, Platinum) via **Stooq** symbols (XAUUSD, XAGUSD, XPTUSD)."
    )

    # ---------- SIDEBAR CONTROLS ----------
    st.sidebar.header("Global Options")

    history_days = st.sidebar.slider(
        "History window (days – all assets)",
        min_value=30,
        max_value=MAX_HISTORY_DAYS,
        value=365,
        step=30,
        help="How far back to fetch data for FX, Crypto, ETFs and Metals (up to ~10 years).",
    )
    assets_years = max(1, int(round(history_days / 365)))

    use_log_scale = st.sidebar.checkbox(
        "Use log scale for price charts",
        value=False,
        help="Helps compare assets that moved a lot in % terms.",
    )

    st.sidebar.markdown("### Quick USD converter")
    usd_amount = st.sidebar.number_input(
        "Amount in USD",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )

    target_fx = st.sidebar.selectbox(
        "Convert into",
        ["AUD", "EUR", "GBP", "JPY"],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "FX is live from Frankfurter.\n\n"
        "Crypto is live from CoinGecko (keyless where possible).\n\n"
        "ETFs and Metals use real end-of-day data from Stooq.\n"
        f"All calls are cached for {CACHE_TTL_SECONDS // 60} minutes."
    )

    # ---------- LOAD DATA ----------
    fx_data, fx_error = None, None
    crypto_data, crypto_error = {}, None
    etf_data_bundle, etf_error = None, None
    metals_data_bundle, metals_error = None, None

    with st.spinner("Loading FX data (USD vs AUD / EUR / GBP / JPY)..."):
        try:
            fx_data = fetch_fx_data(history_days)
        except Exception as e:  # noqa: BLE001
            fx_error = str(e)

    with st.spinner("Loading crypto data (BTC & ETH)..."):
        try:
            crypto_data["BTC"] = fetch_crypto_market_chart(
                "bitcoin", days=history_days, vs_currency="usd"
            )
            crypto_data["ETH"] = fetch_crypto_market_chart(
                "ethereum", days=history_days, vs_currency="usd"
            )
        except Exception as e:  # noqa: BLE001
            crypto_error = str(e)

    with st.spinner("Loading ETF data (IVV, SCHG, SPY, ACWX from Stooq)..."):
        try:
            etf_data_bundle = fetch_etf_bundle(days=history_days)
        except Exception as e:  # noqa: BLE001
            etf_error = str(e)

    with st.spinner("Loading metals data (XAUUSD, XAGUSD, XPTUSD from Stooq)..."):
        try:
            metals_data_bundle = fetch_metals_bundle(days=history_days)
        except Exception as e:  # noqa: BLE001
            metals_error = str(e)

    # ---------- LAST REFRESH TIME ----------
    refresh_candidates = []

    if fx_data is not None:
        t = get_attr_time(fx_data)
        if t:
            refresh_candidates.append(t)

    for coin_df in crypto_data.values():
        t = get_attr_time(coin_df)
        if t:
            refresh_candidates.append(t)

    if etf_data_bundle and etf_data_bundle.get("data"):
        for df in etf_data_bundle["data"].values():
            t = get_attr_time(df)
            if t:
                refresh_candidates.append(t)

    if metals_data_bundle and metals_data_bundle.get("data"):
        for df in metals_data_bundle["data"].values():
            t = get_attr_time(df)
            if t:
                refresh_candidates.append(t)

    if refresh_candidates:
        last_refresh = max(refresh_candidates)
        st.caption(
            f"**Last data refresh:** {last_refresh.strftime('%Y-%m-%d %H:%M:%S')} UTC "
            f"(cache TTL: {CACHE_TTL_SECONDS // 60} min, history ≈ {assets_years}y)"
        )
    else:
        st.caption("**Last data refresh:** unavailable (all data sources failed).")

    # ---------- SIDEBAR FX CONVERTER ----------
    if fx_error is None and fx_data is not None and not fx_data.empty:
        latest_row = fx_data.iloc[-1]
        latest_rate = latest_row.get(target_fx, np.nan)
        if latest_rate is not None and not np.isnan(latest_rate):
            st.sidebar.metric(f"≈ {target_fx} now", f"{usd_amount * latest_rate:,.2f}")
        else:
            st.sidebar.warning(f"No latest USD → {target_fx} rate available.")

    st.markdown("---")

    # ---------- TABS ----------
    tab_overview, tab_fx, tab_crypto, tab_etf, tab_metals = st.tabs(
        [
            "🏠 Overview",
            "💱 FX (USD vs AUD/EUR/GBP/JPY)",
            "₿ Crypto (BTC & ETH)",
            "📊 ETFs (Stooq)",
            "🥇 Metals (Stooq)",
        ]
    )

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

    with tab_fx:
        render_fx_tab(
            fx_data=fx_data,
            fx_error=fx_error,
            assets_years=assets_years,
            use_log_scale=use_log_scale,
        )

    with tab_crypto:
        render_crypto_tab(
            crypto_data=crypto_data,
            crypto_error=crypto_error,
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

    with tab_metals:
        render_metals_tab(
            metals_data_bundle=metals_data_bundle,
            metals_error=metals_error,
            assets_years=assets_years,
            use_log_scale=use_log_scale,
        )


if __name__ == "__main__":
    main()


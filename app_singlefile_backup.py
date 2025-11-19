import io
from datetime import datetime, timedelta

import numpy as np                    # Still here in case you reuse it later
import pandas as pd                   # Table + time series handling
import requests                       # For HTTP requests
import streamlit as st               # Web UI framework
import plotly.express as px          # High-level Plotly charts
import plotly.graph_objects as go    # Low-level Plotly API
from plotly.subplots import make_subplots  # For dual-axis charts

# ============================================================
# 0. GLOBAL STYLE / COLORS
# ============================================================

PRIMARY_COLOR = "#4E79A7"     # Main line / area color (blue)
ACCENT_COLOR = "#F28E2B"      # Second line (orange)
POSITIVE_COLOR = "#59A14F"    # Positive bars (green)
NEGATIVE_COLOR = "#E15759"    # Negative bars (red)
GOLD_COLOR = "#D4AF37"        # Gold line
SILVER_COLOR = "#C0C0C0"      # Silver line

BASE_LAYOUT = dict(
    template="plotly_dark",                 # dark background, light text
    paper_bgcolor="rgba(0,0,0,0)",          # transparent -> Streamlit dark BG
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=60, b=40),
)

# Cache lifetime: 10 minutes
CACHE_TTL_SECONDS = 600

# Global “10y” history ceiling (in days)
MAX_HISTORY_DAYS = 3650


# ============================================================
# 1. BASIC PAGE SETUP
# ============================================================

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

# ============================================================
# 2. SIDEBAR CONTROLS
# ============================================================

st.sidebar.header("Global Options")

# One global history window for ALL assets
history_days = st.sidebar.slider(
    "History window (days – all assets)",
    min_value=30,
    max_value=MAX_HISTORY_DAYS,
    value=365,        # ~1 year
    step=30,
    help="How far back to fetch data for FX, Crypto, ETFs and Metals (up to ~10 years).",
)

# Approximate years for chart titles
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
    "All calls are cached for 10 minutes."
)


# ============================================================
# 3. DATA FETCH HELPERS (CACHED, 10 MIN TTL)
# ============================================================

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
    # Frankfurter ranges are inclusive, so subtract 1 to match slider intent
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

    rows = []
    for date_str, rate_map in sorted(data["rates"].items()):
        row = {"date": pd.to_datetime(date_str)}
        for cur in symbols:
            val = rate_map.get(cur)
            if val is not None:
                row[cur] = float(val)
        rows.append(row)

    if not rows:
        raise RuntimeError("Frankfurter returned no usable FX rates for this period.")

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    # Keep old interface for USD→AUD
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
    No API key in this app; if CoinGecko now requires a key, we'll show a friendly error.

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
        # Helps with some free APIs that dislike "empty" clients
        "User-Agent": "streamlit-markets-dashboard/0.1"
    }

    resp = requests.get(url, params=params, headers=headers, timeout=10)
    try:
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 401:
            raise RuntimeError(
                "CoinGecko returned **401 Unauthorized**. They may now require an API key "
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
    Keyless, end-of-day market data.

    URL pattern:
        https://stooq.com/q/d/l/?s={symbol}&i=d

    Returns:
      - date   (datetime)
      - close  (float)
    """
    days = min(days, MAX_HISTORY_DAYS)

    url = f"https://stooq.com/q/d/l/?s={symbol.lower()}&i=d"
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    text = resp.text

    # Try comma first, then semicolon as fallback
    df = pd.read_csv(io.StringIO(text))
    if not any(str(c).lower() == "date" for c in df.columns):
        df = pd.read_csv(io.StringIO(text), sep=";")

    # Find date + close columns (case-insensitive)
    date_col = next(
        (c for c in df.columns if str(c).lower() == "date"),
        None,
    )
    close_col = next(
        (c for c in df.columns if str(c).lower() == "close"),
        None,
    )
    if date_col is None or close_col is None:
        raise RuntimeError(f"Stooq CSV for {symbol} has unexpected columns: {df.columns}")

    date_vals = df[date_col]

    # Stooq often encodes dates as integers like 20250117
    if pd.api.types.is_integer_dtype(date_vals):
        df["date"] = pd.to_datetime(date_vals.astype(str), format="%Y%m%d", errors="coerce")
    else:
        df["date"] = pd.to_datetime(date_vals, errors="coerce")

    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    df = df.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)

    # Trim to last `days` days
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

    data = {}
    errors = []

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
    Each df has:
      - date
      - close
    """
    days = min(days, MAX_HISTORY_DAYS)

    metals_symbols = {
        "Gold (XAUUSD)": "XAUUSD",
        "Silver (XAGUSD)": "XAGUSD",
        "Platinum (XPTUSD)": "XPTUSD",
    }

    data = {}
    errors = []

    for label, symbol in metals_symbols.items():
        try:
            df = fetch_stooq_ohlc(symbol, days=days)
            data[label] = df
        except Exception as e:  # noqa: BLE001
            errors.append(f"{label} ({symbol}): {e}")

    return {"data": data, "errors": errors}


# ============================================================
# 4. LOAD DATA (WITH ERROR HANDLING)
# ============================================================

fx_data, fx_error = None, None
crypto_data, crypto_error = {}, None
etf_data_bundle, etf_error = None, None
metals_data_bundle, metals_error = None, None

# --- FX ---
with st.spinner("Loading FX data (USD vs AUD / EUR / GBP / JPY)..."):
    try:
        fx_data = fetch_fx_data(history_days)
    except Exception as e:  # noqa: BLE001
        fx_error = str(e)

# --- Crypto (BTC + ETH) ---
with st.spinner("Loading crypto data (BTC & ETH)..."):
    try:
        crypto_data["BTC"] = fetch_crypto_market_chart("bitcoin", days=history_days, vs_currency="usd")
        crypto_data["ETH"] = fetch_crypto_market_chart("ethereum", days=history_days, vs_currency="usd")
    except Exception as e:  # noqa: BLE001
        crypto_error = str(e)

# --- ETFs from Stooq ---
with st.spinner("Loading ETF data (IVV, SCHG, SPY, ACWX from Stooq)..."):
    try:
        etf_data_bundle = fetch_etf_bundle(days=history_days)
    except Exception as e:  # noqa: BLE001
        etf_error = str(e)

# --- Metals from Stooq (Gold, Silver, Platinum) ---
with st.spinner("Loading metals data (XAUUSD, XAGUSD, XPTUSD from Stooq)..."):
    try:
        metals_data_bundle = fetch_metals_bundle(days=history_days)
    except Exception as e:  # noqa: BLE001
        metals_error = str(e)


# ============================================================
# 5. LAST REFRESH TIME (FROM WHATEVER DATA WE HAVE)
# ============================================================

def get_attr_time(df: pd.DataFrame | None) -> datetime | None:
    if df is None:
        return None
    return df.attrs.get("fetched_at")


refresh_candidates: list[datetime] = []

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

# Use latest FX for sidebar converter (if available)
if fx_error is None and fx_data is not None and not fx_data.empty:
    latest_row = fx_data.iloc[-1]
    latest_rate = latest_row.get(target_fx)
    if latest_rate is not None and not np.isnan(latest_rate):
        st.sidebar.metric(f"≈ {target_fx} now", f"{usd_amount * latest_rate:,.2f}")
    else:
        st.sidebar.warning(f"No latest USD → {target_fx} rate available.")

st.markdown("---")


# ============================================================
# 6. TABS (OVERVIEW + PER-ASSET)
# ============================================================

tab_overview, tab_fx, tab_crypto, tab_etf, tab_metals = st.tabs(
    [
        "🏠 Overview",
        "💱 FX (USD vs AUD/EUR/GBP/JPY)",
        "₿ Crypto (BTC & ETH)",
        "📊 ETFs (Stooq)",
        "🥇 Metals (Stooq)",
    ]
)


# ============================================================
# 7. OVERVIEW TAB – INVESTOR-FOCUSED ORDER
#    ETFs → Metals (Gold) → FX → Crypto
# ============================================================

with tab_overview:
    st.subheader("🏠 High-Level Overview")
    st.write(
        "Quick visual snapshot of **ETFs, Metals, FX and Crypto** over your chosen "
        f"history window (~{assets_years}y). Each chart is interactive (zoom, pan, hover). "
        "Use the other tabs for deeper dives."
    )

    # ---- ETF SNAPSHOT (TOP) ----
    st.markdown("### 📊 ETFs – IVV, SCHG, SPY, ACWX (Stooq, EOD)")

    if etf_error:
        st.error(f"ETF loader error: {etf_error}")
    elif not etf_data_bundle or not etf_data_bundle.get("data"):
        err_list = etf_data_bundle["errors"] if etf_data_bundle else []
        if err_list:
            st.error("All ETF symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list))
        else:
            st.warning("No ETF data loaded.")
    else:
        etf_data = etf_data_bundle["data"]

        # Build normalised performance (start = 100)
        norm_rows = []
        for label, df in etf_data.items():
            if df.empty:
                continue
            base = df["close"].iloc[0]
            tmp = df.copy()
            tmp["index_value"] = 100.0 * tmp["close"] / base
            tmp["label"] = label
            norm_rows.append(tmp[["date", "index_value", "label"]])

        if not norm_rows:
            st.warning("ETF series are empty after filtering.")
        else:
            norm_df = pd.concat(norm_rows, ignore_index=True)

            cols = st.columns(len(etf_data))
            for i, (label, df) in enumerate(etf_data.items()):
                latest = df["close"].iloc[-1]
                cols[i].metric(label, f"${latest:,.2f}")

            fig_etf = px.line(
                norm_df,
                x="date",
                y="index_value",
                color="label",
                title=f"Normalised ETF Performance (start = 100, ~{assets_years}y window)",
                labels={
                    "date": "Date",
                    "index_value": "Index (start = 100)",
                    "label": "ETF",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_etf.update_layout(
                **BASE_LAYOUT,
                xaxis_title="Date",
                yaxis_title="Normalised price (start = 100)",
            )
            if use_log_scale:
                fig_etf.update_yaxes(type="log")
            fig_etf.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_etf, width="stretch", key="overview_etf_norm")

    st.markdown("---")

    # ---- METALS SNAPSHOT (GOLD / SILVER / PLATINUM) ----
    st.markdown("### 🥇 Metals – Gold, Silver, Platinum (Stooq, EOD)")

    if metals_error:
        st.error(f"Metals loader error: {metals_error}")
    elif not metals_data_bundle or not metals_data_bundle.get("data"):
        err_list = metals_data_bundle["errors"] if metals_data_bundle else []
        if err_list:
            st.error("All metals symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list))
        else:
            st.warning("No metals data loaded.")
    else:
        metals_data = metals_data_bundle["data"]

        # Metrics row: latest prices
        cols = st.columns(len(metals_data))
        for i, (label, df) in enumerate(metals_data.items()):
            latest = df["close"].iloc[-1]
            cols[i].metric(label, f"${latest:,.2f}")

        # Gold/Silver ratio (using overlapping dates)
        gold_df = None
        silver_df = None
        for label, df in metals_data.items():
            if label.startswith("Gold"):
                gold_df = df
            elif label.startswith("Silver"):
                silver_df = df

        if gold_df is not None and silver_df is not None:
            merged_gs = (
                gold_df.rename(columns={"close": "Gold"})
                .merge(
                    silver_df.rename(columns={"close": "Silver"}),
                    on="date",
                    how="inner",
                )
                .sort_values("date")
            )
            if not merged_gs.empty:
                last_row = merged_gs.iloc[-1]
                ratio = last_row["Gold"] / last_row["Silver"]
                st.caption(f"Latest Gold / Silver price ratio: **{ratio:.1f} : 1**")

        # Normalised performance for metals
        norm_rows = []
        for label, df in metals_data.items():
            if df.empty:
                continue
            base = df["close"].iloc[0]
            tmp = df.copy()
            tmp["index_value"] = 100.0 * tmp["close"] / base
            tmp["label"] = label
            norm_rows.append(tmp[["date", "index_value", "label"]])

        if norm_rows:
            metals_norm_df = pd.concat(norm_rows, ignore_index=True)

            fig_metals = px.line(
                metals_norm_df,
                x="date",
                y="index_value",
                color="label",
                title=f"Normalised Metals Performance (start = 100, ~{assets_years}y window)",
                labels={
                    "date": "Date",
                    "index_value": "Index (start = 100)",
                    "label": "Metal",
                },
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig_metals.update_layout(
                **BASE_LAYOUT,
                xaxis_title="Date",
                yaxis_title="Normalised price (start = 100)",
            )
            if use_log_scale:
                fig_metals.update_yaxes(type="log")
            fig_metals.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_metals, width="stretch", key="overview_metals_norm")

    st.markdown("---")

    # ---- FX SNAPSHOT (USD → AUD focus) ----
    st.markdown("### 💱 FX – USD → AUD (Frankfurter, live)")

    if fx_error:
        st.error(f"FX data error: {fx_error}")
    elif fx_data is None or fx_data.empty:
        st.warning("No FX data loaded.")
    else:
        start_display = fx_data["date"].min().date()
        end_display = fx_data["date"].max().date()
        st.caption(f"Date range: {start_display} → {end_display}")

        latest_row = fx_data.iloc[-1]
        latest_rate = latest_row.get("AUD", np.nan)
        min_rate = fx_data["rate"].min()
        max_rate = fx_data["rate"].max()

        c1, c2, c3 = st.columns(3)
        if not np.isnan(latest_rate):
            c1.metric("USD → AUD (last)", f"{latest_rate:.4f}")
        c2.metric("Min in range (AUD)", f"{min_rate:.4f}")
        c3.metric("Max in range (AUD)", f"{max_rate:.4f}")

        fx_data["ma_7"] = fx_data["rate"].rolling(7).mean()

        fig_fx = go.Figure()
        fig_fx.add_trace(
            go.Scatter(
                x=fx_data["date"],
                y=fx_data["rate"],
                mode="lines",
                name="USD → AUD",
                line=dict(color=PRIMARY_COLOR, width=1.6),
            )
        )
        fig_fx.add_trace(
            go.Scatter(
                x=fx_data["date"],
                y=fx_data["ma_7"],
                mode="lines",
                name="7-day MA (AUD)",
                line=dict(color=ACCENT_COLOR, width=2.0),
            )
        )
        fig_fx.update_layout(
            **BASE_LAYOUT,
            title="USD → AUD (daily with 7-day moving average)",
            xaxis_title="Date",
            yaxis_title="Rate (AUD per 1 USD)",
        )
        if use_log_scale:
            fig_fx.update_yaxes(type="log")
        fig_fx.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_fx, width="stretch", key="overview_fx_aud")

    st.markdown("---")

    # ---- CRYPTO SNAPSHOT (LAST) ----
    st.markdown("### ₿ Crypto – BTC & ETH (CoinGecko)")

    if crypto_error:
        st.error(f"Crypto data error: {crypto_error}")
    elif not crypto_data or "BTC" not in crypto_data or "ETH" not in crypto_data:
        st.warning("No crypto data loaded.")
    else:
        btc = crypto_data["BTC"].rename(columns={"price": "BTC"})
        eth = crypto_data["ETH"].rename(columns={"price": "ETH"})

        merged = (
            btc.merge(eth, on="date", how="inner")
            .sort_values("date")
            .reset_index(drop=True)
        )

        latest_btc = merged["BTC"].iloc[-1]
        latest_eth = merged["ETH"].iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("BTC-USD (last)", f"${latest_btc:,.0f}")
        c2.metric("ETH-USD (last)", f"${latest_eth:,.0f}")
        c3.metric("BTC/ETH ratio", f"{(latest_btc / latest_eth):.2f}")

        fig_crypto = go.Figure()
        fig_crypto.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["BTC"],
                mode="lines",
                name="BTC-USD",
                line=dict(color=PRIMARY_COLOR, width=1.8),
            )
        )
        fig_crypto.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["ETH"],
                mode="lines",
                name="ETH-USD",
                line=dict(color=ACCENT_COLOR, width=1.8),
            )
        )
        fig_crypto.update_layout(
            **BASE_LAYOUT,
            title=f"BTC vs ETH (price in USD, ~{assets_years}y window)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        if use_log_scale:
            fig_crypto.update_yaxes(type="log")
        fig_crypto.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_crypto, width="stretch", key="overview_crypto_btc_eth")


# ============================================================
# 8. DETAILED FX TAB
# ============================================================

with tab_fx:
    st.subheader("💱 FX – USD vs AUD / EUR / GBP / JPY (Frankfurter, live)")

    if fx_error:
        st.error(f"FX data error: {fx_error}")
    elif fx_data is None or fx_data.empty:
        st.warning("No FX data loaded.")
    else:
        start_display = fx_data["date"].min().date()
        end_display = fx_data["date"].max().date()
        st.markdown(f"**Date range:** {start_display} → {end_display}")

        # Latest snapshot for four majors vs USD
        latest_row = fx_data.iloc[-1]
        latest_aud = latest_row.get("AUD", np.nan)
        latest_eur = latest_row.get("EUR", np.nan)
        latest_gbp = latest_row.get("GBP", np.nan)
        latest_jpy = latest_row.get("JPY", np.nan)

        col1, col2, col3, col4 = st.columns(4)
        if not np.isnan(latest_aud):
            col1.metric("USD → AUD (last)", f"{latest_aud:.4f}")
        if not np.isnan(latest_eur):
            col2.metric("USD → EUR (last)", f"{latest_eur:.4f}")
        if not np.isnan(latest_gbp):
            col3.metric("USD → GBP (last)", f"{latest_gbp:.4f}")
        if not np.isnan(latest_jpy):
            col4.metric("USD → JPY (last)", f"{latest_jpy:.2f}")

        # AUD-focused stats
        min_rate = fx_data["rate"].min()
        max_rate = fx_data["rate"].max()
        last_7_avg = fx_data["rate"].tail(7).mean()

        col5, col6, col7 = st.columns(3)
        col5.metric("Min USD → AUD", f"{min_rate:.4f}")
        col6.metric("Max USD → AUD", f"{max_rate:.4f}")
        col7.metric("7-day avg (AUD)", f"{last_7_avg:.4f}")

        fx_data["ma_7"] = fx_data["rate"].rolling(7).mean()
        fx_data["ma_30"] = fx_data["rate"].rolling(30).mean()

        st.markdown("### USD → AUD – Price Over Time (Interactive)")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fx_data["date"],
                y=fx_data["rate"],
                mode="lines+markers",
                name="USD → AUD",
                line=dict(color=PRIMARY_COLOR, width=1.6),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fx_data["date"],
                y=fx_data["ma_7"],
                mode="lines",
                name="7-day MA",
                line=dict(color=ACCENT_COLOR, width=2.2),
            )
        )

        fig.update_layout(
            **BASE_LAYOUT,
            title="USD → AUD Exchange Rate (with 7-day moving average)",
            xaxis_title="Date",
            yaxis_title="Rate (AUD per 1 USD)",
        )
        if use_log_scale:
            fig.update_yaxes(type="log")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, width="stretch", key="fx_aud_detail_price")

        st.markdown("### Daily % Change (Volatility Histogram – USD → AUD)")
        changes = fx_data["change_pct"].dropna()

        fig_hist = px.histogram(
            changes,
            nbins=30,
            labels={"value": "Daily Change (%)"},
            title="Distribution of Daily FX Changes (USD → AUD)",
        )
        fig_hist.update_traces(marker_line_color="black", marker_line_width=0.5)
        fig_hist.update_layout(
            **BASE_LAYOUT,
            xaxis_title="Daily Change (%)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_hist, width="stretch", key="fx_aud_detail_hist")

        # Multi-currency normalised chart
        st.markdown("### Multi-Currency – USD vs AUD / EUR / GBP / JPY (Normalised)")

        fx_norm_rows = []
        for cur in ["AUD", "EUR", "GBP", "JPY"]:
            if cur in fx_data.columns:
                series = fx_data[["date", cur]].dropna()
                if series.empty:
                    continue
                base = series[cur].iloc[0]
                tmp = series.copy()
                tmp["index_value"] = 100.0 * tmp[cur] / base
                tmp["label"] = cur
                fx_norm_rows.append(tmp[["date", "index_value", "label"]])

        if fx_norm_rows:
            fx_norm_df = pd.concat(fx_norm_rows, ignore_index=True)
            fig_fx_multi = px.line(
                fx_norm_df,
                x="date",
                y="index_value",
                color="label",
                title=f"FX: USD vs AUD/EUR/GBP/JPY (Normalised to 100, ~{assets_years}y window)",
                labels={
                    "date": "Date",
                    "index_value": "Index (start = 100)",
                    "label": "Currency",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_fx_multi.update_layout(
                **BASE_LAYOUT,
                xaxis_title="Date",
                yaxis_title="Index (start = 100)",
            )
            if use_log_scale:
                fig_fx_multi.update_yaxes(type="log")
            fig_fx_multi.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_fx_multi, width="stretch", key="fx_multi_detail")

        with st.expander("Show raw FX data"):
            st.dataframe(fx_data.reset_index(drop=True))


# ============================================================
# 9. DETAILED CRYPTO TAB
# ============================================================

with tab_crypto:
    st.subheader("₿ Crypto – BTC & ETH (USD, via CoinGecko – Keyless where possible)")

    if crypto_error:
        st.error(f"Crypto data error: {crypto_error}")
    elif not crypto_data or "BTC" not in crypto_data or "ETH" not in crypto_data:
        st.warning("No crypto data loaded.")
    else:
        btc = crypto_data["BTC"].rename(columns={"price": "BTC"})
        eth = crypto_data["ETH"].rename(columns={"price": "ETH"})

        merged = (
            btc.merge(eth, on="date", how="inner")
            .sort_values("date")
            .reset_index(drop=True)
        )

        merged["BTC_change_pct"] = merged["BTC"].pct_change() * 100
        merged["ETH_change_pct"] = merged["ETH"].pct_change() * 100

        latest_btc = merged["BTC"].iloc[-1]
        latest_eth = merged["ETH"].iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("BTC-USD (last)", f"${latest_btc:,.0f}")
        c2.metric("ETH-USD (last)", f"${latest_eth:,.0f}")
        c3.metric("BTC/ETH price ratio", f"{(latest_btc / latest_eth):.2f}")

        st.markdown(f"### BTC & ETH – Price Over Time (~{assets_years}y window)")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["BTC"],
                mode="lines",
                name="BTC-USD",
                line=dict(color=PRIMARY_COLOR, width=1.8),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["ETH"],
                mode="lines",
                name="ETH-USD",
                line=dict(color=ACCENT_COLOR, width=1.8),
            )
        )

        fig.update_layout(
            **BASE_LAYOUT,
            title=f"BTC vs ETH (CoinGecko, ~{assets_years}y window)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        if use_log_scale:
            fig.update_yaxes(type="log")
        fig.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig, width="stretch", key="crypto_detail_price")

        st.markdown("### Daily % Change – BTC vs ETH")

        fig_change = go.Figure()
        fig_change.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["BTC_change_pct"],
                mode="lines",
                name="BTC % change",
                line=dict(color=PRIMARY_COLOR, width=1.4),
            )
        )
        fig_change.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["ETH_change_pct"],
                mode="lines",
                name="ETH % change",
                line=dict(color=ACCENT_COLOR, width=1.4),
            )
        )

        fig_change.update_layout(
            **BASE_LAYOUT,
            title="Daily Percentage Change – BTC & ETH",
            xaxis_title="Date",
            yaxis_title="Daily Change (%)",
        )
        st.plotly_chart(fig_change, width="stretch", key="crypto_detail_change")

        with st.expander("Show merged BTC/ETH data"):
            st.dataframe(merged.reset_index(drop=True))


# ============================================================
# 10. DETAILED ETF TAB – REAL STOOQ DATA
# ============================================================

with tab_etf:
    st.subheader("📊 ETFs – IVV, SCHG, SPY, ACWX (Stooq, keyless EOD)")

    if etf_error:
        st.error(f"ETF loader error: {etf_error}")
    elif not etf_data_bundle or not etf_data_bundle.get("data"):
        err_list = etf_data_bundle["errors"] if etf_data_bundle else []
        if err_list:
            st.error("All ETF symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list))
        else:
            st.warning("No ETF data loaded.")
    else:
        etf_data = etf_data_bundle["data"]
        err_list = etf_data_bundle["errors"]

        if err_list:
            st.warning(
                "Some ETF symbols failed to load:\n\n" + "\n".join(f"- {e}" for e in err_list)
            )

        # Normalised performance (start = 100)
        norm_rows = []
        for label, df in etf_data.items():
            if df.empty:
                continue
            base = df["close"].iloc[0]
            tmp = df.copy()
            tmp["index_value"] = 100.0 * tmp["close"] / base
            tmp["label"] = label
            norm_rows.append(tmp[["date", "index_value", "label"]])

        if not norm_rows:
            st.warning("ETF time series are empty after filtering.")
        else:
            norm_df = pd.concat(norm_rows, ignore_index=True)

            cols = st.columns(len(etf_data))
            for i, (label, df) in enumerate(etf_data.items()):
                latest = df["close"].iloc[-1]
                cols[i].metric(label, f"${latest:,.2f}")

            st.markdown(f"### Normalised Performance (Start = 100, ~{assets_years}y window)")

            fig = px.line(
                norm_df,
                x="date",
                y="index_value",
                color="label",
                title=f"ETF Performance (Normalised to 100 at Start of Period, ~{assets_years}y)",
                labels={
                    "date": "Date",
                    "index_value": "Index (start = 100)",
                    "label": "ETF",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig.update_layout(
                **BASE_LAYOUT,
                xaxis_title="Date",
                yaxis_title="Normalised price (start = 100)",
            )
            if use_log_scale:
                fig.update_yaxes(type="log")
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, width="stretch", key="etf_detail_norm")

        # Extra chart: correlation heatmap of ETF daily returns
        st.markdown("### Correlation of Daily Returns (Investor View)")

        returns_df = None
        for label, df in etf_data.items():
            if df.empty:
                continue
            tmp = df[["date", "close"]].copy()
            tmp[label] = tmp["close"].pct_change()
            tmp = tmp[["date", label]]
            if returns_df is None:
                returns_df = tmp
            else:
                returns_df = returns_df.merge(tmp, on="date", how="inner")

        if returns_df is not None and not returns_df.empty:
            corr = returns_df.drop(columns=["date"]).corr()
            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                title="Correlation Matrix of ETF Daily Returns",
                labels=dict(color="Correlation"),
            )
            fig_corr.update_layout(**BASE_LAYOUT)
            st.plotly_chart(fig_corr, width="stretch", key="etf_corr_matrix")

        with st.expander("Show raw ETF data"):
            for label, df in etf_data.items():
                st.markdown(f"**{label}**")
                st.dataframe(df.reset_index(drop=True))


# ============================================================
# 11. DETAILED METALS TAB – REAL STOOQ DATA
# ============================================================

with tab_metals:
    st.subheader("🥇 Metals – Gold, Silver, Platinum (Stooq, keyless EOD)")

    st.caption(
        "Gold, Silver, and Platinum spot-like series fetched from Stooq as FX-style pairs:\n"
        "- **XAUUSD** (Gold / USD)\n"
        "- **XAGUSD** (Silver / USD)\n"
        "- **XPTUSD** (Platinum / USD)\n"
        "These are still EOD prices, not tick-level data."
    )

    if metals_error:
        st.error(f"Metals loader error: {metals_error}")
    elif not metals_data_bundle or not metals_data_bundle.get("data"):
        err_list = metals_data_bundle["errors"] if metals_data_bundle else []
        if err_list:
            st.error("All metals symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list))
        else:
            st.warning("No metals data loaded.")
    else:
        metals_data = metals_data_bundle["data"]
        err_list = metals_data_bundle["errors"]

        if err_list:
            st.warning(
                "Some metals symbols failed to load:\n\n" + "\n".join(f"- {e}" for e in err_list)
            )

        # Metrics
        cols = st.columns(len(metals_data))
        for i, (label, df) in enumerate(metals_data.items()):
            latest = df["close"].iloc[-1]
            cols[i].metric(label, f"${latest:,.2f}")

        # Combined price chart
        st.markdown(f"### Metals – Price Over Time (USD, ~{assets_years}y window)")

        price_rows = []
        for label, df in metals_data.items():
            tmp = df.copy()
            tmp["label"] = label
            price_rows.append(tmp[["date", "close", "label"]])

        if price_rows:
            price_df = pd.concat(price_rows, ignore_index=True)

            fig = px.line(
                price_df,
                x="date",
                y="close",
                color="label",
                title=f"Metals Price (USD, ~{assets_years}y window)",
                labels={"date": "Date", "close": "Price (USD)", "label": "Metal"},
                color_discrete_sequence=px.colors.qualitative.Set1,
            )
            fig.update_layout(
                **BASE_LAYOUT,
                xaxis_title="Date",
                yaxis_title="Price (USD)",
            )
            if use_log_scale:
                fig.update_yaxes(type="log")
            fig.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig, width="stretch", key="metals_detail_price")


        # Gold/Silver ratio
        gold_df = None
        silver_df = None
        for label, df in metals_data.items():
            if label.startswith("Gold"):
                gold_df = df
            elif label.startswith("Silver"):
                silver_df = df

        if gold_df is not None and silver_df is not None:
            merged_gs = (
                gold_df.rename(columns={"close": "Gold"})
                .merge(
                    silver_df.rename(columns={"close": "Silver"}),
                    on="date",
                    how="inner",
                )
                .sort_values("date")
            )
            if not merged_gs.empty:
                merged_gs["ratio"] = merged_gs["Gold"] / merged_gs["Silver"]

                st.markdown("### Gold / Silver Ratio Over Time")

                fig_ratio = px.line(
                    merged_gs,
                    x="date",
                    y="ratio",
                    title="Gold / Silver Price Ratio",
                    labels={"date": "Date", "ratio": "Gold ÷ Silver"},
                )
                fig_ratio.update_layout(
                    **BASE_LAYOUT,
                    xaxis_title="Date",
                    yaxis_title="Ratio",
                )
                if use_log_scale:
                    fig_ratio.update_yaxes(type="log")
                st.plotly_chart(fig_ratio, width="stretch", key="metals_detail_ratio")

        with st.expander("Show raw metals data"):
            for label, df in metals_data.items():
                st.markdown(f"**{label}**")
                st.dataframe(df.reset_index(drop=True))

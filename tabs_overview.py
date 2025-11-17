import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PRIMARY_COLOR, ACCENT_COLOR, BASE_LAYOUT


def render_overview(
    fx_data,
    fx_error,
    crypto_data,
    crypto_error,
    etf_data_bundle,
    etf_error,
    metals_data_bundle,
    metals_error,
    assets_years: int,
    use_log_scale: bool,
) -> None:
    st.subheader("🏠 High-Level Overview")
    st.write(
        "Quick visual snapshot of **ETFs, Metals, FX and Crypto** over your chosen "
        f"history window (~{assets_years}y). Each chart is interactive (zoom, pan, hover). "
        "Use the other tabs for deeper dives."
    )

    # --- ETF SNAPSHOT ---
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

    # --- METALS SNAPSHOT ---
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

        cols = st.columns(len(metals_data))
        for i, (label, df) in enumerate(metals_data.items()):
            latest = df["close"].iloc[-1]
            cols[i].metric(label, f"${latest:,.2f}")

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

    # --- FX SNAPSHOT ---
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

        fx_plot = fx_data.copy()
        fx_plot["ma_7"] = fx_plot["rate"].rolling(7).mean()

        fig_fx = go.Figure()
        fig_fx.add_trace(
            go.Scatter(
                x=fx_plot["date"],
                y=fx_plot["rate"],
                mode="lines",
                name="USD → AUD",
                line=dict(color=PRIMARY_COLOR, width=1.6),
            )
        )
        fig_fx.add_trace(
            go.Scatter(
                x=fx_plot["date"],
                y=fx_plot["ma_7"],
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

    # --- CRYPTO SNAPSHOT ---
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


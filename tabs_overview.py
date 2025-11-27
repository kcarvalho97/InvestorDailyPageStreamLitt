"""
Overview + FX tabs for the Markets Dashboard.

This module defines:
- render_overview: main investor dashboard (ETFs, Metals, FX, Crypto)
- render_fx_tab: detailed FX tab

Design tweaks for mobile:
- All charts use use_container_width=True
- Heights kept moderate for readability on phones and desktops
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import (
    BASE_LAYOUT,
    PRIMARY_COLOR,
    ACCENT_COLOR,
    POSITIVE_COLOR,
    NEGATIVE_COLOR,
)

DEFAULT_FIG_HEIGHT = 420


def _get_fx_latest_snapshot(fx_data: pd.DataFrame) -> Dict[str, float]:
    """Return latest FX values for the main currencies if available."""
    latest_row = fx_data.iloc[-1]
    return {
        "AUD": latest_row.get("AUD", np.nan),
        "EUR": latest_row.get("EUR", np.nan),
        "GBP": latest_row.get("GBP", np.nan),
        "JPY": latest_row.get("JPY", np.nan),
        "CNY": latest_row.get("CNY", np.nan),
        "INR": latest_row.get("INR", np.nan),
    }


def render_overview(
    *,
    fx_data: Optional[pd.DataFrame],
    fx_error: Optional[str],
    crypto_data: Dict[str, pd.DataFrame],
    crypto_error: Optional[str],
    etf_bundle: Optional[Dict],
    etf_error: Optional[str],
    metals_bundle: Optional[Dict],
    metals_error: Optional[str],
    assets_years: int,
    use_log_scale: bool,
) -> None:
    """
    Render the main Overview tab.

    Layout:
    - ETFs top
    - Metals
    - FX snapshot (USD→AUD focus)
    - Crypto (BTC & ETH)
    """
    st.subheader("🏠 High-Level Overview")
    st.write(
        "Quick visual snapshot of **ETFs, Metals, FX and Crypto** over your chosen "
        f"history window (~{assets_years}y). Use the other tabs for deeper dives."
    )

    # ========================================================
    # ETFs – TOP SECTION
    # ========================================================
    # Build header string from actual loaded labels
    header_text = "### 📊 ETFs – (No Data)"
    if etf_bundle and etf_bundle.get("data"):
        tickers = [k.split()[0] for k in etf_bundle["data"].keys()]
        header_text = f"### 📊 ETFs – {', '.join(tickers)} (Yahoo Finance, EOD)"
    st.markdown(header_text)

    if etf_error:
        st.error(f"ETF loader error: {etf_error}")
    elif not etf_bundle or not etf_bundle.get("data"):
        err_list = etf_bundle["errors"] if etf_bundle else []
        if err_list:
            st.error("All ETF symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list))
        else:
            st.warning("No ETF data loaded.")
    else:
        etf_data = etf_bundle["data"]

        # Normalised performance (start = 100)
        norm_rows = []
        for label, df in etf_data.items():
            if df is None or df.empty:
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

            # Metrics row – will stack on mobile
            cols = st.columns(len(etf_data))
            for i, (label, df) in enumerate(etf_data.items()):
                if df is None or df.empty:
                    cols[i].metric(label, "n/a")
                else:
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
                height=DEFAULT_FIG_HEIGHT,
                xaxis_title="Date",
                yaxis_title="Normalised price (start = 100)",
            )
            if use_log_scale:
                fig_etf.update_yaxes(type="log")
            fig_etf.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_etf, use_container_width=True, key="overview_etf_norm")

    st.markdown("---")

    # ========================================================
    # METALS – GOLD / SILVER / PLATINUM
    # ========================================================
    st.markdown("### 🥇 Metals – Gold, Silver, Platinum (Yahoo Finance, EOD)")

    if metals_error:
        st.error(f"Metals loader error: {metals_error}")
    elif not metals_bundle or not metals_bundle.get("data"):
        err_list = metals_bundle["errors"] if metals_bundle else []
        if err_list:
            st.error("All metals symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list))
        else:
            st.warning("No metals data loaded.")
    else:
        metals_data = metals_bundle["data"]

        cols = st.columns(len(metals_data))
        for i, (label, df) in enumerate(metals_data.items()):
            if df is None or df.empty:
                cols[i].metric(label, "n/a")
            else:
                latest = df["close"].iloc[-1]
                cols[i].metric(label, f"${latest:,.2f}")

        # Gold/Silver ratio if both present
        gold_df = None
        silver_df = None
        for label, df in metals_data.items():
            if "Gold" in label:
                gold_df = df
            elif "Silver" in label:
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
            if df is None or df.empty:
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
                height=DEFAULT_FIG_HEIGHT,
                xaxis_title="Date",
                yaxis_title="Normalised price (start = 100)",
            )
            if use_log_scale:
                fig_metals.update_yaxes(type="log")
            fig_metals.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_metals, use_container_width=True, key="overview_metals_norm")

    st.markdown("---")

    # ========================================================
    # FX SNAPSHOT – USD → AUD + OTHERS
    # ========================================================
    st.markdown("### 💱 FX – USD → AUD (plus EUR / GBP / JPY / CNY / INR)")

    if fx_error:
        st.error(f"FX data error: {fx_error}")
    elif fx_data is None or fx_data.empty:
        st.warning("No FX data loaded.")
    else:
        start_display = fx_data["date"].min().date()
        end_display = fx_data["date"].max().date()
        st.caption(f"Date range: {start_display} → {end_display}")

        latest = _get_fx_latest_snapshot(fx_data)

        # First line: AUD, EUR, GBP
        c1, c2, c3 = st.columns(3)
        if not np.isnan(latest["AUD"]):
            c1.metric("USD → AUD (last)", f"{latest['AUD']:.4f}")
        if not np.isnan(latest["EUR"]):
            c2.metric("USD → EUR (last)", f"{latest['EUR']:.4f}")
        if not np.isnan(latest["GBP"]):
            c3.metric("USD → GBP (last)", f"{latest['GBP']:.4f}")

        # Second line: JPY, CNY, INR
        c4, c5, c6 = st.columns(3)
        if not np.isnan(latest["JPY"]):
            c4.metric("USD → JPY (last)", f"{latest['JPY']:.2f}")
        if not np.isnan(latest["CNY"]):
            c5.metric("USD → CNY (last)", f"{latest['CNY']:.4f}")
        if not np.isnan(latest["INR"]):
            c6.metric("USD → INR (last)", f"{latest['INR']:.4f}")

        # AUD-focused stats
        min_rate = fx_data["rate"].min()
        max_rate = fx_data["rate"].max()
        last_7_avg = fx_data["rate"].tail(7).mean()

        col5, col6, col7 = st.columns(3)
        col5.metric("Min USD → AUD", f"{min_rate:.4f}")
        col6.metric("Max USD → AUD", f"{max_rate:.4f}")
        col7.metric("7-day avg (AUD)", f"{last_7_avg:.4f}")

        fx_data = fx_data.copy()
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
            height=DEFAULT_FIG_HEIGHT,
            title="USD → AUD (daily with 7-day moving average)",
            xaxis_title="Date",
            yaxis_title="Rate (AUD per 1 USD)",
        )
        if use_log_scale:
            fig_fx.update_yaxes(type="log")
        fig_fx.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_fx, use_container_width=True, key="overview_fx_aud")

    st.markdown("---")

    # ========================================================
    # CRYPTO SNAPSHOT – BTC & ETH
    # ========================================================
    st.markdown("### ₿ Crypto – BTC & ETH (Yahoo Finance)")

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
            height=DEFAULT_FIG_HEIGHT,
            title=f"BTC vs ETH (price in USD, ~{assets_years}y window)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
        )
        if use_log_scale:
            fig_crypto.update_yaxes(type="log")
        fig_crypto.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_crypto, use_container_width=True, key="overview_crypto_btc_eth")


def render_fx_tab(
    *,
    fx_data: Optional[pd.DataFrame],
    fx_error: Optional[str],
    assets_years: int,
    use_log_scale: bool,
) -> None:
    """
    Detailed FX tab, with:
    - Multi-currency metrics
    - USD→AUD price chart with moving averages
    - Histogram of daily % changes
    - Normalised multi-currency chart
    """
    st.subheader("💱 FX – USD vs AUD / EUR / GBP / JPY / CNY / INR (Frankfurter, live)")

    if fx_error:
        st.error(f"FX data error: {fx_error}")
        return
    if fx_data is None or fx_data.empty:
        st.warning("No FX data loaded.")
        return

    fx_data = fx_data.copy()
    start_display = fx_data["date"].min().date()
    end_display = fx_data["date"].max().date()
    st.markdown(f"**Date range:** {start_display} → {end_display}")

    latest = _get_fx_latest_snapshot(fx_data)

    col1, col2, col3 = st.columns(3)
    if not np.isnan(latest["AUD"]):
        col1.metric("USD → AUD (last)", f"{latest['AUD']:.4f}")
    if not np.isnan(latest["EUR"]):
        col2.metric("USD → EUR (last)", f"{latest['EUR']:.4f}")
    if not np.isnan(latest["GBP"]):
        col3.metric("USD → GBP (last)", f"{latest['GBP']:.4f}")

    col4, col5, col6 = st.columns(3)
    if not np.isnan(latest["JPY"]):
        col4.metric("USD → JPY (last)", f"{latest['JPY']:.2f}")
    if not np.isnan(latest["CNY"]):
        col5.metric("USD → CNY (last)", f"{latest['CNY']:.4f}")
    if not np.isnan(latest["INR"]):
        col6.metric("USD → INR (last)", f"{latest['INR']:.4f}")

    # AUD-focused stats
    min_rate = fx_data["rate"].min()
    max_rate = fx_data["rate"].max()
    last_7_avg = fx_data["rate"].tail(7).mean()

    col7, col8, col9 = st.columns(3)
    col7.metric("Min USD → AUD", f"{min_rate:.4f}")
    col8.metric("Max USD → AUD", f"{max_rate:.4f}")
    col9.metric("7-day avg (AUD)", f"{last_7_avg:.4f}")

    # Moving averages
    fx_data["ma_7"] = fx_data["rate"].rolling(7).mean()
    fx_data["ma_30"] = fx_data["rate"].rolling(30).mean()

    st.markdown("### USD → AUD – Price Over Time")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=fx_data["date"],
            y=fx_data["rate"],
            mode="lines+markers",
            name="USD → AUD",
            line=dict(color=PRIMARY_COLOR, width=1.6),
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fx_data["date"],
            y=fx_data["ma_7"],
            mode="lines",
            name="7-day MA",
            line=dict(color=ACCENT_COLOR, width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=fx_data["date"],
            y=fx_data["ma_30"],
            mode="lines",
            name="30-day MA",
            line=dict(color="gray", width=1.5, dash="dash"),
        )
    )

    fig.update_layout(
        **BASE_LAYOUT,
        height=DEFAULT_FIG_HEIGHT,
        title="USD → AUD Exchange Rate (with 7- and 30-day moving averages)",
        xaxis_title="Date",
        yaxis_title="Rate (AUD per 1 USD)",
    )
    if use_log_scale:
        fig.update_yaxes(type="log")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True, key="fx_aud_detail_price")

    # Histogram of daily % changes
    st.markdown("### Daily % Change (USD → AUD)")

    changes = fx_data["change_pct"].dropna()
    if not changes.empty:
        fig_hist = px.histogram(
            changes,
            nbins=30,
            labels={"value": "Daily Change (%)"},
            title="Distribution of Daily FX Changes (USD → AUD)",
        )
        fig_hist.update_traces(marker_line_color="black", marker_line_width=0.5)
        fig_hist.update_layout(
            **BASE_LAYOUT,
            height=DEFAULT_FIG_HEIGHT,
            xaxis_title="Daily Change (%)",
            yaxis_title="Frequency",
        )
        st.plotly_chart(fig_hist, use_container_width=True, key="fx_aud_detail_hist")

    # Multi-currency normalised chart
    st.markdown("### Multi-Currency – USD vs AUD / EUR / GBP / JPY / CNY / INR (Normalised)")

    fx_norm_rows = []
    for cur in ["AUD", "EUR", "GBP", "JPY", "CNY", "INR"]:
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
            title=f"FX: USD vs AUD/EUR/GBP/JPY/CNY/INR (Normalised to 100, ~{assets_years}y window)",
            labels={
                "date": "Date",
                "index_value": "Index (start = 100)",
                "label": "Currency",
            },
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_fx_multi.update_layout(
            **BASE_LAYOUT,
            height=DEFAULT_FIG_HEIGHT,
            xaxis_title="Date",
            yaxis_title="Index (start = 100)",
        )
        if use_log_scale:
            fig_fx_multi.update_yaxes(type="log")
        fig_fx_multi.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_fx_multi, use_container_width=True, key="fx_multi_detail")

    with st.expander("Show raw FX data"):
        st.dataframe(fx_data.reset_index(drop=True))

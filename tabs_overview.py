"""
Overview tab for the Markets Dashboard.

Shows:
- Hero banner with ETF/FX/crypto snapshot
- ETF snapshot (normalised performance)
- Metals snapshot (normalised) + simple ETF performance bar
- FX snapshot (USD→AUD focus + multi-currency normalised)
- Crypto snapshot (BTC & ETH)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from config import (
    BASE_LAYOUT,
    PRIMARY_COLOR,
    ACCENT_COLOR,
)


def _compute_etf_stats(etf_data_bundle: dict | None):
    """Compute simple per-ETF stats used in the hero & bar chart.

    Returns a dict with keys:
      - per_label: {label: {pct, vol, last}}
      - best: {label, pct}
      - worst: {label, pct}
      - composite_return: float
      - most_volatile: {label, vol}   (optional)
    """
    if not etf_data_bundle or "data" not in etf_data_bundle:
        return None

    data = etf_data_bundle["data"]
    per_label: dict[str, dict] = {}

    for label, df in data.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        closes = pd.to_numeric(df["close"], errors="coerce")
        closes = closes.dropna()
        if closes.empty:
            continue

        start = closes.iloc[0]
        end = closes.iloc[-1]
        if start == 0 or not np.isfinite(start) or not np.isfinite(end):
            continue

        pct = float((end / start - 1.0) * 100.0)
        rets = closes.pct_change().dropna()
        vol = float(rets.std() * 100.0) if not rets.empty else None

        per_label[label] = {"pct": pct, "vol": vol, "last": float(end)}

    if not per_label:
        return None

    # Best / worst by total % return
    best_label = max(per_label, key=lambda k: per_label[k]["pct"])
    worst_label = min(per_label, key=lambda k: per_label[k]["pct"])

    composite_return = (
        sum(v["pct"] for v in per_label.values()) / float(len(per_label))
    )

    # Most volatile by daily std-dev of returns
    vol_map = {k: v["vol"] for k, v in per_label.items() if v["vol"] is not None}
    most_volatile = None
    if vol_map:
        mv_label = max(vol_map, key=vol_map.get)
        most_volatile = {"label": mv_label, "vol": float(vol_map[mv_label])}

    stats: dict[str, object] = {
        "per_label": per_label,
        "best": {"label": best_label, "pct": per_label[best_label]["pct"]},
        "worst": {"label": worst_label, "pct": per_label[worst_label]["pct"]},
        "composite_return": composite_return,
    }
    if most_volatile:
        stats["most_volatile"] = most_volatile
    return stats


def render_overview(
    *,
    fx_data: pd.DataFrame | None,
    fx_error: str | None,
    crypto_data: dict,
    crypto_error: str | None,
    etf_data_bundle: dict | None,
    etf_error: str | None,
    metals_data_bundle: dict | None,
    metals_error: str | None,
    assets_years: int,
    use_log_scale: bool,
) -> None:
    """Render the main Overview tab."""

    st.subheader("🏠 High-Level Overview")

    # Pre-compute ETF stats once for hero + bar chart
    etf_stats = None    # dict with per-label stats etc.
    if not etf_error:
        etf_stats = _compute_etf_stats(etf_data_bundle)

    # ============================================================
    # 0) HERO BANNER
    # ============================================================
    hero_cols = st.columns([2, 2])

    # Left: ETF basket snapshot
    with hero_cols[0]:
        st.markdown("#### 🧭 ETF Basket Snapshot")
        if etf_stats:
            st.metric(
                "Unweighted ETF basket return",
                f"{etf_stats['composite_return']:+.1f}%",
                help=(
                    "Simple average of each ETF's % change over the selected "
                    "window. This is NOT financial advice, just a quick feel "
                    "for how the whole basket moved."
                ),
            )
            best = etf_stats["best"]
            worst = etf_stats["worst"]
            caption = (
                f"Best ETF: **{best['label']}** ({best['pct']:+.1f}%) · "
                f"Worst: **{worst['label']}** ({worst['pct']:+.1f}%)"
            )
            mv = etf_stats.get("most_volatile")
            if mv:
                caption += f" · Most volatile: **{mv['label']}** (σ ≈ {mv['vol']:.1f}%/day)"
            st.caption(caption)
        else:
            st.caption(
                "ETF stats unavailable right now – scroll down to see individual "
                "charts and any error messages."
            )

    # Right: Macro snapshot (FX + BTC)
    with hero_cols[1]:
        st.markdown("#### 🌍 Macro Snapshot")

        # FX quick view
        if fx_error:
            st.caption("FX data temporarily unavailable.")
        elif fx_data is None or fx_data.empty:
            st.caption("No FX data loaded.")
        else:
            latest_fx = fx_data.iloc[-1]
            aud = latest_fx.get("AUD", np.nan)
            change_pct = latest_fx.get("change_pct", np.nan)
            delta_text = ""
            if np.isfinite(change_pct):
                delta_text = f"{change_pct:+.2f}% vs previous day"
            if np.isfinite(aud):
                st.metric("USD → AUD (last)", f"{aud:.4f}", delta_text)

        # BTC quick view
        btc_df = crypto_data.get("BTC")
        if btc_df is None:
            btc_df = crypto_data.get("BTC-USD")

        if crypto_error:
            st.caption("Crypto data temporarily unavailable.")
        elif btc_df is None or btc_df.empty:
            st.caption("No BTC data loaded.")
        else:
            btc_df = btc_df.sort_values("date").reset_index(drop=True)
            last = float(btc_df["price"].iloc[-1])
            delta = ""
            if len(btc_df) > 1:
                prev = float(btc_df["price"].iloc[-2])
                if prev != 0 and np.isfinite(prev):
                    delta_pct = (last / prev - 1.0) * 100.0
                    delta = f"{delta_pct:+.1f}% (1-day)"
            st.metric("BTC-USD (last)", f"${last:,.0f}", delta)

    st.caption(
        f"All hero stats use ~{assets_years}y of history. Open the **☰ View & scale options** "
        "control at the top of the page to change the history window or toggle log scale."
    )

    st.markdown("---")

    # ============================================================
    # 1) ETF SNAPSHOT (NORMALISED)
    # ============================================================
    header_text = "### 📊 ETFs – (no data)"
    if etf_data_bundle and etf_data_bundle.get("data"):
        tickers = [k.split()[0] for k in etf_data_bundle["data"].keys()]
        header_text = f"### 📊 ETFs – {', '.join(tickers)} (Stooq, EOD)"

    st.markdown(header_text)

    if etf_error:
        st.error(f"ETF loader error: {etf_error}")
    elif not etf_data_bundle or not etf_data_bundle.get("data"):
        err_list = etf_data_bundle.get("errors") if etf_data_bundle else []
        if err_list:
            st.error(
                "All ETF symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list)
            )
        else:
            st.warning("No ETF data loaded.")
    else:
        etf_data = etf_data_bundle["data"]

        # Normalised performance (start = 100) + shorter legend labels
        norm_rows = []
        for label, df in etf_data.items():
            if df is None or df.empty:
                continue
            base = df["close"].iloc[0]
            if base == 0 or not np.isfinite(base):
                continue
            tmp = df.copy()
            tmp["index_value"] = 100.0 * tmp["close"] / base
            tmp["full_label"] = label
            tmp["series"] = label.split(" (")[0]  # short legend label e.g. "VOO"
            norm_rows.append(tmp[["date", "index_value", "series", "full_label"]])

        if not norm_rows:
            st.warning("ETF series are empty after filtering.")
        else:
            norm_df = pd.concat(norm_rows, ignore_index=True)

            cols = st.columns(len(etf_data))
            for i, (label, df) in enumerate(etf_data.items()):
                if df is None or df.empty:
                    continue
                latest = df["close"].iloc[-1]
                cols[i].metric(label, f"${latest:,.2f}")

            fig_etf = px.line(
                norm_df,
                x="date",
                y="index_value",
                color="series",
                hover_data={"full_label": True, "series": False},
                title=(
                    f"Normalised ETF performance (start = 100, ~{assets_years}y "
                    "window)"
                ),
                labels={
                    "date": "Date",
                    "index_value": "Index (start = 100)",
                    "series": "ETF",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_etf.update_layout(
                **BASE_LAYOUT,
                xaxis_title="Date",
                yaxis_title="Normalised price (start = 100)",
                legend=dict(
                    orientation="h",
                    y=-0.25,
                    x=0.5,
                    xanchor="center",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10),
                    itemclick="toggleothers",
                    itemdoubleclick="toggle",
                ),
            )
            if use_log_scale:
                fig_etf.update_yaxes(type="log")
            fig_etf.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(fig_etf, use_container_width=True, key="overview_etf_norm")

    st.markdown("---")

    # ============================================================
    # 2) METALS SNAPSHOT + SIMPLE ETF BAR CHART
    # ============================================================
    st.markdown("### 🥇 Metals – Gold, Silver, Platinum (Stooq, EOD)")

    if metals_error:
        st.error(f"Metals loader error: {metals_error}")
    elif not metals_data_bundle or not metals_data_bundle.get("data"):
        err_list = metals_data_bundle.get("errors") if metals_data_bundle else []
        if err_list:
            st.error(
                "All metals symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list)
            )
        else:
            st.warning("No metals data loaded.")
    else:
        metals_data = metals_data_bundle["data"]

        # Metrics row: latest prices
        cols = st.columns(len(metals_data))
        for i, (label, df) in enumerate(metals_data.items()):
            if df is None or df.empty:
                continue
            latest = df["close"].iloc[-1]
            cols[i].metric(label, f"${latest:,.2f}")

        # Normalised metals chart + ETF performance bar chart side by side
        chart_cols = st.columns(2)

        # Left: normalised metals
        norm_rows = []
        for label, df in metals_data.items():
            if df is None or df.empty:
                continue
            base = df["close"].iloc[0]
            if base == 0 or not np.isfinite(base):
                continue
            tmp = df.copy()
            tmp["index_value"] = 100.0 * tmp["close"] / base
            tmp["series"] = label.split(" (")[0]  # "Gold", "Silver", "Platinum"
            norm_rows.append(tmp[["date", "index_value", "series"]])

        with chart_cols[0]:
            if norm_rows:
                metals_norm_df = pd.concat(norm_rows, ignore_index=True)
                fig_metals = px.line(
                    metals_norm_df,
                    x="date",
                    y="index_value",
                    color="series",
                    title=(
                        f"Normalised metals performance (start = 100, "
                        f"~{assets_years}y window)"
                    ),
                    labels={
                        "date": "Date",
                        "index_value": "Index (start = 100)",
                        "series": "Metal",
                    },
                    color_discrete_sequence=px.colors.qualitative.Set1,
                )
                fig_metals.update_layout(
                    **BASE_LAYOUT,
                    xaxis_title="Date",
                    yaxis_title="Normalised price (start = 100)",
                    legend=dict(
                        orientation="h",
                        y=-0.3,
                        x=0.5,
                        xanchor="center",
                        bgcolor="rgba(0,0,0,0)",
                        font=dict(size=10),
                        itemclick="toggleothers",
                        itemdoubleclick="toggle",
                    ),
                )
                if use_log_scale:
                    fig_metals.update_yaxes(type="log")
                fig_metals.update_xaxes(rangeslider_visible=True)
                st.plotly_chart(
                    fig_metals,
                    use_container_width=True,
                    key="overview_metals_norm",
                )
            else:
                st.info("No metals series available for the selected window.")

        # Right: beginner-friendly ETF performance bar chart
        with chart_cols[1]:
            st.markdown("#### Simple ETF performance (window % change)")
            if etf_stats and etf_stats.get("per_label"):
                bar_rows = []
                for label, vals in etf_stats["per_label"].items():
                    short = label.split(" (")[0]
                    bar_rows.append({"ETF": short, "Return_%": vals["pct"]})
                bar_df = pd.DataFrame(bar_rows)
                bar_df.sort_values("Return_%", ascending=False, inplace=True)

                fig_bar = px.bar(
                    bar_df,
                    x="ETF",
                    y="Return_%",
                    text="Return_%",
                    labels={"Return_%": "% change", "ETF": "ETF"},
                    title="Which ETF did best over this window?",
                )
                fig_bar.update_traces(
                    texttemplate="%{text:.1f}%",
                    textposition="outside",
                )
                fig_bar.update_layout(
                    **BASE_LAYOUT,
                    xaxis_title="ETF",
                    yaxis_title="% change over window",
                    showlegend=False,
                    uniformtext_minsize=8,
                    uniformtext_mode="hide",
                )
                st.plotly_chart(
                    fig_bar,
                    use_container_width=True,
                    key="overview_etf_bar",
                )
            else:
                st.info(
                    "ETF data not available for the bar chart – scroll up to see "
                    "any ETF error messages."
                )

    st.markdown("---")

    # ============================================================
    # 3) FX SNAPSHOT – MULTI-CURRENCY
    # ============================================================
    st.markdown(
        "### 💱 FX – USD vs AUD / EUR / GBP / JPY / CNY / INR (Frankfurter, live)"
    )

    if fx_error:
        st.error(f"FX data error: {fx_error}")
    elif fx_data is None or fx_data.empty:
        st.warning("No FX data loaded.")
    else:
        start_display = fx_data["date"].min().date()
        end_display = fx_data["date"].max().date()
        st.caption(f"Date range: {start_display} → {end_display}")

        # Latest snapshot for the main majors (if present)
        latest_row = fx_data.iloc[-1]
        latest_aud = latest_row.get("AUD", np.nan)
        latest_eur = latest_row.get("EUR", np.nan)
        latest_gbp = latest_row.get("GBP", np.nan)
        latest_jpy = latest_row.get("JPY", np.nan)

        col1, col2, col3, col4 = st.columns(4)
        if np.isfinite(latest_aud):
            col1.metric("USD → AUD (last)", f"{latest_aud:.4f}")
        if np.isfinite(latest_eur):
            col2.metric("USD → EUR (last)", f"{latest_eur:.4f}")
        if np.isfinite(latest_gbp):
            col3.metric("USD → GBP (last)", f"{latest_gbp:.4f}")
        if np.isfinite(latest_jpy):
            col4.metric("USD → JPY (last)", f"{latest_jpy:.2f}")

        # AUD-focused quick chart
        fx_data = fx_data.copy()
        fx_data["ma_7"] = fx_data["rate"].rolling(7).mean()

        st.markdown("#### USD → AUD (with 7-day moving average)")

        fig_fx_aud = go.Figure()
        fig_fx_aud.add_trace(
            go.Scatter(
                x=fx_data["date"],
                y=fx_data["rate"],
                mode="lines",
                name="USD → AUD",
                line=dict(color=PRIMARY_COLOR, width=1.6),
            )
        )
        fig_fx_aud.add_trace(
            go.Scatter(
                x=fx_data["date"],
                y=fx_data["ma_7"],
                mode="lines",
                name="7-day MA (AUD)",
                line=dict(color=ACCENT_COLOR, width=2.0),
            )
        )
        fig_fx_aud.update_layout(
            **BASE_LAYOUT,
            title="USD → AUD (daily with 7-day moving average)",
            xaxis_title="Date",
            yaxis_title="Rate (AUD per 1 USD)",
            legend=dict(
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10),
            ),
        )
        if use_log_scale:
            fig_fx_aud.update_yaxes(type="log")
        fig_fx_aud.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(fig_fx_aud, use_container_width=True, key="overview_fx_aud")

        # Multi-currency chart (normalised to 100 at start of window)
        st.markdown("#### Multi-currency – USD vs majors (normalised to 100)")

        fx_norm_rows = []
        for col in fx_data.columns:
            if col in ("date", "rate", "change_pct", "ma_7"):
                continue
            series = fx_data[["date", col]].dropna()
            if series.empty:
                continue
            base = series[col].iloc[0]
            if base == 0 or not np.isfinite(base):
                continue
            tmp = series.copy()
            tmp["index_value"] = 100.0 * tmp[col] / base
            tmp["label"] = col
            fx_norm_rows.append(tmp[["date", "index_value", "label"]])

        if fx_norm_rows:
            fx_norm_df = pd.concat(fx_norm_rows, ignore_index=True)
            fig_fx_multi = px.line(
                fx_norm_df,
                x="date",
                y="index_value",
                color="label",
                title=(
                    f"FX: USD vs majors (normalised to 100, ~{assets_years}y "
                    "window)"
                ),
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
                legend=dict(
                    orientation="h",
                    y=-0.3,
                    x=0.5,
                    xanchor="center",
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10),
                    itemclick="toggleothers",
                    itemdoubleclick="toggle",
                ),
            )
            if use_log_scale:
                fig_fx_multi.update_yaxes(type="log")
            fig_fx_multi.update_xaxes(rangeslider_visible=True)
            st.plotly_chart(
                fig_fx_multi,
                use_container_width=True,
                key="overview_fx_multi",
            )

    st.markdown("---")

    # ============================================================
    # 4) CRYPTO SNAPSHOT (BTC & ETH)
    # ============================================================
    st.markdown("### ₿ Crypto – BTC & ETH (Yahoo Finance)")

    if crypto_error:
        st.error(f"Crypto data error: {crypto_error}")
    elif not crypto_data or (
        crypto_data.get("BTC") is None
        and crypto_data.get("BTC-USD") is None
    ):
        st.warning("No crypto data loaded.")
    else:
        btc_df = crypto_data.get("BTC") if crypto_data.get("BTC") is not None else crypto_data.get("BTC-USD")
        eth_df = crypto_data.get("ETH") if crypto_data.get("ETH") is not None else crypto_data.get("ETH-USD")

        if btc_df is None or eth_df is None:
            st.warning("BTC/ETH series missing.")
            return

        btc = btc_df.rename(columns={"price": "BTC"})
        eth = eth_df.rename(columns={"price": "ETH"})

        merged = (
            btc.merge(eth, on="date", how="inner")
            .sort_values("date")
            .reset_index(drop=True)
        )
        if merged.empty:
            st.warning("No overlapping BTC/ETH history for this window.")
            return

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
                visible="legendonly",  # hide ETH by default; tap legend to show
            )
        )
        fig_crypto.update_layout(
            **BASE_LAYOUT,
            title=f"BTC vs ETH (price in USD, ~{assets_years}y window)",
            xaxis_title="Date",
            yaxis_title="Price (USD)",
            legend=dict(
                orientation="h",
                y=-0.25,
                x=0.5,
                xanchor="center",
                bgcolor="rgba(0,0,0,0)",
                font=dict(size=10),
                itemclick="toggleothers",
                itemdoubleclick="toggle",
            ),
        )
        if use_log_scale:
            fig_crypto.update_yaxes(type="log")
        fig_crypto.update_xaxes(rangeslider_visible=True)
        st.plotly_chart(
            fig_crypto,
            use_container_width=True,
            key="overview_crypto_btc_eth",
        )

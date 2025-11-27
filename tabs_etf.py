"""ETF detail tab.

Shows:
- Normalised ETF performance with shorter legend labels (better on mobile)
- Correlation heatmap of daily returns
- Optional raw data tables
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from config import BASE_LAYOUT


def render_etf_tab(
    etf_data_bundle: dict | None,
    etf_error: str | None,
    assets_years: int,
    use_log_scale: bool,
) -> None:
    """Render the ETF detail tab."""

    # 1. Dynamic header text based on what actually loaded
    header_text = "📊 ETFs – (no data loaded)"

    if etf_data_bundle and etf_data_bundle.get("data"):
        labels = list(etf_data_bundle["data"].keys())
        short_labels = [lbl.split()[0] for lbl in labels]
        header_text = f"📊 ETFs – {', '.join(short_labels)} (Stooq, EOD)"

    st.subheader(header_text)

    # 2. Handle errors / missing data up front
    if etf_error:
        st.error(f"ETF loader error: {etf_error}")
        return

    if not etf_data_bundle or not etf_data_bundle.get("data"):
        err_list = etf_data_bundle["errors"] if etf_data_bundle else []
        if err_list:
            st.error(
                "All ETF symbols failed:\n\n" + "\n".join(f"- {e}" for e in err_list)
            )
        else:
            st.warning("No ETF data loaded.")
        return

    etf_data = etf_data_bundle["data"]
    err_list = etf_data_bundle.get("errors", [])

    if err_list:
        st.warning(
            "Some ETF symbols failed to load:\n\n" + "\n".join(f"- {e}" for e in err_list)
        )

    # 3. Normalised performance chart
    norm_rows = []
    for label, df in etf_data.items():
        if df is None or df.empty:
            continue
        if "close" not in df.columns:
            continue
        base = df["close"].iloc[0]
        if base == 0 or not np.isfinite(base):
            continue
        tmp = df.copy()
        tmp["index_value"] = 100.0 * tmp["close"] / base
        tmp["full_label"] = label
        tmp["series"] = label.split(" (")[0]  # shorter legend label
        norm_rows.append(tmp[["date", "index_value", "series", "full_label"]])

    if not norm_rows:
        st.warning("ETF time series are empty after filtering.")
        return

    norm_df = pd.concat(norm_rows, ignore_index=True)

    # Metric row – latest prices
    cols = st.columns(len(etf_data))
    for i, (label, df) in enumerate(etf_data.items()):
        if df is None or df.empty:
            continue
        latest = df["close"].iloc[-1]
        cols[i].metric(label, f"${latest:,.2f}")

    st.markdown(f"### Normalised performance (start = 100, ~{assets_years}y window)")

    fig = px.line(
        norm_df,
        x="date",
        y="index_value",
        color="series",
        hover_data={"full_label": True, "series": False},
        title=(
            f"ETF performance (normalised to 100 at start of period, ~{assets_years}y "
            "window)"
        ),
        labels={
            "date": "Date",
            "index_value": "Index (start = 100)",
            "series": "ETF",
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_layout(
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
        fig.update_yaxes(type="log")
    fig.update_xaxes(rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True, key="etf_detail_norm")

    # 4. Correlation heatmap of daily returns
    st.markdown("### Correlation of daily returns (investor view)")

    returns_df = None
    for label, df in etf_data.items():
        if df is None or df.empty or "close" not in df.columns:
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
            title="Correlation matrix of ETF daily returns",
            labels=dict(color="Correlation"),
        )
        fig_corr.update_layout(**BASE_LAYOUT)
        st.plotly_chart(fig_corr, use_container_width=True, key="etf_corr_matrix")
    else:
        st.info("Not enough overlapping data to compute a correlation matrix.")

    # 5. Optional raw data tables
    with st.expander("Show raw ETF data"):
        for label, df in etf_data.items():
            st.markdown(f"**{label}**")
            st.dataframe(df.reset_index(drop=True))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PRIMARY_COLOR, ACCENT_COLOR, BASE_LAYOUT


def render_fx_tab(fx_data, fx_error, assets_years: int, use_log_scale: bool) -> None:
    st.subheader("💱 FX – USD vs AUD / EUR / GBP / JPY (Frankfurter, live)")

    if fx_error:
        st.error(f"FX data error: {fx_error}")
    elif fx_data is None or fx_data.empty:
        st.warning("No FX data loaded.")
    else:
        start_display = fx_data["date"].min().date()
        end_display = fx_data["date"].max().date()
        st.markdown(f"**Date range:** {start_display} → {end_display}")

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

        min_rate = fx_data["rate"].min()
        max_rate = fx_data["rate"].max()
        last_7_avg = fx_data["rate"].tail(7).mean()

        col5, col6, col7 = st.columns(3)
        col5.metric("Min USD → AUD", f"{min_rate:.4f}")
        col6.metric("Max USD → AUD", f"{max_rate:.4f}")
        col7.metric("7-day avg (AUD)", f"{last_7_avg:.4f}")

        fx_plot = fx_data.copy()
        fx_plot["ma_7"] = fx_plot["rate"].rolling(7).mean()
        fx_plot["ma_30"] = fx_plot["rate"].rolling(30).mean()

        st.markdown("### USD → AUD – Price Over Time (Interactive)")

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=fx_plot["date"],
                y=fx_plot["rate"],
                mode="lines+markers",
                name="USD → AUD",
                line=dict(color=PRIMARY_COLOR, width=1.6),
                marker=dict(size=5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=fx_plot["date"],
                y=fx_plot["ma_7"],
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

        st.markdown(
            f"### Multi-Currency – USD vs AUD / EUR / GBP / JPY "
            f"(Normalised, ~{assets_years}y window)"
        )

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


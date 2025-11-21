import pandas as pd
import plotly.express as px
import streamlit as st

from config import BASE_LAYOUT


def render_etf_tab(etf_data_bundle, etf_error, assets_years: int, use_log_scale: bool) -> None:
    # 1. Determine header text dynamically based on what actually loaded
    header_text = "📊 ETFs – (No Data Loaded)"
    
    if etf_data_bundle and etf_data_bundle.get("data"):
        # Create a string like "IVV, VOO, VTI..." from the keys
        labels = list(etf_data_bundle["data"].keys())
        # Extract just the ticker part if possible, or use full label
        short_labels = [lbl.split()[0] for lbl in labels]
        header_text = f"📊 ETFs – {', '.join(short_labels)} (Stooq)"

    st.subheader(header_text)

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

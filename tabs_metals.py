import pandas as pd
import plotly.express as px
import streamlit as st

from config import BASE_LAYOUT


def render_metals_tab(metals_data_bundle, metals_error, assets_years: int, use_log_scale: bool) -> None:
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

        cols = st.columns(len(metals_data))
        for i, (label, df) in enumerate(metals_data.items()):
            latest = df["close"].iloc[-1]
            cols[i].metric(label, f"${latest:,.2f}")

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


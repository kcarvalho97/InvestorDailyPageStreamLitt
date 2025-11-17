import plotly.graph_objects as go
import streamlit as st

from config import PRIMARY_COLOR, ACCENT_COLOR, BASE_LAYOUT


def render_crypto_tab(crypto_data, crypto_error, assets_years: int, use_log_scale: bool) -> None:
    st.subheader("₿ Crypto – BTC & ETH (USD, via CoinGecko – keyless where possible)")

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


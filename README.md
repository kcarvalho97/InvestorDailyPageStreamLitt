# 📈 Keyless Markets Dashboard

A simple, opinionated **Streamlit** dashboard for long-term investors.

It focuses on **no-key / free data sources** only and gives you a clean, fast overview of:

- **Equity ETFs** – VOO, QQQ, SCHD, IWM, TLT, VXUS (via **Stooq** end-of-day CSV)
- **Crypto** – BTC-USD and ETH-USD (via **Yahoo Finance** through `yfinance`)
- **Precious metals** – Gold (XAUUSD), Silver (XAGUSD), Platinum (XPTUSD) (via **Stooq**)
- **FX** – USD vs AUD, EUR, GBP, JPY, CNY, INR (via **Frankfurter** FX API)

No API keys, no secrets, just rate-limited public endpoints + caching.

---

## 🔍 What the app shows

### Overview tab

The default **Overview** tab gives a high-level investor snapshot:

- **ETFs section**
  - Normalised performance chart (all ETFs start at 100 at the beginning of the selected window).
  - Latest close for each ETF as Streamlit metrics.
- **Metals section**
  - Gold, Silver, Platinum normalised performance.
  - Latest **Gold / Silver ratio**.
- **FX section**
  - Detailed USD → AUD chart with 7-day moving average.
  - Multi-currency normalised FX chart (AUD, EUR, GBP, JPY, and any extra FX columns like CNY/INR if the API returns them).
- **Crypto section**
  - BTC and ETH price in USD over time.
  - Latest BTC, ETH prices and BTC/ETH ratio.

### ETF tab

The **ETFs** tab is a deeper dive into the ETF basket:

- Same normalised performance plot (start = 100).
- Per-ETF “latest price” metrics.
- **Correlation matrix** of daily returns across all ETFs to see how diversified they really are.
- Optional raw data tables for each ETF if you want to eyeball the numbers.

---

## 🧩 Tech stack & architecture

- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **Charting**: [Plotly](https://plotly.com/python/) (dark theme, responsive layout)
- **Data sources**:
  - **FX**: [Frankfurter API](https://www.frankfurter.app/) – USD-based rates.
  - **ETFs & metals**: [Stooq](https://stooq.com/) CSV endpoints (no key, end-of-day).
  - **Crypto**: [Yahoo Finance](https://finance.yahoo.com/) via the `yfinance` Python package.

**Performance features:**

- All data access is centralised in `data_loaders.py`.
- `fetch_all_data_concurrently(...)` uses a `ThreadPoolExecutor` to fetch FX, ETFs, metals and crypto in parallel.
- Streamlit’s `@st.cache_data` is used to cache each dataset for `CACHE_TTL_SECONDS` (defined in `config.py`, default 10 minutes) to avoid hammering the free APIs.

---

## 📁 Project structure

The repo is deliberately small:

```text
.
├── app.py              # Main Streamlit entrypoint (tabs + layout + wiring)
├── config.py           # Colors, Plotly base layout, cache settings
├── data_loaders.py     # All external data fetching + caching + concurrency
├── tabs_overview.py    # Overview tab (ETFs, metals, FX, crypto snapshot)
├── tabs_etf.py         # ETF deep-dive tab (performance + correlation, etc.)
├── requirements.txt    # Python dependencies for local + Streamlit Cloud
└── README.md           # This file


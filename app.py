# ==============================================================
# Project AlphaSent – Streamlit App (Refactored v2.1)
# Sentiment‑Driven Crypto Allocation Dashboard
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt
import plotly.express as px
from openai import OpenAI
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --------------------------------------------------------------
# PAGE CONFIGURATION & SETUP
# --------------------------------------------------------------
st.set_page_config(
    page_title="Project AlphaSent",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Project AlphaSent")
st.subheader("Sentiment‑Driven Crypto Allocation")

# --- CONFIGURATION ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
CRYPTOCOMPARE_API_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
SENTIMENT_ZSCORE_THRESHOLD = 1.0
TRANSACTION_COST_BPS = 25  # bps per round‑trip

# --------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------

def format_percent(x: float) -> str:
    return f"{x:.2%}" if np.isfinite(x) else "—"


def _sentiment_emoji(score: float) -> str:
    if score > 0.25:
        return "🚀"
    if score < -0.25:
        return "⚠️"
    return "😐"


@st.cache_resource
def setup_nltk():
    import nltk
    with st.spinner("Setting up NLTK resources…"):
        nltk.download("vader_lexicon", quiet=True)
    st.success("NLTK is ready!")


setup_nltk()

# --------------------------------------------------------------
# BACKEND HELPERS
# --------------------------------------------------------------


@st.cache_data(show_spinner=False)
def create_requests_session() -> requests.Session:
    """Create a resilient requests session."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url, index_col=0, parse_dates=True).rename_axis("time")
    except Exception as e:
        st.error(f"Data load failed → {e}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False, ttl=300)
def fetch_and_analyze_live_news(api_key: str) -> pd.DataFrame:
    """Fetch latest headlines & compute sentiment (cached 5 min)."""
    if not api_key:
        return pd.DataFrame()
    session = create_requests_session()
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        data = session.get(url, timeout=10).json().get("Data", [])
        df = pd.DataFrame(data).head(20)
        analyzer = SentimentIntensityAnalyzer()
        df["compound"] = df["title"].fillna("").apply(lambda t: analyzer.polarity_scores(t)["compound"])
        return df[["title", "source", "compound", "url"]]
    except Exception:
        return pd.DataFrame()


def get_portfolio_weights(prices: pd.DataFrame, model: str = "max_sharpe") -> pd.Series:
    mu = expected_returns.mean_historical_return(prices, frequency=365)
    S = risk_models.sample_cov(prices, frequency=365)
    ef = EfficientFrontier(mu, S)
    try:
        if model == "max_sharpe":
            ef.max_sharpe()
        elif model == "min_variance":
            ef.min_volatility()
        return pd.Series(ef.clean_weights())
    except Exception:
        return pd.Series(1 / len(prices.columns), index=prices.columns)


def run_backtest(prices_df: pd.DataFrame, sentiment_df: pd.DataFrame):
    st.write("Running Sentiment‑Regime Back‑test…")
    daily_ret = prices_df.pct_change().dropna()
    reb_dates = prices_df.resample("W-FRI").last().index
    if len(reb_dates) < 2:
        return None, None, None

    sent_z = (sentiment_df["compound"] - sentiment_df["compound"].rolling(90).mean()) / sentiment_df["compound"].rolling(90).std()

    port_rets, last_w, regimes = [], pd.Series(dtype=float), []

    for i in range(len(reb_dates) - 1):
        start, end = reb_dates[i], reb_dates[i + 1]
        s_slice = sent_z.loc[:start].dropna()
        if s_slice.empty:
            continue
        sig = s_slice.iloc[-1]
        risk_on = sig > SENTIMENT_ZSCORE_THRESHOLD
        regimes.append({"date": start, "regime": int(risk_on)})

        hist_prices = prices_df.loc[:start].tail(180)
        if len(hist_prices) < 90:
            continue

        mvo_w = get_portfolio_weights(hist_prices, "max_sharpe")
        minvar_w = get_portfolio_weights(hist_prices, "min_variance")
        a, b = ((0.8, 0.2) if risk_on else (0.2, 0.8))
        target_w = (a * mvo_w + b * minvar_w).fillna(0)

        cost = ((target_w - last_w.reindex(target_w.index).fillna(0)).abs().sum() / 2) * (TRANSACTION_COST_BPS / 10000)
        pr = (daily_ret.loc[start:end] * target_w).sum(axis=1)
        if not pr.empty:
            pr.iloc[0] -= cost
            port_rets.append(pr)
        last_w = target_w

    if not port_rets:
        return None, None, None

    strat_r = pd.concat(port_rets).sort_index()
    regime_df = pd.DataFrame(regimes).set_index("date")
    st.write("✓ Back‑test complete.")
    return strat_r, last_w, regime_df


def generate_gemini_summary(results: dict, latest_sent: float, wts):
    if not OPENROUTER_API_KEY:
        return "Add an OpenRouter API key to enable AI summaries."
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    prompt = f"""You are a FinTech analyst summarizing a back‑test of a crypto strategy 'AlphaSent'.\n\nPerformance\n • CAGR: {results['CAGR']}\n • Sharpe: {results['Sharpe']}\n\nLatest 7‑day sentiment: {latest_sent:.2f}\n\nNext‑period allocation\n{wts if isinstance(wts, str) else wts.to_string()}\n\nGive three short sections: Performance, Outlook, Allocation."""
    try:
        chat = client.chat.completions.create(model="google/gemini-2.5-flash", messages=[{"role": "user", "content": prompt}])
        return chat.choices[0].message.content
    except Exception as e:
        return f"AI summary failed → {e}"

# --------------------------------------------------------------
# SIDEBAR – LIVE NEWS & CONTROLS
# --------------------------------------------------------------

st.sidebar.header("Live News Feed")
news_df = fetch_and_analyze_live_news(CRYPTOCOMPARE_API_KEY)
if not news_df.empty:
    for _, r in news_df.iterrows():
        st.sidebar.markdown(f"{_sentiment_emoji(r['compound'])} **{r['source']}**")
        st.sidebar.markdown(f"[{r['title'][:55]}…]({r['url']})")
        st.sidebar.progress(int((r['compound'] + 1) / 2 * 100))
        st.sidebar.markdown("---")
else:
    st.sidebar.info("Live news feed unavailable.")

st.sidebar.divider()
st.sidebar.header("Simulation Controls")
run_btn = st.sidebar.button("▶️  Run / Refresh Back‑test", type="secondary")

# --------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------

if run_btn:
    df_raw = load_data(DATA_URL)
    if df_raw.empty:
        st.stop()

    prices = df_raw.drop(columns=["compound"], errors="ignore")
    sentiments = df_raw[["compound"]].dropna()

    with st.spinner("Back‑testing… this may take 15‑30 s"):
        strat_ret, last_w, regime = run_backtest(prices, sentiments)

    if strat_ret is None or strat_ret.empty:
        st.error("Back‑test could not be completed – insufficient data.")
        st.stop()
    if strat_ret.count() < 30:
        st.warning("Not enough history (< 30 obs) to compute meaningful metrics.")
        st.stop()

    # KPIs
    cum_rets = (1 + strat_ret).cumprod()
    n_days = (cum_rets.index[-1] - cum_rets.index[0]).days or 1
    cagr = cum_rets.iloc[-1] ** (365 / n_days) - 1
    ann_vol = strat_ret.std(ddof=0) * np.sqrt(365)
    sharpe = cagr / ann_vol if ann_vol else np.nan

    tab_perf, tab_diag, tab_ai = st.tabs(["Performance", "Internals", "AI Analysis"])

    with tab_perf:
        c1, c2, c3 = st.columns(3)
        c1.metric("CAGR", format_percent(cagr))
        c2.metric("Volatility", format_percent(ann_vol))
        c3.metric("Sharpe", f"{sharpe:.2f}" if np.isfinite(sharpe) else "—")

        if "BTC" in prices.columns:
            bench = (1 + prices["BTC"].pct_change()).cumprod().loc[cum_rets.index]
            px_df = pd.DataFrame({"Bitcoin": bench, "AlphaSent": cum_rets})
        else:
            px_df = cum_rets.to_frame("AlphaSent")
        fig = px.line(px_df, log_y=True, title="Cumulative Return (log‑scale)")
        st.plotly_chart(fig, use_container_width=True)

    with tab_diag:
        with st.expander("Sentiment Regime Details", expanded=False):
            fig_r, ax_r = plt.subplots(figsize=(12, 4))
            sent_z = (sentiments["compound"] - sentiments["compound"].rolling(90).mean()) / sentiments["compound"].rolling(90).std()
            ax_r.plot(sent_z.index, sent_z.values, color="purple", alpha=0.7, label="Sentiment Z‑Score")
            ax_r.axhline(SENTIMENT_ZSCORE_THRESHOLD, color="red", ls="--", label="Risk‑On threshold")
            if regime is not None and not regime.empty:
                ax_r.fill_between(regime.index, 0, 1, where=regime["regime"] == 1, color="green", alpha=0.2, transform=ax_r.get_xaxis_transform(), label="Risk‑On")
            ax_r.legend(loc="upper left")
            st.pyplot(fig_r)

        st.subheader("Next‑Period Allocation")
        if last_w is not None and not last_w.empty and last_w.sum() > 0:
            fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
            last_w[last_w > 0.01].plot.pie(ax=ax_pie, autopct="%1.1f%%", startangle=90)
            ax_pie.set_ylabel("")
            st.pyplot(fig_pie)
        else:
            st.info("100 % Cash – Risk‑Off regime.")

    with tab_ai:
        st.header("Gemini AI Analysis")
        with st.spinner("Generating AI summary…"):
            res_dict = {"CAGR": format_percent(cagr), "Sharpe": f"{sharpe:.2f}"}
            latest_sent = sentiments["compound"].tail(7).mean()
            top_w = last_w[last_w > 0.01].sort_values(ascending=False) if last_w is not None else "Cash"
            st.markdown(generate_gemini_summary(res_dict, latest_sent, top_w))

else:
    st.info("Hit **Run / Refresh** to simulate the strategy with the most recent data.")

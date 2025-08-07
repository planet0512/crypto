# ==============================================================
# Project AlphaSent – Streamlit App (Refactored v2.2)
# Sentiment‑Driven Crypto Allocation Dashboard
# (Changelog v2.2 ➜ nicer KPIs, cleaner charts, better allocation view)
# ==============================================================

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --------------------------------------------------------------
# PAGE CONFIGURATION & SETUP
# --------------------------------------------------------------
st.set_page_config(page_title="Project AlphaSent", page_icon="📈", layout="wide")

st.title("📈 Project AlphaSent")
st.subheader("Sentiment‑Driven Crypto Allocation")

# --- CONFIGURATION ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
CRYPTOCOMPARE_API_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
SENTIMENT_ZSCORE_THRESHOLD = 1.0
TRANSACTION_COST_BPS = 25  # round‑trip cost in bps

# --------------------------------------------------------------
# UTILITIES
# --------------------------------------------------------------

def format_percent(x: float) -> str:
    """Return a percent string or em‑dash for NaN/inf."""
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
    nltk.download("vader_lexicon", quiet=True)


setup_nltk()

# --------------------------------------------------------------
# BACKEND HELPERS
# --------------------------------------------------------------


@st.cache_data(show_spinner=False)
def create_requests_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504]))
    session.mount("http://", adapter); session.mount("https://", adapter)
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
    if not api_key:
        return pd.DataFrame()
    session = create_requests_session()
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        data = session.get(url, timeout=10).json()["Data"][:20]
        df = pd.DataFrame(data)
        analyzer = SentimentIntensityAnalyzer()
        df["compound"] = df["title"].fillna("").apply(lambda t: analyzer.polarity_scores(t)["compound"])
        return df[["title", "source", "compound", "url"]]
    except Exception:
        return pd.DataFrame()


def get_portfolio_weights(prices: pd.DataFrame, model: str) -> pd.Series:
    mu = expected_returns.mean_historical_return(prices, frequency=365)
    cov = risk_models.sample_cov(prices, frequency=365)
    ef = EfficientFrontier(mu, cov)
    try:
        getattr(ef, "max_sharpe" if model == "max_sharpe" else "min_volatility")()
        return pd.Series(ef.clean_weights())
    except Exception:
        return pd.Series(1 / len(prices.columns), index=prices.columns)


def run_backtest(prices: pd.DataFrame, sentiment: pd.Series):
    st.write("Running Sentiment‑Regime Back‑test…")
    daily_ret = prices.pct_change().dropna()
    reb_dates = prices.resample("W-FRI").last().index
    if len(reb_dates) < 2:
        return None, None, None

    sent_z = (sentiment - sentiment.rolling(90).mean()) / sentiment.rolling(90).std()
    port_rets, last_w, regime_hist = [], pd.Series(dtype=float), []

    for i in range(len(reb_dates) - 1):
        start, end = reb_dates[i], reb_dates[i + 1]
        z_val = sent_z.loc[:start].dropna().iloc[-1] if not sent_z.loc[:start].dropna().empty else 0
        risk_on = z_val > SENTIMENT_ZSCORE_THRESHOLD
        regime_hist.append({"date": start, "regime": int(risk_on)})

        hist_prices = prices.loc[:start].tail(180)
        if len(hist_prices) < 90:
            continue

        mvo_w = get_portfolio_weights(hist_prices, "max_sharpe")
        minvar_w = get_portfolio_weights(hist_prices, "min_variance")
        a, b = ((0.8, 0.2) if risk_on else (0.2, 0.8))
        w = (a * mvo_w + b * minvar_w).fillna(0)

        cost = ((w - last_w.reindex(w.index).fillna(0)).abs().sum() / 2) * (TRANSACTION_COST_BPS / 10000)
        period_ret = (daily_ret.loc[start:end] * w).sum(axis=1)
        if not period_ret.empty:
            period_ret.iloc[0] -= cost
            port_rets.append(period_ret)
        last_w = w

    if not port_rets:
        return None, None, None

    strat_r = pd.concat(port_rets).sort_index()
    regime_df = pd.DataFrame(regime_hist).set_index("date")
    st.write("✓ Back‑test complete.")
    return strat_r, last_w, regime_df


def ai_summary(results: dict, latest_sent: float, wts):
    if not OPENROUTER_API_KEY:
        return "(Add an OpenRouter API key in secrets to enable summaries)"
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    prompt = (
        "You are a FinTech analyst summarizing the back‑test of 'AlphaSent'.\n\n"
        f"CAGR: {results['CAGR']}, Sharpe: {results['Sharpe']}.\n"
        f"Latest 7‑day sentiment: {latest_sent:.2f}.\n"
        "Next‑period allocation: " + (wts if isinstance(wts, str) else wts.to_string())
    )
    try:
        msg = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        return msg.choices[0].message.content
    except Exception as e:
        return f"AI summary failed → {e}"

# --------------------------------------------------------------
# SIDEBAR – LIVE NEWS & CONTROLS
# --------------------------------------------------------------
st.sidebar.header("Live News Feed")
news = fetch_and_analyze_live_news(CRYPTOCOMPARE_API_KEY)
if not news.empty:
    for _, r in news.iterrows():
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
    df = load_data(DATA_URL)
    if df.empty:
        st.stop()

    price_df = df.drop(columns=["compound"], errors="ignore")
    sent_series = df["compound"].dropna()

    with st.spinner("Back‑testing…"):
        strat_ret, last_w, regime_df = run_backtest(price_df, sent_series)

    if strat_ret is None or strat_ret.count() < 30:
        st.error("Back‑test could not be completed or history too short.")
        st.stop()

    # KPIs
    cum = (1 + strat_ret).cumprod()
    n_days = max((cum.index[-1] - cum.index[0]).days, 1)
    cagr = cum.iloc[-1] ** (365 / n_days) - 1
    vol = strat_ret.std(ddof=0) * np.sqrt(365)
    sharpe = cagr / vol if vol > 0 else np.nan

    # Layout Tabs
    tab_perf, tab_diag, tab_ai = st.tabs(["Performance", "Internals", "AI Analysis"])

    # -------------- PERFORMANCE TAB ---------------------------
    with tab_perf:
        k1, k2, k3 = st.columns(3)
        k1.metric("CAGR", format_percent(cagr))
        k2.metric("Volatility", format_percent(vol))
        k3.metric("Sharpe", f"{sharpe:.2f}" if np.isfinite(sharpe) else "—")

        # Interactive cumulative chart
        show_cols = {"AlphaSent": cum}
        if "BTC" in price_df.columns:
            bench = (1 + price_df["BTC"].pct_change()).cumprod().loc[cum.index]
            show_cols["Bitcoin"] = bench
        fig = px.line(pd.DataFrame(show_cols), log_y=True, labels={"value": "Cum‑Return", "index": "Date", "variable": "Series"})
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    # -------------- INTERNALS TAB -----------------------------
    with tab_diag:
        with st.expander("Sentiment Regime Details", expanded=False):
            sent_z = (sent_series - sent_series.rolling(90).mean()) / sent_series.rolling(90).std()
            fig_z = go.Figure()
            fig_z.add_trace(go.Scatter(x=sent_z.index, y=sent_z.values, name="Sentiment Z", line=dict(color="purple")))
            fig_z.add_hline(y=SENTIMENT_ZSCORE_THRESHOLD, line_dash="dash", line_color="red", annotation_text="Risk‑On")
            for r in regime_df.itertuples():
                if r.regime:
                    fig_z.add_vrect(x0=r.Index, x1=r.Index + timedelta(days=7), fillcolor="green", opacity=0.1, line_width=0)
            st.plotly_chart(fig_z, use_container_width=True)

        st.subheader("Next‑Period Allocation")
        if last_w is not None and last_w.sum() > 0:
            alloc = last_w[last_w > 0.01].sort_values(ascending=False)
            fig_alloc = px.bar(alloc, orientation="h", text_auto=".1%", labels={"value": "Weight", "index": "Asset"})
            fig_alloc.update_layout(yaxis=dict(categoryorder="total ascending"))
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("100 % Cash – Risk‑Off regime.")

    # -------------- AI ANALYSIS TAB ---------------------------
    with tab_ai:
        st.header("Gemini AI Analysis")
        with st.spinner("Generating summary…"):
            res = {"CAGR": format_percent(cagr), "Sharpe": f"{sharpe:.2f}"}
            latest_sent_val = sent_series.tail(7).mean()
            w_display = "Cash" if last_w is None else last_w[last_w > 0.01].sort_values(ascending=False)
            st.markdown(ai_summary(res, latest_sent_val, w_display))
else:
    st.info("Hit **Run / Refresh** to simulate the strategy with the most recent data.")

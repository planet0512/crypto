import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.objective_functions import L2_reg
from pypfopt.exceptions import OptimizationError
import cvxpy as cp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================================================================
# PAGE CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(page_title="Project AlphaSent", page_icon="📈", layout="wide")

st.title("📈 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- CONFIGURATION ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
CRYPTOCOMPARE_API_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
SENTIMENT_ZSCORE_THRESHOLD = 1.0

@st.cache_resource
def setup_nltk():
    """Download NLTK data (runs once)."""
    import nltk
    with st.spinner("Setting up NLTK resources…"):
        nltk.download('vader_lexicon', quiet=True)
    st.success("NLTK resources are ready.")

setup_nltk()

# ==============================================================================
# BACKEND HELPERS
# ==============================================================================

@st.cache_data
def create_requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session

@st.cache_data
def load_data(url: str) -> pd.DataFrame:
    st.write("Loading historical backtest data from GitHub…")
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df.index.name = 'time'
        st.write("✓ Backtest data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_and_analyze_live_news(_session: requests.Session, api_key: str) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        data = _session.get(url).json().get('Data', [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data).head(20)
        analyzer = SentimentIntensityAnalyzer()
        df['compound'] = df['title'].fillna('').apply(lambda txt: analyzer.polarity_scores(txt)['compound'])
        return df[['title', 'source', 'compound', 'url']]
    except Exception:
        return pd.DataFrame()

# ------------------------------------------------------------------------------
# Robust portfolio optimisation wrapper
# ------------------------------------------------------------------------------

def get_portfolio_weights(prices: pd.DataFrame, model: str = "max_sharpe") -> pd.Series:
    """Return optimal weights; fall back to equal‑weight if optimiser fails."""
    # Clean & filter data
    returns = prices.pct_change().dropna(how="all")
    valid_cols = returns.std()[lambda s: s > 0].index  # remove zero‑variance assets
    if len(valid_cols) < 2:
        return pd.Series(1 / len(prices.columns), index=prices.columns)

    prices = prices[valid_cols]

    # Estimate moments with shrinkage
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    ef.add_objective(L2_reg, gamma=0.001)  # small ridge

    try:
        if model == "max_sharpe":
            ef.max_sharpe()
        elif model == "min_variance":
            ef.min_volatility()
        weights = ef.clean_weights()
        return pd.Series(weights)
    except (OptimizationError, ValueError, cp.error.SolverError) as e:
        st.warning(f"Optimiser failed ({e.__class__.__name__}); using equal weights.")
        return pd.Series(1 / len(prices.columns), index=prices.columns)

# ------------------------------------------------------------------------------
# Backtest engine
# ------------------------------------------------------------------------------

def run_backtest(prices_df: pd.DataFrame, sentiment_index: pd.DataFrame):
    st.write("Running Sentiment‑Regime Backtest…")

    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('ME').last().index
    if len(rebalance_dates) < 2:
        return None, None, None

    portfolio_returns = []
    last_weights = pd.Series()
    regime_history = []

    sentiment_zscore = (
        (sentiment_index['compound'] - sentiment_index['compound'].rolling(90).mean()) /
        sentiment_index['compound'].rolling(90).std()
    )

    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i + 1]

        # Sentiment and regime
        sentiment_slice = sentiment_zscore.loc[:start_date].dropna()
        if sentiment_slice.empty:
            continue
        signal = sentiment_slice.iloc[-1]
        is_risk_on = signal > SENTIMENT_ZSCORE_THRESHOLD
        regime_history.append({'date': start_date, 'regime': int(is_risk_on)})

        mvo_blend, min_var_blend = (0.8, 0.2) if is_risk_on else (0.2, 0.8)

        # Look‑back window and cleaning
        hist_prices = prices_df.loc[:start_date].tail(90)
        hist_prices = hist_prices.dropna(axis=1, how='all').ffill()
        hist_prices = hist_prices.loc[:, hist_prices.nunique() > 1]
        if hist_prices.shape[0] < 60 or hist_prices.shape[1] < 2:
            continue

        # Optimise
        mvo_w  = get_portfolio_weights(hist_prices, "max_sharpe")
        min_w  = get_portfolio_weights(hist_prices, "min_variance")
        target = (mvo_blend * mvo_w + min_var_blend * min_w).fillna(0)

        # Transaction cost (25bps round trip)
        costs = (target - last_weights.reindex(target.index).fillna(0)).abs().sum() / 2 * 0.0025

        # Realised returns over next month
        period_returns = (daily_returns.loc[start_date:end_date] * target).sum(axis=1)
        if not period_returns.empty:
            period_returns.iloc[0] -= costs
        portfolio_returns.append(period_returns)
        last_weights = target

    if not portfolio_returns:
        return None, None, None

    strategy_returns = pd.concat(portfolio_returns)
    regime_df = pd.DataFrame(regime_history).set_index('date')
    st.write("✓ Backtest complete.")
    return strategy_returns, last_weights, regime_df

# ------------------------------------------------------------------------------
# Gemini summary helper
# ------------------------------------------------------------------------------

def generate_gemini_summary(results: dict, latest_sentiment: float, latest_weights: pd.Series):
    if not OPENROUTER_API_KEY:
        return "Please add your OpenRouter API Key to Streamlit secrets."

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    prompt = f"""
    You are a FinTech analyst summarising a backtest of 'AlphaSent', a Sentiment‑Regime Switching crypto strategy.

    Backtest results:
    - Annual Return: {results['Annual Return']}
    - Sharpe Ratio: {results['Sharpe Ratio']}

    Latest 7‑day average sentiment score: {latest_sentiment:.2f}

    Next‑period recommended allocation:\n{latest_weights.to_string()}

    Provide:
    1. **Performance Summary**
    2. **Current Outlook**
    3. **Recommended Allocation**
    """
    try:
        completion = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate Gemini summary: {e}"

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================

session = create_requests_session()

# Sidebar – live news
st.sidebar.header("Live News Feed")
live_news = fetch_and_analyze_live_news(session, CRYPTOCOMPARE_API_KEY)
if not live_news.empty:
    for _, row in live_news.iterrows():
        st.sidebar.markdown(f"**{row['source']}**")
        st.sidebar.markdown(f"[{row['title'][:55]}…]({row['url']})")
        st.sidebar.progress(int((row['compound'] + 1) / 2 * 100))
        st.sidebar.markdown("---")
else:
    st.sidebar.info("Live news feed unavailable.")

st.sidebar.divider()
st.sidebar.header("AlphaSent Controls")

if st.sidebar.button("🚀 Run Full Backtest", type="primary"):
    data = load_data(DATA_URL)

    if not data.empty:
        prices_df = data.drop(columns=['compound'], errors='ignore')
        sentiment_idx = data[['compound']].dropna()

        with st.spinner("Running backtest…"):
            strategy_rets, latest_wts, regime_df = run_backtest(prices_df, sentiment_idx)

        if strategy_rets is not None:
            st.success("Analysis Complete!")

            cum_rets = (1 + strategy_rets).cumprod()
            ann_ret = cum_rets.iloc[-1] ** (365 / len(cum_rets)) - 1
            ann_vol = strategy_rets.std() * np.sqrt(365)
            sharpe = ann_ret / ann_vol if ann_vol else 0

            tab1, tab2, tab3 = st.tabs(["📈 Performance Dashboard", "🔬 Strategy Internals", "🤖 Gemini AI Analysis"])

            # --------------------------------------------------------------
            with tab1:
                st.header("Backtest Performance Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Annual Return", f"{ann_ret:.2%}")
                col2.metric("Annual Volatility", f"{ann_vol:.2%}")
                col3.metric("Sharpe Ratio", f"{sharpe:.2f}")

                fig, ax = plt.subplots(figsize=(12, 6))
                if 'BTC' in prices_df.columns:
                    bench = (1 + prices_df['BTC'].pct_change()).cumprod()
                    ax.plot(bench.loc[cum_rets.index], label='Bitcoin', ls='--', color='gray')
                ax.plot(cum_rets, label='AlphaSent', lw=2)
                ax.set_yscale('log')
                ax.set_title('Strategy vs. Bitcoin (Log Scale)')
                ax.set_ylabel('Cumulative Returns')
                ax.legend()
                st.pyplot(fig)

            # --------------------------------------------------------------
            with tab2:
                st.header("Strategy Internals & Diagnostics")

                st.subheader("Sentiment Regime Indicator")
                fig_r, ax_r = plt.subplots(figsize=(12, 4))
                z = (sentiment_idx['compound'] - sentiment_idx['compound'].rolling(90).mean()) / sentiment_idx['compound'].rolling(90).std()
                ax_r.plot(z.index, z, label='Sentiment Z‑Score', color='purple', alpha=0.7)
                ax_r.axhline(SENTIMENT_ZSCORE_THRESHOLD, color='red', ls='--', label='Risk‑On Threshold')
                ax_r.fill_between(regime_df.index, 0, 1, where=regime_df['regime']==1, color='green', alpha=0.2, transform=ax_r.get_xaxis_transform(), label='Risk‑On')
                ax_r.set_title('Sentiment Z‑Score & Regime')
                ax_r.legend()
                st.pyplot(fig_r)

                st.subheader("Final Recommended Portfolio Allocation")
                if latest_wts is not None and not latest_wts.empty:
                    fig_p, ax_p = plt.subplots(figsize=(6, 6))
                    latest_wts[latest_wts > 0.01].plot.pie(ax=ax_p, autopct='%1.1f%%', startangle=90)
                    ax_p.set_ylabel('')
                    ax_p.set_title('Next‑Period Allocation')
                    st.pyplot(fig_p)
                else:
                    st.info("Final allocation is 100% Cash.")

            # --------------------------------------------------------------
            with tab3:
                st.header("Gemini AI Analysis")
                with st.spinner("Generating summary…"):
                    res_dict = {"Annual Return": f"{ann_ret:.2%}", "Sharpe Ratio": f"{sharpe:.2f}"}
                    latest_sent = sentiment_idx['compound'].tail(7).mean()
                    top_wts = latest_wts[latest_wts > 0.01].sort_values(ascending=False) if latest_wts is not None else pd.Series(["Cash"])
                    summary = generate_gemini_summary(res_dict, latest_sent, top_wts)
                    st.markdown(summary)
        else:
            st.error("Backtest failed – no valid windows.")
else:
    st.info("Click the button in the sidebar to run the backtest.")

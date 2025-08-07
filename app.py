
# app.py
# FINAL SUBMISSION VERSION

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pypfopt import EfficientFrontier, risk_models, expected_returns
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Project AlphaSent", page_icon="📈", layout="wide")
st.title("📈 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- CONFIGURATION ---
DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/final_app_data.csv"

CRYPTOCOMPARE_API_KEY = st.secrets.get("94962fd845ea903749954d66cd59c12c0ee2ee6d8d1f45b3c74e461e9cdc5757", "")

@st.cache_resource
def setup_nltk():
    import nltk; nltk.download('vader_lexicon', quiet=True)
setup_nltk()

# ==============================================================================
# BACKEND FUNCTIONS
# ==============================================================================
@st.cache_data
def load_data(url):
    st.write(f"Loading historical backtest data from GitHub...")
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df.index.name = 'time'
        st.write("✓ Backtest data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}"); return pd.DataFrame()

def run_backtest(full_data_df):
    st.write("Running Sentiment-Regime Backtest...")
    prices_df = full_data_df.drop(columns=['compound'], errors='ignore')
    sentiment_index = full_data_df[['compound']].dropna()
    if prices_df.empty or sentiment_index.empty: return None, None
    
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('W-FRI').last().index
    if len(rebalance_dates) < 2: return None, None
        
    portfolio_returns, last_weights = [], pd.Series()
    sentiment_zscore = (sentiment_index['compound'] - sentiment_index['compound'].rolling(90).mean()) / sentiment_index['compound'].rolling(90).std()
    
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i+1]
        sentiment_slice = sentiment_zscore.loc[:start_date].dropna()
        if sentiment_slice.empty: continue
        sentiment_signal = sentiment_slice.iloc[-1]
        if pd.isna(sentiment_signal): sentiment_signal = 0
        mvo_weight, min_var_weight = (0.8, 0.2) if sentiment_signal > 1.0 else (0.2, 0.8)
        hist_prices = prices_df.loc[:start_date].tail(90)
        if hist_prices.shape[0] < 90: continue
        
        try:
            mu = expected_returns.mean_historical_return(hist_prices)
            S = risk_models.sample_cov(hist_prices)
            ef_mvo = EfficientFrontier(mu, S); ef_mvo.max_sharpe()
            mvo_weights = pd.Series(ef_mvo.clean_weights())
            ef_min_var = EfficientFrontier(mu, S); ef_min_var.min_volatility()
            min_var_weights = pd.Series(ef_min_var.clean_weights())
            target_weights = (mvo_weight * mvo_weights + min_var_weight * min_var_weights).fillna(0)
        except Exception:
            # Fallback to equal weight if optimizer fails
            target_weights = pd.Series(1/len(hist_prices.columns), index=hist_prices.columns)

        turnover = (target_weights - last_weights.reindex(target_weights.index).fillna(0)).abs().sum() / 2
        costs = turnover * (25 / 10000)
        period_returns = (daily_returns.loc[start_date:end_date] * target_weights).sum(axis=1)
        if not period_returns.empty: period_returns.iloc[0] -= costs
        portfolio_returns.append(period_returns)
        last_weights = target_weights

    if not portfolio_returns: return None, None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns, last_weights

@st.cache_data
def fetch_and_analyze_live_news(_session, api_key):
    """Fetches latest news and analyzes sentiment for display."""
    st.sidebar.write("Fetching live news...")
    if not api_key:
        st.sidebar.warning("CryptoCompare API Key not found in secrets.")
        return pd.DataFrame()
    
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        data = _session.get(url).json().get('Data', [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data).head(10)
        analyzer = SentimentIntensityAnalyzer()
        df['compound'] = df['title'].fillna('').apply(lambda txt: analyzer.polarity_scores(txt)['compound'])
        return df[['title', 'source', 'compound']]
    except Exception:
        return pd.DataFrame()

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
session = create_requests_session()

# --- Live News Sidebar ---
st.sidebar.header("Live News Sentiment")
live_news = fetch_and_analyze_live_news(session, CRYPTOCOMPARE_API_KEY)
if not live_news.empty:
    for _, row in live_news.iterrows():
        st.sidebar.metric(label=f"{row['source']}", value=f"{row['title'][:35]}...", delta=f"{row['compound']:.2f}")
else:
    st.sidebar.info("Live news feed unavailable.")

st.sidebar.divider()
st.sidebar.header("Historical Backtest")
if st.sidebar.button("🚀 Run Full Backtest", type="primary"):
    
    backtest_data = load_data(DATA_URL)
    
    if not backtest_data.empty:
        with st.spinner("Running backtest..."):
            strategy_returns, _ = run_backtest(backtest_data)

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            
            cumulative_returns = (1 + strategy_returns).cumprod()
            annual_return = cumulative_returns.iloc[-1]**(365/len(cumulative_returns)) - 1
            annual_volatility = strategy_returns.std() * (365**0.5)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
            
            st.header("Backtest Performance Results")
            col1, col2, col3 = st.columns(3)
            col1.metric("Annual Return", f"{annual_return:.2%}")
            col2.metric("Annual Volatility", f"{annual_volatility:.2%}")
            col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

            fig, ax = plt.subplots(figsize=(12, 6))
            prices_df = backtest_data.drop(columns=['compound'], errors='ignore')
            if 'BTC' in prices_df.columns:
                benchmark = (1 + prices_df['BTC'].pct_change()).cumprod()
                ax.plot(benchmark.loc[strategy_returns.index], label='Bitcoin (Benchmark)', color='gray', linestyle='--')
            ax.plot(cumulative_returns, label='Sentiment-Regime Strategy', color='royalblue', linewidth=2)
            ax.set_title('Sentiment-Regime Strategy vs. Bitcoin'); ax.set_ylabel('Cumulative Returns (Log Scale)'); ax.set_yscale('log'); ax.legend(); st.pyplot(fig)
        else:
            st.error("Could not complete the backtest.")
else:
    st.info("Click the button to run the backtest on the pre-processed historical data.")




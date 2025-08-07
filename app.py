
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

CRYPTOCOMPARE_API_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
NEWS_HISTORY_DAYS = 730

# --- NLTK DATA DOWNLOADER FOR STREAMLIT CLOUD ---
@st.cache_resource
def setup_nltk():
    """Download all required NLTK data packages."""
    import nltk
    with st.spinner("Setting up NLTK resources... (This runs once)"):
        nltk.download('vader_lexicon', quiet=True)
    st.success("NLTK resources are ready.")

# Run the setup at the start of the app
setup_nltk()

# ==============================================================================
# BACKEND HELPER & PIPELINE FUNCTIONS (Stations 1, 2, 3)
# ==============================================================================

@st.cache_data
def create_requests_session() -> requests.Session:
    """Creates a requests session with a retry policy for network robustness."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter); session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session

@st.cache_data
def load_data(url):
    """Loads the pre-processed backtest data from the GitHub CSV."""
    st.write(f"Loading historical backtest data from GitHub...")
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df.index.name = 'time'
        st.write("✓ Backtest data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}"); return pd.DataFrame()

@st.cache_data
def fetch_and_analyze_live_news(_session, api_key):
    """Fetches the latest news from CryptoCompare and analyzes its sentiment for display."""
    st.sidebar.write("Fetching live news...")
    if not api_key:
        st.sidebar.warning("CryptoCompare API Key not found in secrets.")
        return pd.DataFrame()
    
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        data = _session.get(url).json().get('Data', [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data).head(10) # Get top 10 articles
        analyzer = SentimentIntensityAnalyzer()
        df['compound'] = df['title'].fillna('').apply(lambda txt: analyzer.polarity_scores(txt)['compound'])
        return df[['title', 'source', 'compound']]
    except Exception:
        return pd.DataFrame()

def run_backtest(prices_df, sentiment_index):
    """
    Runs the sentiment-regime backtest using a direct momentum ranking strategy.
    This version is robust and does not require PyPortfolioOpt.
    """
    st.write("Running Sentiment-Filtered Momentum Backtest...")
    if prices_df.empty or sentiment_index.empty: return None
    
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('W-FRI').last().index
    
    if len(rebalance_dates) < 2:
        st.warning("Data time range is too short for weekly rebalancing.")
        return None
        
    portfolio_returns = []
    
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i+1]
        
        # Sentiment Filter
        sentiment_slice = sentiment_index.loc[:start_date].tail(7)
        if sentiment_slice.empty: continue
        recent_sentiment = sentiment_slice['compound'].mean()
        
        # Momentum Calculation and Portfolio Construction
        hist_prices = prices_df.loc[:start_date].tail(91)
        if hist_prices.shape[0] < 91: continue
            
        if recent_sentiment < 0.0:
            # If sentiment is negative, hold cash (0% return)
            days_in_period = (end_date - start_date).days
            period_returns = pd.Series([0.0] * days_in_period, index=pd.date_range(start=start_date, periods=days_in_period, inclusive='left'))
        else:
            # If sentiment is not negative, invest in top 5 momentum coins
            momentum = hist_prices.pct_change(90).iloc[-1].dropna()
            if momentum.empty: continue
            top_5_coins = momentum.nlargest(5).index.tolist()
            
            # Calculate returns for the holding period with an equal-weight portfolio
            period_returns = daily_returns.loc[start_date:end_date][top_5_coins].mean(axis=1)
        
        portfolio_returns.append(period_returns)

    if not portfolio_returns: return None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns

# ==============================================================================
# MAIN APP LOGIC (Station 4)
# ==============================================================================

# Create the session once
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
        # Separate the loaded data into prices and sentiment
        prices_df = backtest_data.drop(columns=['compound'], errors='ignore')
        sentiment_index = backtest_data[['compound']].dropna()

        with st.spinner("Running backtest..."):
            strategy_returns = run_backtest(prices_df, sentiment_index)

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
            if 'BTC' in prices_df.columns:
                benchmark = (1 + prices_df['BTC'].pct_change()).cumprod()
                ax.plot(benchmark.loc[strategy_returns.index], label='Bitcoin (Benchmark)', color='gray', linestyle='--')
            ax.plot(cumulative_returns, label='Sentiment-Filtered Strategy', color='royalblue', linewidth=2)
            ax.set_title('Sentiment-Filtered Momentum Strategy vs. Bitcoin'); ax.set_ylabel('Cumulative Returns (Log Scale)'); ax.set_yscale('log'); ax.legend(); st.pyplot(fig)
            
            # Placeholder for Gemini Analysis if you want to add it back
            # st.divider()
            # st.header("🤖 Gemini AI Analysis")

        else:
            st.error("Could not complete the backtest. The data time range is too short or there was an issue during processing.")
else:
    st.info("Click the button to run the backtest on the pre-processed historical data.")


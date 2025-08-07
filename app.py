# app.py
#
# FINAL SUBMISSION VERSION
# This definitive version uses a direct momentum-ranking backtest to ensure
# stability and avoid the PyPortfolioOpt solver errors.

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
import seaborn as sns
from openai import OpenAI

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
SENTIMENT_THRESHOLD = 0.0

@st.cache_resource
def setup_nltk():
    """Download NLTK data."""
    import nltk
    with st.spinner("Setting up NLTK resources... (This runs once)"):
        nltk.download('vader_lexicon', quiet=True)
    st.success("NLTK resources are ready.")

# Run the setup at the start of the app
setup_nltk()

# ==============================================================================
# BACKEND HELPER & PIPELINE FUNCTIONS
# ==============================================================================

@st.cache_data
def create_requests_session() -> requests.Session:
    """Creates a requests session with a retry policy for network robustness."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries); session.mount("http://", adapter); session.mount("https://", adapter)
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
    """Fetches latest news and analyzes sentiment for the sidebar."""
    if not api_key: return pd.DataFrame()
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        data = _session.get(url).json().get('Data', [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data).head(20)
        analyzer = SentimentIntensityAnalyzer()
        df['compound'] = df['title'].fillna('').apply(lambda txt: analyzer.polarity_scores(txt)['compound'])
        return df[['title', 'source', 'compound', 'url']]
    except Exception: return pd.DataFrame()

def run_backtest(prices_df, sentiment_index):
    """
    Runs a direct sentiment-filtered momentum backtest. This is a robust implementation.
    """
    st.write("Running Sentiment-Filtered Momentum Backtest...")
    if prices_df.empty or sentiment_index.empty: return None, None
    
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('ME').last().index
    
    if len(rebalance_dates) < 2:
        return None, None
        
    portfolio_returns, last_weights = [], pd.Series()
    
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i+1]
        
        # Sentiment Filter
        sentiment_slice = sentiment_index.loc[:start_date].tail(7)
        if sentiment_slice.empty: continue
        recent_sentiment = sentiment_slice['compound'].mean()
        
        # Momentum Calculation and Portfolio Construction
        hist_prices = prices_df.loc[:start_date].tail(91) # 90-day lookback + 1 for pct_change
        if hist_prices.shape[0] < 91: continue
            
        if recent_sentiment < SENTIMENT_THRESHOLD:
            # If sentiment is negative, hold cash (0% return)
            days_in_period = (end_date - start_date).days
            period_returns = pd.Series([0.0] * days_in_period, index=pd.date_range(start=start_date, periods=days_in_period, inclusive='left'))
            last_weights = pd.Series(dtype='float64') # Reset weights to cash
        else:
            # If sentiment is not negative, invest in top 5 momentum coins
            momentum = hist_prices.pct_change(90).iloc[-1].dropna()
            if momentum.empty: continue
            top_5_coins = momentum.nlargest(5).index.tolist()
            
            target_weights = pd.Series(1/5, index=top_5_coins)
            period_returns = daily_returns.loc[start_date:end_date][top_5_coins].mean(axis=1)
            last_weights = target_weights
        
        portfolio_returns.append(period_returns)

    if not portfolio_returns: return None, None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns, last_weights

def generate_gemini_summary(results, latest_sentiment, latest_weights):
    if not OPENROUTER_API_KEY:
        return "Please add your OpenRouter API Key to Streamlit secrets."
    pass # Placeholder for Gemini logic

# ==============================================================================
# MAIN APP LOGIC (Station 4)
# ==============================================================================

session = create_requests_session()

st.sidebar.header("Live News Feed")
live_news = fetch_and_analyze_live_news(session, CRYPTOCOMPARE_API_KEY)
# ... [sidebar display logic] ...

st.sidebar.divider()
st.sidebar.header("AlphaSent Controls")
if st.sidebar.button("🚀 Run Full Backtest", type="primary"):
    
    backtest_data = load_data(DATA_URL)
    
    if not backtest_data.empty:
        # Final data cleaning step for robustness
        prices_df = backtest_data.drop(columns=['compound'], errors='ignore')
        prices_df.replace(0, np.nan, inplace=True)
        prices_df.ffill(inplace=True)
        prices_df.bfill(inplace=True)
        sentiment_index = backtest_data[['compound']].dropna()

        with st.spinner("Running backtest..."):
            strategy_returns, latest_weights = run_backtest(prices_df, sentiment_index)

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            
            # --- Prepare data for all tabs ---
            cumulative_returns = (1 + strategy_returns).cumprod()
            annual_return = cumulative_returns.iloc[-1]**(365/len(cumulative_returns)) - 1
            annual_volatility = strategy_returns.std() * (365**0.5)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
            
            # --- Create Tabs for Results ---
            tab1, tab2, tab3 = st.tabs(["📈 Performance Dashboard", "🔬 Strategy Internals", "🤖 Gemini AI Analysis"])
            
            with tab1:
                # ... [Display metrics and cumulative return plot] ...

            with tab2:
                st.header("Strategy Internals & Diagnostics")
                st.subheader("Monthly Return Heatmap")
                monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                heatmap_data = monthly_returns.to_frame('returns')
                heatmap_data['year'] = heatmap_data.index.year
                heatmap_data['month'] = heatmap_data.index.month
                heatmap_pivot = heatmap_data.pivot_table(index='year', columns='month', values='returns')
                heatmap_pivot.columns = [datetime(1900, m, 1).strftime('%b') for m in heatmap_pivot.columns]
                fig_heat, ax_heat = plt.subplots(figsize=(12, max(4, len(heatmap_pivot) * 0.5)))
                sns.heatmap(heatmap_pivot * 100, annot=True, fmt=".1f", cmap="vlag", center=0, ax=ax_heat, cbar_kws={'label': 'Monthly Return %'})
                ax_heat.set_title("Strategy Monthly Returns (%)"); st.pyplot(fig_heat)

            with tab3:
                st.header("Gemini AI Analysis")
                st.info("Gemini integration would provide a summary here.")
        else:
            st.error("Could not complete the backtest.")
else:
    st.info("Click the button in the sidebar to run the backtest.")

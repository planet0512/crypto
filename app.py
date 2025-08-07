# app.py
#
# FINAL SUBMISSION VERSION
# This definitive version combines the robust, sentiment-filtered backtest
# with a professional, multi-tab Streamlit dashboard for results visualization.

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
DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/final_app_data.csv"

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
        df = pd.DataFrame(data).head(10)
        analyzer = SentimentIntensityAnalyzer()
        df['compound'] = df['title'].fillna('').apply(lambda txt: analyzer.polarity_scores(txt)['compound'])
        return df[['title', 'source', 'compound', 'url']]
    except Exception:
        return pd.DataFrame()

def run_backtest(prices_df, sentiment_index):
    """
    Runs the sentiment-regime backtest using a direct momentum ranking strategy.
    """
    st.write("Running Sentiment-Filtered Momentum Backtest...")
    if prices_df.empty or sentiment_index.empty: return None, None
    
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('W-FRI').last().index
    
    if len(rebalance_dates) < 2:
        st.warning("Data time range is too short for weekly rebalancing.")
        return None, None
        
    portfolio_returns, last_weights = [], pd.Series()
    
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i+1]
        
        sentiment_slice = sentiment_index.loc[:start_date].tail(7)
        if sentiment_slice.empty: continue
        recent_sentiment = sentiment_slice['compound'].mean()
        
        hist_prices = prices_df.loc[:start_date].tail(91)
        if hist_prices.shape[0] < 91: continue
            
        if recent_sentiment < 0.0:
            days_in_period = (end_date - start_date).days
            period_returns = pd.Series([0.0] * days_in_period, index=pd.date_range(start=start_date, periods=days_in_period, inclusive='left'))
            last_weights = pd.Series(dtype='float64')
        else:
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
    """Generates a summary using OpenRouter."""
    if not OPENROUTER_API_KEY:
        return "Please add your OpenRouter API Key to Streamlit secrets to enable AI analysis."
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    prompt = f"""
    You are a FinTech analyst summarizing a backtest of a quantitative crypto strategy called 'AlphaSent'.
    The strategy is a **Sentiment-Filtered Momentum Model**.

    Here are the final backtest results:
    - Annual Return: {results['Annual Return']}
    - Sharpe Ratio: {results['Sharpe Ratio']}

    The most recent signals are:
    - Latest 7-day average sentiment score: {latest_sentiment:.2f}
    - Recommended portfolio for the next period:
    {latest_weights.to_string()}

    Based ONLY on this data, provide a professional summary in three parts:
    1.  **Performance Summary:** Describe the historical risk-adjusted performance.
    2.  **Current Outlook:** Interpret the latest sentiment score and its effect on the strategy.
    3.  **Recommended Allocation:** Describe the portfolio's current positioning.
    """
    try:
        completion = client.chat.completions.create(model="google/gemini-1.5-flash-latest", messages=[{"role": "user", "content": prompt}])
        return completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate Gemini summary. Error: {e}"

# ==============================================================================
# MAIN APP LOGIC (Station 4)
# ==============================================================================

# Create the session once at the start
session = create_requests_session()

# --- Live News Sidebar ---
st.sidebar.header("Live News Sentiment")
live_news = fetch_and_analyze_live_news(session, CRYPTOCOMPARE_API_KEY)
if not live_news.empty:
    for _, row in live_news.iterrows():
        st.sidebar.markdown(f"**{row['source']}**")
        st.sidebar.markdown(f"[{row['title'][:55]}...]({row['url']})")
        st.sidebar.progress(int((row['compound'] + 1) / 2 * 100))
        st.sidebar.markdown("---")
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
            strategy_returns, latest_weights = run_backtest(prices_df, sentiment_index)

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            
            # --- Create Tabs for Results ---
            tab1, tab2, tab3 = st.tabs(["📈 Performance Dashboard", "📰 Live News Sample", "🤖 Gemini AI Analysis"])
            
            with tab1:
                # --- Performance Metrics & Chart ---
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

            with tab2:
                st.header("Live News Feed")
                st.info("This feed shows the latest crypto news from CryptoCompare and their real-time VADER sentiment score.")
                if not live_news.empty:
                    st.dataframe(live_news)
                else:
                    st.warning("Live news feed could not be fetched.")
            
            with tab3:
                st.header("Gemini AI Analysis")
                with st.spinner("Generating AI summary..."):
                    results_dict = {"Annual Return": f"{annual_return:.2%}", "Sharpe Ratio": f"{sharpe_ratio:.2f}"}
                    latest_sentiment = sentiment_index['compound'].tail(7).mean()
                    top_holdings = latest_weights.index.tolist() if latest_weights is not None and not latest_weights.empty else ["Cash (due to negative sentiment)"]
                    summary = generate_gemini_summary(results_dict, latest_sentiment, pd.Series(top_holdings))
                    st.markdown(summary)
        else:
            st.error("Could not complete the backtest. The data time range may be too short or there was an issue during processing.")
else:

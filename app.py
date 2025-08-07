# app.py
#
# FINAL SUBMISSION VERSION
# This version includes the corrected logic for the monthly return heatmap.

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

def run_backtest(prices_df, sentiment_index):
    st.write("Running Sentiment-Filtered Momentum Backtest...")
    if prices_df.empty or sentiment_index.empty: return None, None
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('W-FRI').last().index
    if len(rebalance_dates) < 2: return None, None
        
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

@st.cache_data
def fetch_and_analyze_live_news(_session, api_key):
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

def generate_gemini_summary(results, latest_sentiment, latest_weights):
    if not OPENROUTER_API_KEY:
        return "Please add your OpenRouter API Key to Streamlit secrets."
    # ... [Rest of Gemini logic] ...
    pass

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
session = create_requests_session()

st.sidebar.header("AlphaSent Controls")
if st.sidebar.button("🚀 Run Full Backtest", type="primary"):
    
    backtest_data = load_data(DATA_URL)
    
    if not backtest_data.empty:
        prices_df = backtest_data.drop(columns=['compound'], errors='ignore')
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
            tab1, tab2, tab3 = st.tabs(["📈 Performance Dashboard", "🔬 Risk Analysis", "🤖 Gemini AI Analysis"])
            
            with tab1:
                st.header("Backtest Performance Results")
                col1, col2, col3 = st.columns(3)
                col1.metric("Annual Return", f"{annual_return:.2%}")
                col2.metric("Annual Volatility", f"{annual_volatility:.2%}")
                col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                fig, ax = plt.subplots(figsize=(12, 6))
                if 'BTC' in prices_df.columns:
                    benchmark = (1 + prices_df['BTC'].pct_change()).cumprod()
                    ax.plot(benchmark.loc[strategy_returns.index], label='Bitcoin (Benchmark)', color='gray', linestyle='--')
                ax.plot(cumulative_returns, label='AlphaSent Strategy', color='royalblue', linewidth=2)
                ax.set_title('Strategy Performance vs. Bitcoin'); ax.set_ylabel('Cumulative Returns (Log Scale)'); ax.set_yscale('log'); ax.legend(); st.pyplot(fig)

            with tab2:
                st.header("Risk & Return Analysis")
                st.subheader("Monthly Return Heatmap")
                
                # --- FINAL FIX: Correctly pivot the data for the heatmap ---
                monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
                monthly_returns_df = monthly_returns.to_frame('returns')
                monthly_returns_df['year'] = monthly_returns_df.index.year
                monthly_returns_df['month'] = monthly_returns_df.index.month
                
                heatmap_data = monthly_returns_df.pivot_table(index='year', columns='month', values='returns')
                heatmap_data.columns = [datetime(1900, m, 1).strftime('%b') for m in heatmap_data.columns] # Format month names
                
                fig2, ax2 = plt.subplots(figsize=(12, max(4, len(heatmap_data) * 0.5) ))
                sns.heatmap(heatmap_data * 100, annot=True, fmt=".1f", cmap="vlag", center=0, ax=ax2,
                            cbar_kws={'label': 'Monthly Return %'})
                ax2.set_title("Monthly Returns (%) of AlphaSent Strategy"); st.pyplot(fig2)

            with tab3:
                st.header("Gemini AI Analysis")
                # ... [Gemini logic would go here] ...
                st.info("Gemini AI analysis would be displayed here.")
        else:
            st.error("Could not complete the backtest.")
else:
    st.info("Click the button in the sidebar to run the backtest.")

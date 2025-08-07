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

def run_backtest(prices_df, sentiment_index):
    """
    Runs the sentiment-regime backtest using a direct momentum ranking strategy.
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
        
        sentiment_slice = sentiment_index.loc[:start_date].tail(7)
        if sentiment_slice.empty: continue
        recent_sentiment = sentiment_slice['compound'].mean()
        
        hist_prices = prices_df.loc[:start_date].tail(91)
        if hist_prices.shape[0] < 91: continue
            
        if recent_sentiment < 0.0:
            days_in_period = (end_date - start_date).days
            period_returns = pd.Series([0.0] * days_in_period, index=pd.date_range(start=start_date, periods=days_in_period, inclusive='left'))
        else:
            momentum = hist_prices.pct_change(90).iloc[-1].dropna()
            if momentum.empty: continue
            top_5_coins = momentum.nlargest(5).index.tolist()
            period_returns = daily_returns.loc[start_date:end_date][top_5_coins].mean(axis=1)
        
        portfolio_returns.append(period_returns)

    if not portfolio_returns: return None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns

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
        # --- FINAL FIX: Separate the data before calling the function ---
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
            
        else:
            st.error("Could not complete the backtest. The data time range may be too short or there was an issue during processing.")
else:
    st.info("Click the button to run the backtest on the pre-processed historical data.")



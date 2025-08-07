# app.py
#
# Final Streamlit app for the AlphaSent project.
# Loads pre-processed data from GitHub for speed and reliability.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from openai import OpenAI
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Project AlphaSent", page_icon="📈", layout="wide")
st.title("📈 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- CONFIGURATION ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

# --- GITHUB DATA URLS ---

MAIN_DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/final_app_data.csv"

NEWS_DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/processed_news_with_sentiment.csv"

# ==============================================================================
# BACKEND FUNCTIONS
# ==============================================================================
@st.cache_data
def load_data(url, is_news_data=False):
    st.write(f"Loading data from {url.split('/')[-1]}...")
    try:
        if is_news_data:
            df = pd.read_csv(url, index_col=0)
        else:
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.index.name = 'time'
        st.write("✓ Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}"); return pd.DataFrame()

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
        
        mu = expected_returns.mean_historical_return(hist_prices)
        S = risk_models.sample_cov(hist_prices)
        ef_mvo = EfficientFrontier(mu, S); ef_mvo.max_sharpe()
        mvo_weights = pd.Series(ef_mvo.clean_weights())
        ef_min_var = EfficientFrontier(mu, S); ef_min_var.min_volatility()
        min_var_weights = pd.Series(ef_min_var.clean_weights())

        target_weights = (mvo_weight * mvo_weights + min_var_weight * min_var_weights).fillna(0)
        turnover = (target_weights - last_weights.reindex(target_weights.index).fillna(0)).abs().sum() / 2
        costs = turnover * (25 / 10000)
        period_returns = (daily_returns.loc[start_date:end_date] * target_weights).sum(axis=1)
        if not period_returns.empty: period_returns.iloc[0] -= costs
        portfolio_returns.append(period_returns)
        last_weights = target_weights

    if not portfolio_returns: return None, None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns, last_weights
    
def generate_gemini_summary(results, latest_sentiment, latest_weights):
    if not OPENROUTER_API_KEY: return "Please add your OpenRouter API Key to Streamlit secrets."
    pass # Placeholder for Gemini logic

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
st.sidebar.header("AlphaSent Controls")
if st.sidebar.button("🚀 Run Backtest", type="primary"):
    app_data = load_data(MAIN_DATA_URL)
    if not app_data.empty:
        with st.spinner("Running backtest..."):
            strategy_returns, latest_weights = run_backtest(app_data)
        if strategy_returns is not None:
            st.success("Analysis Complete!")
            # Display results...
        else:
            st.error("Could not complete backtest. Data time range may be too short.")
else:
    st.info("Click the button in the sidebar to run the backtest.")

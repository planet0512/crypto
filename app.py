# app.py
#
# Final version for submission. Loads all data from the user's GitHub repository
# and displays backtest results and a table of the latest news with sentiment.

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from openai import OpenAI

# Import PyPortfolioOpt components
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Project AlphaSent", page_icon="📈", layout="wide")

st.title("📈 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- CONFIGURATION ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
# --- YOUR GITHUB URLs ---
MAIN_DATA_URL = "https://github.com/planet0512/crypto/main/final_app_data.csv"
NEWS_DATA_URL = "https://github.com/planet0512/crypto/main/processed_news_with_sentiment.csv"

# ==============================================================================
# BACKEND FUNCTIONS
# ==============================================================================

@st.cache_data
def load_data(url, is_news_data=False):
    """Loads a CSV file from a raw GitHub URL."""
    st.write(f"Loading data from {url.split('/')[-1]}...")
    try:
        if is_news_data:
            # The news data does not have a date index
            df = pd.read_csv(url, index_col=0)
        else:
            # The main app data uses the first column as the date index
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.index.name = 'time'
        
        st.write("✓ Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub URL: {url}. Error: {e}")
        return pd.DataFrame()

def run_backtest(full_data_df):
    """Runs the backtest on the pre-loaded and aligned data."""
    st.write("Running Sentiment-Regime Backtest...")
    
    prices_df = full_data_df.drop(columns=['compound'], errors='ignore')
    sentiment_index = full_data_df[['compound']].dropna()

    if prices_df.empty or sentiment_index.empty: return None, None
    
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('W-FRI').last().index
    
    if len(rebalance_dates) < 2:
        st.warning("Data time range is too short for the weekly rebalancing frequency.")
        return None, None
        
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
        with st.spinner("Running backtest on historical data..."):
            strategy_returns, latest_weights = run_backtest(app_data)

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
            prices_df = app_data.drop(columns=['compound'], errors='ignore')
            if 'BTC' in prices_df.columns:
                benchmark = (1 + prices_df['BTC'].pct_change()).cumprod()
                ax.plot(benchmark.loc[strategy_returns.index], label='Bitcoin (Benchmark)', color='gray', linestyle='--')
            ax.plot(cumulative_returns, label='Sentiment-Regime Strategy', color='royalblue', linewidth=2)
            ax.set_title('Sentiment-Regime Strategy vs. Bitcoin'); ax.set_ylabel('Cumulative Returns (Log Scale)'); ax.set_yscale('log'); ax.legend(); st.pyplot(fig)

            st.divider()

            st.header("📰 Latest News & Sentiment Scores")
            st.write("This table shows a sample of the most recent news articles used to generate the sentiment index.")
            news_display_df = load_data(NEWS_DATA_URL, is_news_data=True)
            if not news_display_df.empty:
                news_display_df['date'] = pd.to_datetime(news_display_df['date']).dt.strftime('%Y-%m-%d %H:%M')
                st.dataframe(news_display_df[['date', 'TITLE', 'compound']].head(15))

            # (Gemini integration would go here)
        else:
            st.error("Could not complete the backtest. The data time range may be too short for the rebalancing frequency.")
else:
    st.info("Click the button in the sidebar to run the backtest on the pre-processed data from GitHub.")

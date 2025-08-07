# app.py
#
# Final submission version. Loads historical price data and RAW news data
# from GitHub, then runs the full sentiment and backtesting pipeline live.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pypfopt import EfficientFrontier, risk_models, expected_returns
import matplotlib.pyplot as plt

# ==============================================================================
# PAGE CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(page_title="Project AlphaSent", page_icon="📈", layout="wide")

st.title("📈 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- CONFIGURATION ---
# Corrected RAW GitHub URLs
PRICE_DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/final_app_data.csv"
NEWS_DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/stage_1_news_raw.csv.gz"

# --- NLTK SETUP ---
@st.cache_resource
def setup_nltk():
    import nltk
    with st.spinner("Setting up NLTK resources..."):
        nltk.download('vader_lexicon', quiet=True)
setup_nltk()

# ==============================================================================
# BACKEND FUNCTIONS
# ==============================================================================

@st.cache_data
def load_data(url, is_news=False):
    """Loads and prepares data from a GitHub URL."""
    st.write(f"Loading data from {url.split('/')[-1]}...")
    try:
        if is_news:
            # Load and decompress the raw news data
            df = pd.read_csv(url, compression='gzip', index_col=0)
        else:
            # Load the pre-processed price data
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.index.name = 'time'
        st.write("✓ Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}"); return pd.DataFrame()

@st.cache_data
def run_sentiment_pipeline(news_df: pd.DataFrame) -> pd.DataFrame:
    """Processes raw news data and calculates daily sentiment."""
    st.write("Running Sentiment Pipeline on raw news data...")
    if news_df.empty: return pd.DataFrame()
    
    df = news_df.copy()
    # Ensure column names are consistent
    df.columns = [col.lower() for col in df.columns]
    
    df['text_to_analyze'] = df['title'].fillna('') + ". " + df['body'].fillna('')
    df['clean_text'] = df['text_to_analyze'].apply(lambda text: re.sub(r'[^A-Za-z\s]+', '', BeautifulSoup(str(text), "html.parser").get_text()).lower().strip())
    
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df['clean_text'].apply(lambda text: analyzer.polarity_scores(text))
    df = pd.concat([df[['published_on']], sentiment_scores.apply(pd.Series)], axis=1)
    
    df['date'] = pd.to_datetime(df['published_on'], unit='s')
    df['date_only'] = df['date'].dt.date
    
    daily_sentiment_index = df.groupby('date_only')[['compound']].mean()
    daily_sentiment_index.index = pd.to_datetime(daily_sentiment_index.index)
    st.write("✓ Daily sentiment index created."); return daily_sentiment_index

def run_backtest(prices_df, sentiment_index):
    """Runs the sentiment-regime backtest."""
    st.write("Running Sentiment-Regime Backtest...")
    if prices_df.empty or sentiment_index.empty: return None
    
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('W-FRI').last().index
    if len(rebalance_dates) < 2: return None
        
    portfolio_returns = []
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
        costs = (target_weights - portfolio_returns[-1].name if portfolio_returns else target_weights).abs().sum() / 2 * (25 / 10000)
        
        period_returns = (daily_returns.loc[start_date:end_date] * target_weights).sum(axis=1)
        if not period_returns.empty: period_returns.iloc[0] -= costs
        portfolio_returns.append(period_returns)

    if not portfolio_returns: return None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
st.sidebar.header("AlphaSent Controls")

if st.sidebar.button("🚀 Run Full Analysis", type="primary"):
    
    # Load both data files from GitHub
    prices_df = load_data(PRICE_DATA_URL)
    raw_news_df = load_data(NEWS_DATA_URL, is_news=True)
    
    if not prices_df.empty and not raw_news_df.empty:
        with st.spinner("Processing sentiment and running backtest..."):
            
            sentiment_index = run_sentiment_pipeline(raw_news_df)
            
            # Align data
            common_start_date = max(prices_df.index.min(), sentiment_index.index.min())
            st.write(f"Aligning data... Backtest will run from {common_start_date.date()}.")
            prices_df = prices_df[prices_df.index >= common_start_date]
            sentiment_index = sentiment_index[sentiment_index.index >= common_start_date]
            
            # Combine for the backtester
            full_data = prices_df.merge(sentiment_index, left_index=True, right_index=True, how='left').ffill()

            strategy_returns = run_backtest(full_data)

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            
            # --- Display Results ---
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
            ax.plot(cumulative_returns, label='Sentiment-Regime Strategy', color='royalblue', linewidth=2)
            ax.set_title('Sentiment-Regime Strategy vs. Bitcoin'); ax.set_ylabel('Cumulative Returns (Log Scale)'); ax.set_yscale('log'); ax.legend(); st.pyplot(fig)
        else:
            st.error("Could not complete the backtest. The data time range may be too short.")
else:
    st.info("Click the button in the sidebar to run the analysis.")

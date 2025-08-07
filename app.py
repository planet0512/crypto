

# app.py
#
# FINAL SUBMISSION VERSION.
# - Loads pre-processed price data from final_app_data.csv.
# - Loads RAW news data from stage_1_news_raw.csv.gz.
# - Runs the full sentiment and backtesting pipeline in the app.

import streamlit as st
import pandas as pd
import numpy as np
import re
from datetime import datetime
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ==============================================================================
# PAGE CONFIGURATION & SETUP
# ==============================================================================
st.set_page_config(page_title="Project AlphaSent", page_icon="📈", layout="wide")
st.title("📈 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- CONFIGURATION ---
# IMPORTANT: These must be the 'raw' URLs from your public GitHub repository
PRICE_DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/final_app_data.csv"
NEWS_DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/refs/heads/main/stage_1_news_raw.csv.gz"

@st.cache_resource
def setup_nltk():
    import nltk; nltk.download('vader_lexicon', quiet=True)
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
            df = pd.read_csv(url, compression='gzip', low_memory=False)
        else:
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.index.name = 'time'
        st.write("✓ Data loaded successfully.")
        return df
    except Exception as e:
        st.error(f"Error loading data from GitHub: {e}"); return pd.DataFrame()

@st.cache_data
def run_sentiment_pipeline(raw_news_df: pd.DataFrame) -> pd.DataFrame:
    """Processes raw news data and calculates daily sentiment."""
    st.write("Running Sentiment Pipeline on raw news data...")
    if raw_news_df.empty: return pd.DataFrame()
    df = raw_news_df.copy()
    df.columns = [col.lower() for col in df.columns]
    
    df['published_on'] = pd.to_numeric(df['published_on'], errors='coerce')
    df.dropna(subset=['published_on'], inplace=True)
    
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
    
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i+1]
        
        # Get the single sentiment score for the day
        if start_date.date() not in sentiment_index.index.date: continue
        sentiment_signal = sentiment_index.loc[str(start_date.date())]['compound'].iloc[0]
        if pd.isna(sentiment_signal): continue
        
        # Simple Strategy: If sentiment > 0, go long BTC. If < 0, hold cash.
        if sentiment_signal > 0:
            period_returns = daily_returns.loc[start_date:end_date]['BTC']
        else:
            period_returns = pd.Series(0, index=[end_date])

        portfolio_returns.append(period_returns)

    if not portfolio_returns: return None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns


# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
st.sidebar.header("AlphaSent Controls")
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    
    # Load both data files from GitHub
    prices_df = load_data(PRICE_DATA_URL)
    raw_news_df = load_data(RAW_NEWS_URL, is_news=True)
    
    if not prices_df.empty and not raw_news_df.empty:
        with st.spinner("Processing sentiment and running backtest..."):
            sentiment_index = run_sentiment_pipeline(raw_news_df)
            
            # Align data by merging
            full_data = prices_df.merge(sentiment_index, left_index=True, right_index=True, how='left').ffill()
            
            strategy_returns = run_backtest(full_data.drop(columns=['compound']), full_data[['compound']])

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            # Display results...
        else:
            st.error("Could not complete the backtest.")
else:
    st.info("Click the button in the sidebar to run the analysis.")

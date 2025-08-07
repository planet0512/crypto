# app.py
#
# FINAL SUBMISSION VERSION.
# - Includes a robust data cleaning step to handle corrupted rows in the news file.
# - Runs a 1-month backtest as requested for a fast and reliable demonstration.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import re
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
# The app will load these two files from your GitHub
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
def run_sentiment_pipeline(news_df: pd.DataFrame) -> pd.DataFrame:
    """Processes raw news data and calculates daily sentiment."""
    st.write("Running Sentiment Pipeline on raw news data...")
    if news_df.empty: return pd.DataFrame()
    
    df = news_df.copy()
    df.columns = [col.lower() for col in df.columns]

    # --- FINAL FIX: Clean the 'published_on' column ---
    # Convert to numeric, turning any non-numeric text into NaN
    df['published_on'] = pd.to_numeric(df['published_on'], errors='coerce')
    # Drop the rows with corrupted data
    df.dropna(subset=['published_on'], inplace=True)
    df['published_on'] = df['published_on'].astype(int)
    
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
    # Use daily rebalancing for the short 1-month period
    rebalance_dates = prices_df.resample('D').last().index 
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

if st.sidebar.button("🚀 Run 1-Month Analysis", type="primary"):
    
    prices_df_full = load_data(PRICE_DATA_URL)
    raw_news_df = load_data(NEWS_DATA_URL, is_news=True)
    
    if not prices_df_full.empty and not raw_news_df.empty:
        with st.spinner("Processing sentiment and running backtest..."):
            
            sentiment_index = run_sentiment_pipeline(raw_news_df)
            
            # --- Use last 30 days of data for the backtest ---
            start_date = datetime.now() - timedelta(days=30)
            prices_df_monthly = prices_df_full[prices_df_full.index >= start_date]
            sentiment_index_monthly = sentiment_index[sentiment_index.index >= start_date]
            
            strategy_returns = run_backtest(prices_df_monthly, sentiment_index_monthly)

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            
            # --- Display Results ---
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            st.header("1-Month Backtest Performance")
            col1, col2 = st.columns(2)
            col1.metric("Total Return", f"{(cumulative_returns.iloc[-1] - 1):.2%}")
            col2.metric("Volatility (Annualized)", f"{(strategy_returns.std() * np.sqrt(365)):.2%}")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(cumulative_returns, label='Sentiment Strategy', color='royalblue', linewidth=2)
            
            btc_benchmark = (1 + prices_df_monthly['BTC'].pct_change()).cumprod()
            ax.plot(btc_benchmark.loc[cumulative_returns.index], label='Bitcoin (Benchmark)', color='gray', linestyle='--')
            
            ax.set_title('1-Month Strategy Performance vs. Bitcoin'); ax.set_ylabel('Cumulative Returns'); ax.legend(); ax.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.error("Could not complete the backtest.")
else:
    st.info("Click the button in the sidebar to run the analysis.")

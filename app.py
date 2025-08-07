# app.py
#
# FINAL SUBMISSION VERSION for the AlphaSent Project.
# This app features a multi-tab dashboard to showcase the sentiment-regime
# backtest, live news analysis, and Gemini AI integration.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI
from pypfopt import EfficientFrontier, risk_models, expected_returns
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

@st.cache_resource
def setup_nltk():
    import nltk; nltk.download('vader_lexicon', quiet=True)
setup_nltk()

# ==============================================================================
# BACKEND FUNCTIONS
# ==============================================================================

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

def run_backtest(prices_df, sentiment_index):
    """Runs the full sentiment-regime backtest."""
    st.write("Running Sentiment-Regime Backtest...")
    if prices_df.empty or sentiment_index.empty: return None, None
    
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('ME').last().index
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
        
        try:
            mu = expected_returns.mean_historical_return(hist_prices)
            S = risk_models.sample_cov(hist_prices)
            ef_mvo = EfficientFrontier(mu, S); ef_mvo.max_sharpe()
            mvo_weights = pd.Series(ef_mvo.clean_weights())
            ef_min_var = EfficientFrontier(mu, S); ef_min_var.min_volatility()
            min_var_weights = pd.Series(ef_min_var.clean_weights())
            target_weights = (mvo_weight * mvo_weights + min_var_weight * min_var_weights).fillna(0)
        except Exception:
            target_weights = pd.Series(1/len(hist_prices.columns), index=hist_prices.columns)

        turnover = (target_weights - last_weights.reindex(target_weights.index).fillna(0)).abs().sum() / 2
        costs = turnover * (25 / 10000)
        period_returns = (daily_returns.loc[start_date:end_date] * target_weights).sum(axis=1)
        if not period_returns.empty: period_returns.iloc[0] -= costs
        portfolio_returns.append(period_returns)
        last_weights = target_weights

    if not portfolio_returns: return None, None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete."); return strategy_returns, last_weights

@st.cache_data
def fetch_and_analyze_live_news(_session, api_key):
    """Fetches latest news and analyzes sentiment for display."""
    if not api_key: return pd.DataFrame()
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        data = _session.get(url).json().get('Data', [])
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data).head(15) # Get top 15 articles
        analyzer = SentimentIntensityAnalyzer()
        df['compound'] = df['title'].fillna('').apply(lambda txt: analyzer.polarity_scores(txt)['compound'])
        return df[['title', 'source', 'compound', 'url']]
    except Exception: return pd.DataFrame()

def generate_gemini_summary(results, latest_sentiment, latest_weights):
    """Generates a summary using OpenRouter and the OpenAI library."""
    if not OPENROUTER_API_KEY:
        return "Please add your OpenRouter API Key to Streamlit secrets to enable AI analysis."
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    
    prompt = f"""
    You are a FinTech analyst summarizing a backtest of a quantitative crypto strategy called 'AlphaSent'.
    The strategy is a **Sentiment-Regime Switching Model**. It blends an aggressive portfolio (Max Sharpe Ratio) with a defensive portfolio (Minimum Variance) based on news sentiment.

    Here are the final backtest results:
    - Annual Return: {results['Annual Return']}
    - Sharpe Ratio: {results['Sharpe Ratio']}

    The current market state is:
    - Latest 7-day average sentiment score: {latest_sentiment:.2f}

    The final recommended portfolio allocation for the next period is:
    {latest_weights.to_string()}

    Based ONLY on this data, provide a professional summary in three parts:
    1.  **Performance Summary:** Describe the historical risk-adjusted performance.
    2.  **Current Outlook:** Interpret the latest sentiment score and its implication for the strategy's current risk posture (Risk-On or Risk-Off).
    3.  **Recommended Allocation:** Describe the portfolio's recommended holdings.
    """
    try:
        completion = client.chat.completions.create(
            model="google/gemini-1.5-flash-latest",
            messages=[{"role": "user", "content": prompt}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate Gemini summary. Error: {e}"

# ==============================================================================
# MAIN APP LOGIC (Station 4)
# ==============================================================================

# --- Sidebar ---
st.sidebar.header("AlphaSent Controls")
run_button = st.sidebar.button("🚀 Run Full Backtest", type="primary")
st.sidebar.divider()
st.sidebar.header("Live News Feed")
session = create_requests_session()
live_news = fetch_and_analyze_live_news(session, CRYPTOCOMPARE_API_KEY)
if not live_news.empty:
    for _, row in live_news.iterrows():
        st.sidebar.markdown(f"**{row['source']}** \n[{row['title'][:50]}...]({row['url']})")
        st.sidebar.progress(int((row['compound'] + 1) / 2 * 100))
else:
    st.sidebar.info("Live news feed unavailable. Add your CryptoCompare API key to secrets.")

# --- Main Panel ---
if run_button:
    backtest_data = load_data(DATA_URL)
    
    if not backtest_data.empty:
        with st.spinner("Running full historical backtest..."):
            strategy_returns, latest_weights = run_backtest(backtest_data)

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            
            # --- Create Tabs for Results ---
            tab1, tab2, tab3 = st.tabs(["📈 Performance Dashboard", "📰 Latest News Sample", "🤖 Gemini AI Analysis"])
            
            with tab1:
                st.header("Backtest Performance Results")
                cumulative_returns = (1 + strategy_returns).cumprod()
                annual_return = cumulative_returns.iloc[-1]**(365/len(cumulative_returns)) - 1
                annual_volatility = strategy_returns.std() * (365**0.5)
                sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Annual Return", f"{annual_return:.2%}")
                col2.metric("Annual Volatility", f"{annual_volatility:.2%}")
                col3.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

                fig, ax = plt.subplots(figsize=(12, 6))
                prices_df = backtest_data.drop(columns=['compound'], errors='ignore')
                if 'BTC' in prices_df.columns:
                    benchmark = (1 + prices_df['BTC'].pct_change()).cumprod()
                    ax.plot(benchmark.loc[strategy_returns.index], label='Bitcoin (Benchmark)', color='gray', linestyle='--')
                ax.plot(cumulative_returns, label='AlphaSent Strategy', color='royalblue', linewidth=2)
                ax.set_title('Strategy Performance vs. Bitcoin'); ax.set_ylabel('Cumulative Returns (Log Scale)'); ax.set_yscale('log'); ax.legend(); st.pyplot(fig)

            with tab2:
                st.header("Sample of News Data Used in Backtest")
                st.info("This is a sample of the raw news data used to generate the historical sentiment index.")
                # Load the other CSV for display
                news_url = DATA_URL.replace("final_app_data.csv", "processed_news_with_sentiment.csv")
                news_sample_df = pd.read_csv(news_url, index_col=0) if "planet0512" in news_url else pd.DataFrame() # Avoids error if file doesn't exist
                if not news_sample_df.empty:
                    news_sample_df['date'] = pd.to_datetime(news_sample_df['date']).dt.strftime('%Y-%m-%d %H:%M')
                    st.dataframe(news_sample_df[['date', 'TITLE', 'compound']].head(20))
            
            with tab3:
                st.header("Gemini AI Analysis")
                with st.spinner("Generating AI summary..."):
                    results_dict = {"Annual Return": f"{annual_return:.2%}", "Sharpe Ratio": f"{sharpe_ratio:.2f}"}
                    latest_sentiment = backtest_data['compound'].tail(7).mean()
                    top_holdings = latest_weights[latest_weights > 0.01].sort_values(ascending=False) if latest_weights is not None else pd.Series(["Cash"])
                    summary = generate_gemini_summary(results_dict, latest_sentiment, top_holdings)
                    st.markdown(summary)
        else:
            st.error("Could not complete the backtest. The historical data's time range may be too short.")
else:
    st.info("Click the button in the sidebar to run the backtest on the pre-processed historical data.")

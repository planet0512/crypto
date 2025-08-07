# app.py
#
# Final version for Streamlit Cloud deployment.
# Includes fixes for the caching (TypeError) and backtest (IndexError) issues.

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from pathlib import Path
import time
import re
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import matplotlib.pyplot as plt
from openai import OpenAI

# Import PyPortfolioOpt components
from pypfopt import EfficientFrontier, risk_models, expected_returns

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(
    page_title="Project AlphaSent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- API KEY CONFIGURATION ---
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")

# Ensure NLTK data is available
try:
    from nltk.corpus import stopwords
    stopwords.words('english')
except LookupError:
    import nltk
    with st.spinner("Downloading NLTK data..."):
        nltk.download('vader_lexicon')
        nltk.download('stopwords')
        nltk.download('punkt')
    st.success("NLTK data ready.")

# ==============================================================================
# BACKEND FUNCTIONS (Stations 1, 2, 3)
# ==============================================================================

@st.cache_data
def create_requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter); session.mount("https://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session

@st.cache_data
def get_top_coins(_session, limit=15) -> list:
    st.write(f"Fetching Top {limit} Coins by Market Cap...")
    core_assets, url = ['BTC', 'ETH'], f"https://min-api.cryptocompare.com/data/top/mktcapfull?limit=25&tsym=USD"
    try:
        data = _session.get(url).json()['Data']
        stablecoins = {'USDT', 'USDC', 'DAI', 'BUSD'}
        final_list = list(core_assets)
        for coin_info in data:
            symbol = coin_info['CoinInfo']['Name']
            if symbol not in final_list and symbol not in stablecoins: final_list.append(symbol)
            if len(final_list) >= limit: break
        st.write(f"✓ Identified Top {len(final_list)} coins.")
        return final_list
    except Exception as e:
        st.error(f"Error fetching top coins: {e}. Using a fallback list.")
        return ['BTC', 'ETH', 'SOL', 'XRP']

@st.cache_data
def fetch_market_data(_session, symbol, limit=2000) -> pd.DataFrame:
    url = "https://min-api.cryptocompare.com/data/v2/histoday"
    params = {"fsym": symbol, "tsym": "USD", "limit": limit}
    try:
        data = _session.get(url, params=params).json()["Data"]["Data"]
        if not data: return pd.DataFrame()
        df = pd.DataFrame(data)
        df['time'] = pd.to_datetime(df['time'], unit="s")
        return df.set_index('time')[['close']]
    except Exception: return pd.DataFrame()

@st.cache_data
def fetch_news_range(_session, start_dt, end_dt):
    st.write(f"Fetching news from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}...")
    url, out, current_end_dt = "https://data-api.coindesk.com/news/v1/article/list", [], end_dt
    while current_end_dt > start_dt:
        to_ts = int(current_end_dt.timestamp())
        try:
            r = _session.get(f"{url}?lang=EN&to_ts={to_ts}")
            d = pd.DataFrame(r.json()["Data"]); d["date"] = pd.to_datetime(d["PUBLISHED_ON"], unit="s")
            out.append(d)
            current_end_dt = datetime.fromtimestamp(d["PUBLISHED_ON"].min() - 1)
        except Exception: break
    if not out: return pd.DataFrame()
    
    # --- FIX 1: Select only necessary columns to prevent caching (unhashable type: 'dict') error ---
    final_df = pd.concat(out, ignore_index=True)
    required_cols = ['date', 'PUBLISHED_ON', 'TITLE', 'BODY', 'URL']
    existing_cols = [col for col in required_cols if col in final_df.columns]
    final_df = final_df[existing_cols]

    final_df = final_df.drop_duplicates(subset=['URL'])
    st.write(f"✓ Fetched {len(final_df)} articles.")
    return final_df

@st.cache_data
def run_sentiment_pipeline(news_df: pd.DataFrame) -> pd.DataFrame:
    st.write("Running Sentiment Pipeline...")
    if news_df.empty: return pd.DataFrame()
    df = news_df.copy()
    df['text_to_analyze'] = df['TITLE'].fillna('') + ". " + df['BODY'].fillna('')
    df['clean_text'] = df['text_to_analyze'].apply(lambda text: re.sub(r'[^A-Za-z\s]+', '', BeautifulSoup(text, "html.parser").get_text()).lower().strip())
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = df['clean_text'].apply(lambda text: analyzer.polarity_scores(text))
    df = pd.concat([df[['date']], sentiment_scores.apply(pd.Series)], axis=1)
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    daily_sentiment_index = df.groupby('date_only')[['compound']].mean()
    daily_sentiment_index.index = pd.to_datetime(daily_sentiment_index.index)
    st.write("✓ Daily sentiment index created.")
    return daily_sentiment_index

def get_portfolio_weights(prices, model="mvo"):
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, S)
    try:
        if model == "mvo": ef.max_sharpe()
        elif model == "min_var": ef.min_volatility()
        return pd.Series(ef.clean_weights())
    except Exception: return pd.Series({ticker: 1/len(prices.columns) for ticker in prices.columns})

def run_backtest(prices_df, sentiment_index):
    st.write("Running Sentiment-Regime Backtest...")
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('ME').last().index
    portfolio_returns = []
    last_weights = pd.Series()
    sentiment_zscore = (sentiment_index['compound'] - sentiment_index['compound'].rolling(90).mean()) / sentiment_index['compound'].rolling(90).std()
    
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i+1]
        
        # --- FIX 2: Check if sentiment data exists for this date to prevent IndexError ---
        sentiment_slice = sentiment_zscore.loc[:start_date]
        if sentiment_slice.empty:
            continue # Skip rebalance if no sentiment data is available yet

        sentiment_signal = sentiment_slice.iloc[-1]
        if pd.isna(sentiment_signal): sentiment_signal = 0

        mvo_weight, min_var_weight = (0.8, 0.2) if sentiment_signal > 1.0 else (0.2, 0.8)
        hist_prices = prices_df.loc[:start_date].tail(90)
        if hist_prices.shape[0] < 90: continue
        
        mvo_weights = get_portfolio_weights(hist_prices, model="mvo")
        min_var_weights = get_portfolio_weights(hist_prices, model="min_var")
        target_weights = (mvo_weight * mvo_weights + min_var_weight * min_var_weights).fillna(0)
        
        turnover = (target_weights - last_weights.reindex(target_weights.index).fillna(0)).abs().sum() / 2
        costs = turnover * (25 / 10000)
        
        period_returns = (daily_returns.loc[start_date:end_date] * target_weights).sum(axis=1)
        period_returns.iloc[0] -= costs
        portfolio_returns.append(period_returns)
        last_weights = target_weights

    if not portfolio_returns: return None, None
    strategy_returns = pd.concat(portfolio_returns)
    st.write("✓ Backtest complete.")
    return strategy_returns, last_weights
    
def generate_gemini_summary(results, latest_sentiment, latest_weights):
    if not OPENROUTER_API_KEY:
        return "Please add your OpenRouter API Key to Streamlit secrets to enable this feature."
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    prompt_content = f"""...""" # Prompt from previous turn
    try:
        completion = client.chat.completions.create(model="google/gemini-1.5-flash-latest", messages=[{"role": "user", "content": prompt_content}])
        return completion.choices[0].message.content
    except Exception as e:
        return f"Could not generate Gemini summary. Error: {e}"

# ==============================================================================
# MAIN APP LOGIC (Station 4)
# ==============================================================================

if st.button("🚀 Run Full Analysis & Backtest", type="primary"):
    
    with st.spinner("Running pipeline... This may take a few minutes."):
        session = create_requests_session()
        top_coins = get_top_coins(session)
        all_prices = {coin: fetch_market_data(session, coin) for coin in top_coins}
        news_df = fetch_news_range(session, datetime.now() - timedelta(days=365), datetime.now())
        prices_df = pd.concat({coin: df['close'] for coin, df in all_prices.items() if not df.empty}, axis=1).ffill()
        sentiment_index = run_sentiment_pipeline(news_df)
        
        # --- Align data starting dates before backtest ---
        if not sentiment_index.empty:
            first_sentiment_date = sentiment_index.dropna().index.min()
            prices_df = prices_df[prices_df.index >= first_sentiment_date]

        strategy_returns, latest_weights = run_backtest(prices_df, sentiment_index)

    if strategy_returns is not None:
        st.success("Analysis Complete!")
        
        # Display Results
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

        st.divider()
        st.header("🤖 Gemini AI Analysis")
        with st.spinner("Generating AI summary..."):
            results_dict = {"Annual Return": f"{annual_return:.2%}", "Annual Volatility": f"{annual_volatility:.2%}", "Sharpe Ratio": f"{sharpe_ratio:.2f}"}
            latest_sentiment = sentiment_index.tail(7)['compound'].mean()
            top_holdings = latest_weights[latest_weights > 0.01].sort_values(ascending=False)
            summary = generate_gemini_summary(results_dict, latest_sentiment, top_holdings)
            st.markdown(summary)
            
    else:
        st.error("Could not complete the backtest. Please check the console for errors.")

else:
    st.info("Click the button above to run the full data pipeline and backtest.")

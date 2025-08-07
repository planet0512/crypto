# app.py
#
# FINAL SUBMISSION VERSION
# This definitive version implements the full "Sentiment-as-Regime-Switch" model,
# blending MVO, MinVar, and ERC portfolios, and displays results in a multi-tab dashboard.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.objective_functions import L2_reg
from pypfopt.exceptions import OptimizationError

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Project AlphaSent", page_icon="🔭", layout="wide")
st.title("🔭 Project AlphaSent")
st.subheader("A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation")

# --- CONFIGURATION ---
DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
SENTIMENT_ZSCORE_THRESHOLD = 1.0

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

def get_portfolio_weights(prices, model="max_sharpe"):
    """Calculates optimal portfolio weights using PyPortfolioOpt."""
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.sample_cov(prices)
    
    # ERC does not use EfficientFrontier
    if model == "erc":
        # Simple inverse volatility as a proxy for ERC, as PyPortfolioOpt's ERC is complex to implement here
        inv_vol = 1 / prices.pct_change().std()
        return inv_vol / inv_vol.sum()

    ef = EfficientFrontier(mu, S)
    ef.add_objective(L2_reg)
    try:
        if model == "max_sharpe": ef.max_sharpe()
        elif model == "min_variance": ef.min_volatility()
        weights = ef.clean_weights()
        return pd.Series(weights)
    except (OptimizationError, ValueError):
        return pd.Series(1/len(prices.columns), index=prices.columns)

def run_backtest(prices_df, sentiment_index):
    """Runs the full Sentiment-as-Regime-Switch backtest."""
    st.write("Running Sentiment-Regime Backtest...")
    daily_returns = prices_df.pct_change()
    rebalance_dates = prices_df.resample('ME').last().index
    if len(rebalance_dates) < 2: return None, None, None, None
        
    portfolio_returns, last_weights, regime_history, allocation_history = [], pd.Series(), [], []
    sentiment_zscore = (sentiment_index['compound'] - sentiment_index['compound'].rolling(90).mean()) / sentiment_index['compound'].rolling(90).std()
    
    for i in range(len(rebalance_dates) - 1):
        start_date, end_date = rebalance_dates[i], rebalance_dates[i+1]
        sentiment_slice = sentiment_zscore.loc[:start_date].dropna()
        if sentiment_slice.empty: continue
        sentiment_signal = sentiment_slice.iloc[-1]
        if pd.isna(sentiment_signal): sentiment_signal = 0
        
        is_risk_on = sentiment_signal > SENTIMENT_ZSCORE_THRESHOLD
        regime_history.append({'date': start_date, 'regime': 1 if is_risk_on else 0})
        
        if is_risk_on: # Risk-On: Blend MVO and ERC
            mvo_blend, min_var_blend, erc_blend = 0.8, 0.0, 0.2
        else: # Risk-Off: Blend Minimum Variance and ERC
            mvo_blend, min_var_blend, erc_blend = 0.0, 0.5, 0.5
            
        allocation_history.append({'date': start_date, 'MVO': mvo_blend, 'MinVar': min_var_blend, 'ERC': erc_blend})
        
        hist_prices = prices_df.loc[:start_date].tail(90)
        if hist_prices.shape[0] < 90: continue
        
        # Drop columns that have no data in the lookback window
        hist_prices = hist_prices.dropna(axis=1, how='all').ffill()
        if hist_prices.shape[1] < 2: continue
        
        mvo_weights = get_portfolio_weights(hist_prices, model="max_sharpe")
        min_var_weights = get_portfolio_weights(hist_prices, model="min_variance")
        erc_weights = get_portfolio_weights(hist_prices, model="erc")
        
        target_weights = (mvo_blend * mvo_weights + min_var_blend * min_var_weights + erc_blend * erc_weights).fillna(0)
        costs = (target_weights - last_weights.reindex(target_weights.index).fillna(0)).abs().sum() / 2 * (25 / 10000)
        period_returns = (daily_returns.loc[start_date:end_date] * target_weights).sum(axis=1)
        if not period_returns.empty: period_returns.iloc[0] -= costs
        portfolio_returns.append(period_returns); last_weights = target_weights

    if not portfolio_returns: return None, None, None, None
    strategy_returns = pd.concat(portfolio_returns)
    regime_df = pd.DataFrame(regime_history).set_index('date')
    allocation_df = pd.DataFrame(allocation_history).set_index('date')
    st.write("✓ Backtest complete."); return strategy_returns, last_weights, regime_df, allocation_df

# ==============================================================================
# MAIN APP LOGIC
# ==============================================================================
st.sidebar.header("AlphaSent Controls")
if st.sidebar.button("🚀 Run Full Backtest", type="primary"):
    
    backtest_data = load_data(DATA_URL)
    
    if not backtest_data.empty:
        prices_df = backtest_data.drop(columns=['compound'], errors='ignore')
        sentiment_index = backtest_data[['compound']].dropna()

        with st.spinner("Running backtest..."):
            strategy_returns, latest_weights, regime_df, allocation_df = run_backtest(prices_df, sentiment_index)

        if strategy_returns is not None:
            st.success("Analysis Complete!")
            
            # --- Prepare data for all tabs ---
            cumulative_returns = (1 + strategy_returns).cumprod()
            annual_return = cumulative_returns.iloc[-1]**(365/len(cumulative_returns)) - 1
            annual_volatility = strategy_returns.std() * (365**0.5)
            sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
            
            # --- Create Tabs for Results ---
            tab1, tab2 = st.tabs(["📈 Performance Dashboard", "🔬 Strategy Internals"])
            
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
                st.header("Strategy Internals & Diagnostics")
                st.subheader("Sentiment Regime Indicator")
                fig_regime, ax_regime = plt.subplots(figsize=(12, 4))
                sentiment_zscore = (sentiment_index['compound'] - sentiment_index['compound'].rolling(90).mean()) / sentiment_index['compound'].rolling(90).std()
                ax_regime.plot(sentiment_zscore.index, sentiment_zscore.values, label='Sentiment Z-Score', color='purple', alpha=0.7)
                ax_regime.axhline(SENTIMENT_ZSCORE_THRESHOLD, color='red', linestyle='--', label='Risk-On Threshold')
                ax_regime.fill_between(regime_df.index, 0, 1, where=regime_df['regime']==1, color='green', alpha=0.2, transform=ax_regime.get_xaxis_transform(), label='Risk-On Regime')
                ax_regime.set_title("Sentiment Z-Score and Resulting Market Regime"); ax_regime.legend(); st.pyplot(fig_regime)
                
                st.subheader("Final Recommended Portfolio Allocation")
                if latest_weights is not None and not latest_weights.empty:
                    fig_pie, ax_pie = plt.subplots(figsize=(6,6))
                    latest_weights[latest_weights > 0.01].plot.pie(ax=ax_pie, autopct='%1.1f%%', startangle=90)
                    ax_pie.set_ylabel(''); ax_pie.set_title("Recommended Allocation for Next Period"); st.pyplot(fig_pie)
                else:
                    st.info("The final allocation is 100% Cash.")
            
            with tab3:
                st.header("Gemini AI Analysis")
                with st.spinner("Generating AI summary..."):
                    results_dict = {"Annual Return": f"{annual_return:.2%}", "Sharpe Ratio": f"{sharpe_ratio:.2f}"}
                    latest_sentiment = sentiment_index['compound'].tail(7).mean()
                    top_holdings = latest_weights[latest_weights > 0.01].sort_values(ascending=False) if latest_weights is not None and not latest_weights.empty else pd.Series(["Cash"])
                    summary = generate_gemini_summary(results_dict, latest_sentiment, top_holdings)
                    st.markdown(summary)
        else:
            st.error("Could not complete the backtest.")
else:
    st.info("Click the button in the sidebar to run the backtest.")

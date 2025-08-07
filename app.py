import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def diagnose_backtest_issues(prices_df, sentiment_data, strategy_returns, allocation_history):
    """Comprehensive diagnostics for backtest issues"""
    
    st.markdown("## 🔍 Backtest Diagnostics")
    
    # 1. Data Quality Checks
    st.markdown("### 1. Data Quality Assessment")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Price Data:**")
        st.write(f"- Shape: {prices_df.shape}")
        st.write(f"- Date range: {prices_df.index.min()} to {prices_df.index.max()}")
        st.write(f"- Missing values: {prices_df.isnull().sum().sum()}")
        st.write(f"- Assets: {list(prices_df.columns[:5])}...")
    
    with col2:
        st.markdown("**Sentiment Data:**")
        st.write(f"- Shape: {sentiment_data.shape}")
        st.write(f"- Date range: {sentiment_data.index.min()} to {sentiment_data.index.max()}")
        st.write(f"- Sentiment range: {sentiment_data['compound'].min():.3f} to {sentiment_data['compound'].max():.3f}")
    
    # 2. Strategy Returns Analysis
    st.markdown("### 2. Strategy Returns Issues")
    
    if not strategy_returns.empty:
        # Check for problematic returns
        inf_returns = np.isinf(strategy_returns).sum()
        nan_returns = np.isnan(strategy_returns).sum()
        extreme_returns = (abs(strategy_returns) > 1).sum()  # >100% daily returns
        
        st.write(f"- Total return observations: {len(strategy_returns)}")
        st.write(f"- Infinite returns: {inf_returns}")
        st.write(f"- NaN returns: {nan_returns}")
        st.write(f"- Extreme returns (>100% daily): {extreme_returns}")
        
        if inf_returns > 0 or nan_returns > 0 or extreme_returns > 0:
            st.error("⚠️ **Critical Issue**: Invalid return values detected!")
            
            # Show problematic returns
            problematic = strategy_returns[
                np.isinf(strategy_returns) | 
                np.isnan(strategy_returns) | 
                (abs(strategy_returns) > 1)
            ]
            if not problematic.empty:
                st.write("**Problematic returns:**")
                st.dataframe(problematic.head(10))
        
        # Basic statistics (with safety checks)
        clean_returns = strategy_returns.replace([np.inf, -np.inf], np.nan).dropna()
        if not clean_returns.empty:
            st.write(f"- Mean daily return: {clean_returns.mean():.4f}")
            st.write(f"- Std daily return: {clean_returns.std():.4f}")
            st.write(f"- Min return: {clean_returns.min():.4f}")
            st.write(f"- Max return: {clean_returns.max():.4f}")
    
    # 3. Allocation History Analysis
    st.markdown("### 3. Portfolio Allocation Issues")
    
    if not allocation_history.empty:
        st.write(f"- Rebalancing periods: {len(allocation_history)}")
        
        # Check allocation validity
        allocation_df = pd.DataFrame(allocation_history)
        
        # Get asset columns (exclude metadata)
        meta_cols = ['date', 'regime', 'transaction_cost', 'method', 'expected_return', 
                    'volatility', 'sharpe_ratio', 'n_assets', 'reason']
        asset_cols = [col for col in allocation_df.columns if col not in meta_cols]
        
        if asset_cols:
            # Check if allocations sum to 1
            allocation_sums = allocation_df[asset_cols].sum(axis=1)
            st.write(f"- Average allocation sum: {allocation_sums.mean():.3f}")
            st.write(f"- Allocation sum range: {allocation_sums.min():.3f} to {allocation_sums.max():.3f}")
            
            # Check for extreme allocations
            max_allocations = allocation_df[asset_cols].max(axis=1)
            st.write(f"- Max single asset allocation: {max_allocations.max():.1%}")
            
            if allocation_sums.min() < 0.5 or allocation_sums.max() > 1.5:
                st.error("⚠️ **Critical Issue**: Portfolio allocations don't sum to ~1.0!")
        
        # Show regime distribution
        if 'regime' in allocation_df.columns:
            regime_counts = allocation_df['regime'].value_counts()
            st.write(f"- Regime distribution: {dict(regime_counts)}")
    
    return clean_returns if 'clean_returns' in locals() else pd.Series()


def fixed_calculate_performance_metrics(returns: pd.Series, prices_df: pd.DataFrame) -> dict:
    """Fixed performance metrics calculation with proper error handling"""
    
    if returns.empty:
        return {}
    
    # Clean returns first
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if len(clean_returns) < 2:
        return {"error": "Insufficient valid returns data"}
    
    try:
        # Basic metrics with safety checks
        total_return = (1 + clean_returns).prod() - 1
        
        # Annualized metrics
        trading_days = len(clean_returns)
        years = trading_days / 365.25
        
        if years > 0 and total_return > -0.99:  # Avoid log of negative numbers
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0
        
        # Volatility
        annualized_vol = clean_returns.std() * np.sqrt(365.25) if len(clean_returns) > 1 else 0
        
        # Sharpe ratio
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + clean_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio
        downside_returns = clean_returns[clean_returns < 0]
        if len(downside_returns) > 1:
            downside_vol = downside_returns.std() * np.sqrt(365.25)
            sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        else:
            sortino_ratio = sharpe_ratio
        
        # Win rate
        win_rate = (clean_returns > 0).mean()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'trading_days': trading_days,
            'valid_returns': len(clean_returns)
        }
        
    except Exception as e:
        return {"error": f"Calculation failed: {str(e)}"}


def fixed_backtest_engine_core(prices_df, sentiment_data, config):
    """Simplified, more robust backtest engine"""
    
    st.info("🔧 Running Fixed Backtest Engine...")
    
    # Data preparation
    daily_returns = prices_df.pct_change().dropna()
    
    # More conservative rebalancing (quarterly instead of monthly)
    rebalance_dates = prices_df.resample('QE').last().index  # Quarterly end
    rebalance_dates = [d for d in rebalance_dates if d in prices_df.index]
    
    if len(rebalance_dates) < 2:
        st.error("Insufficient rebalancing periods")
        return pd.Series(), pd.DataFrame(), {}
    
    st.write(f"Rebalancing {len(rebalance_dates)} times from {rebalance_dates[0]} to {rebalance_dates[-1]}")
    
    # Initialize tracking
    all_returns = []
    allocations = []
    
    # Simple equal-weight strategy first (for testing)
    n_assets = min(5, len(prices_df.columns))  # Top 5 assets only
    selected_assets = prices_df.columns[:n_assets]
    equal_weight = 1.0 / n_assets
    
    progress_bar = st.progress(0)
    
    for i in range(len(rebalance_dates) - 1):
        current_date = rebalance_dates[i]
        next_date = rebalance_dates[i + 1]
        
        progress_bar.progress((i + 1) / (len(rebalance_dates) - 1))
        
        # Simple equal weight allocation
        weights = pd.Series(equal_weight, index=selected_assets)
        
        # Get period returns
        period_returns = daily_returns.loc[current_date:next_date, selected_assets]
        
        if not period_returns.empty and len(period_returns) > 1:
            # Calculate portfolio returns
            portfolio_returns = (period_returns * weights).sum(axis=1)
            
            # Remove first day to avoid look-ahead bias
            portfolio_returns = portfolio_returns.iloc[1:]
            
            if not portfolio_returns.empty:
                all_returns.append(portfolio_returns)
        
        # Record allocation
        allocation_record = weights.to_dict()
        allocation_record.update({
            'date': current_date,
            'regime': 'neutral',  # Simplified
            'method': 'equal_weight'
        })
        allocations.append(allocation_record)
    
    progress_bar.empty()
    
    if not all_returns:
        st.error("No returns generated")
        return pd.Series(), pd.DataFrame(), {}
    
    # Combine results
    strategy_returns = pd.concat(all_returns)
    allocation_df = pd.DataFrame(allocations)
    
    # Calculate metrics
    metrics = fixed_calculate_performance_metrics(strategy_returns, prices_df)
    
    st.success(f"✅ Generated {len(strategy_returns)} return observations")
    
    return strategy_returns, allocation_df, metrics


# Add this diagnostic function call to your main app
def run_diagnostic_analysis(data):
    """Run diagnostic version of backtest"""
    
    st.markdown("## 🚨 Running Diagnostic Mode")
    
    # Prepare data
    prices_df = data.drop(columns=['compound'], errors='ignore')
    sentiment_data = data[['compound']].dropna()
    
    # Run simplified backtest
    config = type('Config', (), {
        'TRANSACTION_COST': 0.0025,
        'MAX_POSITION_SIZE': 0.30,
        'LOOKBACK_PERIOD': 90,
        'SENTIMENT_THRESHOLDS': {'risk_on': 1.0, 'risk_off': -1.0}
    })()
    
    strategy_returns, allocation_history, performance_metrics = fixed_backtest_engine_core(
        prices_df, sentiment_data, config
    )
    
    # Run diagnostics
    clean_returns = diagnose_backtest_issues(
        prices_df, sentiment_data, strategy_returns, allocation_history
    )
    
    # Show corrected metrics
    if performance_metrics and 'error' not in performance_metrics:
        st.markdown("### ✅ Corrected Performance Metrics")
        
        metrics_display = pd.DataFrame([
            ["Total Return", f"{performance_metrics.get('total_return', 0):.2%}"],
            ["Annualized Return", f"{performance_metrics.get('annualized_return', 0):.2%}"],
            ["Annualized Volatility", f"{performance_metrics.get('annualized_volatility', 0):.2%}"],
            ["Sharpe Ratio", f"{performance_metrics.get('sharpe_ratio', 0):.3f}"],
            ["Sortino Ratio", f"{performance_metrics.get('sortino_ratio', 0):.3f}"],
            ["Maximum Drawdown", f"{performance_metrics.get('max_drawdown', 0):.2%}"],
            ["Win Rate", f"{performance_metrics.get('win_rate', 0):.1%}"],
            ["Valid Trading Days", f"{performance_metrics.get('valid_returns', 0)}"],
        ], columns=["Metric", "Value"])
        
        st.dataframe(metrics_display, hide_index=True)
    elif 'error' in performance_metrics:
        st.error(f"Metrics calculation failed: {performance_metrics['error']}")
    
    return strategy_returns, allocation_history, performance_metrics

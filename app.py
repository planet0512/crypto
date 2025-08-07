import streamlit as st
import pandas as pd
import numpy as np
import requests

st.title("🔧 AlphaSent Diagnostics")

@st.cache_data(ttl=1800)
def load_test_data():
    """Load data for testing"""
    url = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
data = load_test_data()

if not data.empty:
    st.success(f"✅ Data loaded: {data.shape}")
    
    # Basic info
    st.markdown("## Data Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Shape:**", data.shape)
        st.write("**Columns:**", list(data.columns)[:10])
        st.write("**Date Range:**", f"{data.index.min()} to {data.index.max()}")
    
    with col2:
        st.write("**Data Types:**")
        st.write(data.dtypes.head(10))
    
    # Show first few rows
    st.markdown("## Sample Data")
    st.dataframe(data.head())
    
    # Test individual assets
    st.markdown("## Asset Analysis")
    
    # Get non-compound columns
    asset_columns = [col for col in data.columns if col != 'compound']
    
    if asset_columns:
        selected_asset = st.selectbox("Select asset to analyze:", asset_columns)
        
        if selected_asset:
            asset_data = data[selected_asset].dropna()
            returns = asset_data.pct_change().dropna()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Price Observations", len(asset_data))
                st.metric("Return Observations", len(returns))
            
            with col2:
                st.metric("Mean Daily Return", f"{returns.mean():.4f}")
                st.metric("Daily Volatility", f"{returns.std():.4f}")
            
            with col3:
                st.metric("Min Return", f"{returns.min():.4f}")
                st.metric("Max Return", f"{returns.max():.4f}")
            
            # Data quality checks
            st.markdown("### Data Quality")
            
            problems = []
            if np.isinf(returns).any():
                problems.append("❌ Contains infinite values")
            if np.isnan(returns).any():
                problems.append("❌ Contains NaN values")
            if (abs(returns) > 1).any():
                problems.append("❌ Contains extreme returns (>100%)")
            
            if problems:
                for problem in problems:
                    st.write(problem)
            else:
                st.success("✅ Data quality looks good")
            
            # Simple plot
            import plotly.graph_objects as go
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=asset_data.index,
                y=asset_data.values,
                name=f'{selected_asset} Price'
            ))
            fig.update_layout(title=f"{selected_asset} Price History", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Sentiment analysis
    if 'compound' in data.columns:
        st.markdown("## Sentiment Analysis")
        
        sentiment = data['compound'].dropna()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Sentiment Observations", len(sentiment))
            st.metric("Mean Sentiment", f"{sentiment.mean():.3f}")
        
        with col2:
            st.metric("Sentiment Std", f"{sentiment.std():.3f}")
            st.metric("Range", f"{sentiment.min():.3f} to {sentiment.max():.3f}")
        
        # Sentiment plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sentiment.index,
            y=sentiment.values,
            name='Sentiment Score'
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_layout(title="Sentiment Over Time", height=400)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Failed to load data")

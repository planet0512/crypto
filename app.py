import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from openai import OpenAI
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.objective_functions import L2_reg
from pypfopt.exceptions import OptimizationError
import cvxpy as cp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import warnings
import logging
from typing import Dict, Tuple, Optional
import yfinance as yf

# Configure logging and warnings
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings('ignore')

# ==============================================================================
# PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="Project AlphaSent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sentiment-gauge {
        text-align: center;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    .news-item {
        border-left: 4px solid #3498db;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
        border-radius: 0 10px 10px 0;
    }
    .risk-on { border-left-color: #2ecc71 !important; }
    .risk-off { border-left-color: #e74c3c !important; }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Project AlphaSent")
st.markdown("### *A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation*")

# ==============================================================================
# CONFIGURATION & CONSTANTS
# ==============================================================================
class Config:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    CRYPTOCOMPARE_API_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
    DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
    SENTIMENT_THRESHOLDS = {'risk_on': 1.0, 'risk_off': -1.0}
    TRANSACTION_COST = 0.0025  # 25 basis points
    MAX_POSITION_SIZE = 0.30
    LOOKBACK_PERIOD = 90
    
@st.cache_resource
def setup_nltk():
    """Download NLTK data (runs once)."""
    import nltk
    try:
        nltk.download('vader_lexicon', quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        return False

# ==============================================================================
# ENHANCED DATA HANDLING & CACHING
# ==============================================================================

@st.cache_data(ttl=300)  # Cache for 5 minutes
def create_requests_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504, 429]
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return session

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_data(url: str) -> pd.DataFrame:
    """Load and validate historical backtest data."""
    try:
        with st.spinner("Loading historical data..."):
            df = pd.read_csv(url, index_col=0, parse_dates=True)
            df.index.name = 'time'
            
            # Data validation
            if df.empty:
                raise ValueError("Loaded data is empty")
            
            # Check for required columns
            required_cols = ['compound']  # At minimum, we need sentiment data
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                st.warning(f"Missing columns: {missing_cols}")
            
            # Remove columns with all NaN values
            df = df.dropna(axis=1, how='all')
            
            st.success(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} assets")
            return df
            
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900)  # Cache for 15 minutes
def fetch_live_news(_session: requests.Session, api_key: str) -> pd.DataFrame:
    """Fetch and analyze live cryptocurrency news."""
    if not api_key:
        return pd.DataFrame()
    
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    
    try:
        response = _session.get(url, timeout=30)
        response.raise_for_status()
        data = response.json().get('Data', [])
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data).head(20)
        
        # Sentiment analysis
        analyzer = SentimentIntensityAnalyzer()
        df['compound'] = df['title'].fillna('').apply(
            lambda txt: analyzer.polarity_scores(txt)['compound']
        )
        
        # Add sentiment labels
        df['sentiment_label'] = df['compound'].apply(get_sentiment_label)
        df['impact_score'] = df['compound'].abs()
        
        return df[['title', 'source', 'compound', 'sentiment_label', 'impact_score', 'url']]
        
    except requests.exceptions.RequestException as e:
        st.warning(f"Failed to fetch news: {e}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error processing news: {e}")
        return pd.DataFrame()

def get_sentiment_label(score: float) -> str:
    """Convert sentiment score to human-readable label."""
    if score > 0.5:
        return "Very Positive"
    elif score > 0.1:
        return "Positive"
    elif score > -0.1:
        return "Neutral"
    elif score > -0.5:
        return "Negative"
    else:
        return "Very Negative"

# ==============================================================================
# ENHANCED PORTFOLIO OPTIMIZATION
# ==============================================================================

class PortfolioOptimizer:
    """Enhanced portfolio optimization with multiple models and risk controls."""
    
    def __init__(self, transaction_cost: float = 0.0025, max_weight: float = 0.30):
        self.transaction_cost = transaction_cost
        self.max_weight = max_weight
    
    def clean_price_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data."""
        # Remove columns with insufficient data
        min_observations = max(30, len(prices) // 10)
        valid_cols = prices.dropna().count()[lambda x: x >= min_observations].index
        
        if len(valid_cols) < 2:
            raise ValueError("Insufficient valid price data")
        
        cleaned = prices[valid_cols].ffill().dropna(how='all')
        
        # Remove zero-variance assets
        returns = cleaned.pct_change().dropna()
        valid_assets = returns.std()[lambda x: x > 1e-6].index
        
        return cleaned[valid_assets]
    
    def get_optimized_weights(
        self, 
        prices: pd.DataFrame, 
        model: str = "max_sharpe",
        sentiment_scores: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict]:
        """
        Get optimized portfolio weights with enhanced error handling.
        
        Returns:
            Tuple of (weights, metadata)
        """
        try:
            # Clean data
            clean_prices = self.clean_price_data(prices)
            
            if len(clean_prices.columns) < 2:
                return self._fallback_weights(prices.columns), {"method": "equal_weight", "reason": "insufficient_assets"}
            
            # Calculate returns and moments
            returns = clean_prices.pct_change().dropna()
            
            # Enhanced moment estimation
            mu = expected_returns.ema_historical_return(clean_prices, frequency=365)
            
            # Incorporate sentiment adjustment
            if sentiment_scores is not None:
                mu = self._adjust_returns_for_sentiment(mu, sentiment_scores)
            
            # Robust covariance estimation
            S = risk_models.CovarianceShrinkage(clean_prices).ledoit_wolf()
            
            # Initialize optimizer
            ef = EfficientFrontier(mu, S, weight_bounds=(0, self.max_weight))
            ef.add_objective(L2_reg, gamma=0.01)  # L2 regularization
            
            # Optimize based on model type
            if model == "max_sharpe":
                weights = ef.max_sharpe()
            elif model == "min_variance":
                weights = ef.min_volatility()
            elif model == "max_quadratic_utility":
                weights = ef.max_quadratic_utility(risk_aversion=1.0)
            else:
                raise ValueError(f"Unknown optimization model: {model}")
            
            # Clean and normalize weights
            cleaned_weights = ef.clean_weights(cutoff=0.01)
            weights_series = pd.Series(cleaned_weights)
            
            # Calculate performance metrics
            portfolio_return = weights_series.dot(mu)
            portfolio_vol = np.sqrt(weights_series.T @ S @ weights_series)
            sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            
            metadata = {
                "method": model,
                "expected_return": portfolio_return,
                "volatility": portfolio_vol,
                "sharpe_ratio": sharpe,
                "n_assets": len(weights_series[weights_series > 0])
            }
            
            return weights_series.reindex(prices.columns, fill_value=0), metadata
            
        except (OptimizationError, ValueError, cp.error.SolverError) as e:
            st.warning(f"Optimization failed ({e.__class__.__name__}): {str(e)[:100]}...")
            return self._fallback_weights(prices.columns), {"method": "equal_weight", "reason": str(e)}
    
    def _adjust_returns_for_sentiment(self, mu: pd.Series, sentiment_scores: pd.Series) -> pd.Series:
        """Adjust expected returns based on sentiment scores."""
        # Align indices
        common_assets = mu.index.intersection(sentiment_scores.index)
        if len(common_assets) == 0:
            return mu
        
        adjusted_mu = mu.copy()
        for asset in common_assets:
            # Sentiment adjustment factor (capped at ±20% of original return)
            sentiment_factor = np.clip(sentiment_scores[asset] * 0.1, -0.2, 0.2)
            adjusted_mu[asset] *= (1 + sentiment_factor)
        
        return adjusted_mu
    
    def _fallback_weights(self, asset_names) -> pd.Series:
        """Fallback to equal weights if optimization fails."""
        return pd.Series(1.0 / len(asset_names), index=asset_names)

# ==============================================================================
# ENHANCED BACKTESTING ENGINE
# ==============================================================================

class BacktestEngine:
    """Enhanced backtesting engine with regime switching and risk controls."""
    
    def __init__(self, optimizer: PortfolioOptimizer, config: Config):
        self.optimizer = optimizer
        self.config = config
    
    def run_backtest(
        self, 
        prices_df: pd.DataFrame, 
        sentiment_data: pd.DataFrame
    ) -> Tuple[pd.Series, pd.DataFrame, Dict]:
        """
        Run comprehensive backtest with regime switching.
        
        Returns:
            Tuple of (strategy_returns, allocation_history, performance_metrics)
        """
        st.info("🚀 Initiating Enhanced Backtest Engine...")
        
        # Prepare data
        daily_returns = prices_df.pct_change().fillna(0)
        
        # Generate rebalancing dates (monthly)
        rebalance_dates = prices_df.resample('ME').last().index
        rebalance_dates = [d for d in rebalance_dates if d in prices_df.index]
        
        if len(rebalance_dates) < 3:
            st.error("Insufficient data for backtesting")
            return pd.Series(), pd.DataFrame(), {}
        
        # Initialize tracking variables
        portfolio_returns = []
        allocation_history = []
        regime_history = []
        last_weights = pd.Series()
        transaction_costs = []
        
        # Calculate rolling sentiment z-scores
        sentiment_zscore = self._calculate_sentiment_zscore(sentiment_data)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Main backtesting loop
        for i, (current_date, next_date) in enumerate(zip(rebalance_dates[:-1], rebalance_dates[1:])):
            progress = (i + 1) / (len(rebalance_dates) - 1)
            progress_bar.progress(progress)
            status_text.text(f"Processing: {current_date.strftime('%Y-%m')} ({i+1}/{len(rebalance_dates)-1})")
            
            # Determine current regime
            regime = self._get_regime(sentiment_zscore, current_date)
            
            # Get historical data for optimization
            hist_start = current_date - timedelta(days=self.config.LOOKBACK_PERIOD)
            hist_prices = prices_df.loc[hist_start:current_date].dropna(how='all')
            
            if len(hist_prices) < 30:
                continue
            
            # Optimize portfolio based on regime
            target_weights, opt_metadata = self._optimize_for_regime(hist_prices, regime, sentiment_zscore)
            
            # Calculate transaction costs
            if not last_weights.empty:
                turnover = (target_weights - last_weights.reindex(target_weights.index, fill_value=0)).abs().sum()
                txn_cost = turnover * self.config.TRANSACTION_COST / 2
                transaction_costs.append(txn_cost)
            else:
                transaction_costs.append(0)
            
            # Calculate period returns
            period_returns = self._calculate_period_returns(
                daily_returns, target_weights, current_date, next_date, transaction_costs[-1]
            )
            
            if not period_returns.empty:
                portfolio_returns.append(period_returns)
            
            # Record allocation and regime
            allocation_record = target_weights.to_dict()
            allocation_record.update({
                'date': current_date,
                'regime': regime,
                'transaction_cost': transaction_costs[-1],
                **opt_metadata
            })
            allocation_history.append(allocation_record)
            
            regime_history.append({
                'date': current_date,
                'regime': regime,
                'sentiment_zscore': sentiment_zscore.loc[sentiment_zscore.index <= current_date].iloc[-1] if not sentiment_zscore.empty else 0
            })
            
            last_weights = target_weights
        
        progress_bar.empty()
        status_text.empty()
        
        if not portfolio_returns:
            st.error("No valid returns generated")
            return pd.Series(), pd.DataFrame(), {}
        
        # Combine results
        strategy_returns = pd.concat(portfolio_returns)
        allocation_df = pd.DataFrame(allocation_history)
        regime_df = pd.DataFrame(regime_history)
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            strategy_returns, prices_df, transaction_costs
        )
        
        st.success("✅ Backtest completed successfully!")
        return strategy_returns, allocation_df, performance_metrics
    
    def _calculate_sentiment_zscore(self, sentiment_data: pd.DataFrame, window: int = 90) -> pd.Series:
        """Calculate rolling z-score of sentiment."""
        if 'compound' not in sentiment_data.columns:
            return pd.Series()
        
        sentiment = sentiment_data['compound'].fillna(0)
        rolling_mean = sentiment.rolling(window=window, min_periods=30).mean()
        rolling_std = sentiment.rolling(window=window, min_periods=30).std()
        
        return (sentiment - rolling_mean) / rolling_std.replace(0, 1)
    
    def _get_regime(self, sentiment_zscore: pd.Series, date) -> str:
        """Determine market regime based on sentiment."""
        if sentiment_zscore.empty:
            return "neutral"
        
        # Get most recent sentiment score up to the date
        recent_sentiment = sentiment_zscore.loc[sentiment_zscore.index <= date]
        if recent_sentiment.empty:
            return "neutral"
        
        current_zscore = recent_sentiment.iloc[-1]
        
        if current_zscore > self.config.SENTIMENT_THRESHOLDS['risk_on']:
            return "risk_on"
        elif current_zscore < self.config.SENTIMENT_THRESHOLDS['risk_off']:
            return "risk_off"
        else:
            return "neutral"
    
    def _optimize_for_regime(self, prices: pd.DataFrame, regime: str, sentiment_zscore: pd.Series) -> Tuple[pd.Series, Dict]:
        """Optimize portfolio based on current regime."""
        # Get recent sentiment scores for assets (if available)
        sentiment_scores = None
        
        # Regime-specific optimization
        if regime == "risk_on":
            # Aggressive optimization for growth
            weights, metadata = self.optimizer.get_optimized_weights(
                prices, model="max_sharpe", sentiment_scores=sentiment_scores
            )
            metadata['regime'] = 'risk_on'
        
        elif regime == "risk_off":
            # Conservative optimization for capital preservation
            weights, metadata = self.optimizer.get_optimized_weights(
                prices, model="min_variance", sentiment_scores=sentiment_scores
            )
            metadata['regime'] = 'risk_off'
        
        else:  # neutral
            # Balanced approach
            weights, metadata = self.optimizer.get_optimized_weights(
                prices, model="max_quadratic_utility", sentiment_scores=sentiment_scores
            )
            metadata['regime'] = 'neutral'
        
        return weights, metadata
    
    def _calculate_period_returns(
        self, 
        daily_returns: pd.DataFrame, 
        weights: pd.Series, 
        start_date, 
        end_date, 
        txn_cost: float
    ) -> pd.Series:
        """Calculate portfolio returns for a specific period."""
        period_returns = daily_returns.loc[start_date:end_date]
        
        if period_returns.empty:
            return pd.Series()
        
        # Align weights with returns
        common_assets = weights.index.intersection(period_returns.columns)
        aligned_weights = weights.reindex(common_assets, fill_value=0)
        aligned_returns = period_returns[common_assets]
        
        # Calculate portfolio returns
        portfolio_returns = (aligned_returns * aligned_weights).sum(axis=1)
        
        # Apply transaction cost to first return
        if not portfolio_returns.empty:
            portfolio_returns.iloc[0] -= txn_cost
        
        return portfolio_returns
    
    def _calculate_performance_metrics(
        self, 
        returns: pd.Series, 
        prices_df: pd.DataFrame, 
        transaction_costs: list
    ) -> Dict:
        """Calculate comprehensive performance metrics."""
        if returns.empty:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (365 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(365)
        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
        
        # Drawdown analysis
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(365) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_vol if downside_vol > 0 else 0
        
        # Benchmark comparison (if BTC available)
        benchmark_metrics = {}
        if 'BTC' in prices_df.columns:
            btc_returns = prices_df['BTC'].pct_change().dropna()
            if len(btc_returns) > 0:
                btc_total_return = (1 + btc_returns).prod() - 1
                btc_annual_return = (1 + btc_total_return) ** (365 / len(btc_returns)) - 1
                btc_vol = btc_returns.std() * np.sqrt(365)
                btc_sharpe = btc_annual_return / btc_vol if btc_vol > 0 else 0
                
                btc_cumulative = (1 + btc_returns).cumprod()
                btc_rolling_max = btc_cumulative.expanding().max()
                btc_drawdown = (btc_cumulative - btc_rolling_max) / btc_rolling_max
                btc_max_drawdown = btc_drawdown.min()
                
                benchmark_metrics = {
                    'btc_annual_return': btc_annual_return,
                    'btc_volatility': btc_vol,
                    'btc_sharpe_ratio': btc_sharpe,
                    'btc_max_drawdown': btc_max_drawdown,
                    'excess_return': annualized_return - btc_annual_return,
                    'excess_sharpe': sharpe_ratio - btc_sharpe
                }
        
        # Transaction cost impact
        total_txn_costs = sum(transaction_costs)
        avg_annual_txn_cost = total_txn_costs * (365 / len(returns)) if len(returns) > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'total_transaction_costs': total_txn_costs,
            'avg_annual_transaction_cost': avg_annual_txn_cost,
            **benchmark_metrics
        }

# ==============================================================================
# ENHANCED VISUALIZATION FUNCTIONS
# ==============================================================================

def create_performance_dashboard(strategy_returns: pd.Series, prices_df: pd.DataFrame, metrics: Dict):
    """Create comprehensive performance dashboard using Plotly."""
    
    # Cumulative returns chart
    fig_perf = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Cumulative Performance', 'Rolling Sharpe Ratio', 'Drawdown Analysis', 'Monthly Returns Heatmap'),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Calculate cumulative returns
    strategy_cum = (1 + strategy_returns).cumprod()
    
    # Plot strategy performance
    fig_perf.add_trace(
        go.Scatter(x=strategy_cum.index, y=strategy_cum.values, name='AlphaSent Strategy', line=dict(color='#3B82F6', width=3)),
        row=1, col=1
    )
    
    # Add Bitcoin benchmark if available
    if 'BTC' in prices_df.columns:
        btc_returns = prices_df['BTC'].pct_change().dropna()
        btc_cum = (1 + btc_returns).cumprod()
        # Align with strategy returns
        btc_aligned = btc_cum.reindex(strategy_cum.index, method='ffill')
        fig_perf.add_trace(
            go.Scatter(x=btc_aligned.index, y=btc_aligned.values, name='Bitcoin', line=dict(color='#F7931A', width=2, dash='dash')),
            row=1, col=1
        )
    
    # Rolling Sharpe ratio
    rolling_sharpe = strategy_returns.rolling(window=90).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0
    )
    fig_perf.add_trace(
        go.Scatter(x=rolling_sharpe.index, y=rolling_sharpe.values, name='90-Day Rolling Sharpe', line=dict(color='#10B981')),
        row=1, col=2
    )
    
    # Drawdown analysis
    rolling_max = strategy_cum.expanding().max()
    drawdown = (strategy_cum - rolling_max) / rolling_max * 100
    fig_perf.add_trace(
        go.Scatter(x=drawdown.index, y=drawdown.values, fill='tonexty', name='Drawdown %', line=dict(color='#EF4444')),
        row=2, col=1
    )
    
    # Monthly returns heatmap
    monthly_returns = strategy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1) * 100
    monthly_pivot = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack(fill_value=0)
    
    if not monthly_pivot.empty:
        fig_perf.add_trace(
            go.Heatmap(
                z=monthly_pivot.values,
                x=[f"Month {i}" for i in range(1, 13)],
                y=monthly_pivot.index,
                colorscale='RdYlGn',
                name='Monthly Returns %'
            ),
            row=2, col=2
        )
    
    fig_perf.update_layout(height=800, showlegend=True, title_text="AlphaSent Performance Dashboard")
    fig_perf.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig_perf.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    fig_perf.update_yaxes(title_text="Drawdown %", row=2, col=1)
    
    return fig_perf

def create_sentiment_analysis_chart(sentiment_data: pd.DataFrame, strategy_returns: pd.Series):
    """Create sentiment analysis visualization."""
    
    if 'compound' not in sentiment_data.columns:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Market Sentiment Index', 'Sentiment vs Strategy Performance'),
        vertical_spacing=0.1,
        specs=[[{"secondary_y": False}],
               [{"secondary_y": True}]]
    )
    
    # Sentiment z-score
    sentiment = sentiment_data['compound'].fillna(0)
    sentiment_zscore = (sentiment - sentiment.rolling(90).mean()) / sentiment.rolling(90).std()
    
    fig.add_trace(
        go.Scatter(x=sentiment_zscore.index, y=sentiment_zscore.values, 
                  name='Sentiment Z-Score', line=dict(color='purple')),
        row=1, col=1
    )
    
    # Add regime thresholds
    fig.add_hline(y=1.0, line_dash="dash", line_color="green", 
                  annotation_text="Risk-On Threshold", row=1, col=1)
    fig.add_hline(y=-1.0, line_dash="dash", line_color="red", 
                  annotation_text="Risk-Off Threshold", row=1, col=1)
    
    # Strategy performance overlay
    if not strategy_returns.empty:
        strategy_cum = (1 + strategy_returns).cumprod()
        fig.add_trace(
            go.Scatter(x=strategy_cum.index, y=strategy_cum.values, 
                      name='Strategy Performance', line=dict(color='#3B82F6')),
            row=2, col=1, secondary_y=True
        )
    
    fig.update_layout(height=600, title_text="Sentiment Analysis Dashboard")
    fig.update_yaxes(title_text="Z-Score", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=1, secondary_y=True)
    
    return fig

# ==============================================================================
# ENHANCED AI SUMMARY GENERATOR
# ==============================================================================

def generate_enhanced_ai_summary(
    performance_metrics: Dict, 
    latest_sentiment: float, 
    latest_weights: pd.Series,
    regime_history: pd.DataFrame
) -> str:
    """Generate enhanced AI summary using OpenRouter."""
    
    if not Config.OPENROUTER_API_KEY:
        return "⚠️ Please add your OpenRouter API Key to generate AI analysis."
    
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=Config.OPENROUTER_API_KEY
        )
        
        # Prepare context data
        top_holdings = latest_weights.nlargest(5)
        regime_stats = regime_history['regime'].value_counts() if not regime_history.empty else pd.Series()
        
        prompt = f"""
        As a quantitative finance analyst, provide a comprehensive analysis of the AlphaSent cryptocurrency portfolio strategy:

        ## PERFORMANCE SUMMARY
        - Annual Return: {performance_metrics.get('annualized_return', 0):.2%}
        - Sharpe Ratio: {performance_metrics.get('sharpe_ratio', 0):.2f}
        - Maximum Drawdown: {performance_metrics.get('max_drawdown', 0):.2%}
        - Volatility: {performance_metrics.get('annualized_volatility', 0):.2%}
        
        ## BENCHMARK COMPARISON
        - Excess Return vs BTC: {performance_metrics.get('excess_return', 0):.2%}
        - Excess Sharpe vs BTC: {performance_metrics.get('excess_sharpe', 0):.2f}
        
        ## CURRENT MARKET CONDITIONS
        - Latest Sentiment Z-Score: {latest_sentiment:.2f}
        - Top 5 Holdings: {top_holdings.to_dict()}
        - Regime Distribution: {regime_stats.to_dict()}

        ## ANALYSIS REQUIREMENTS
        1. **Executive Summary**: Brief performance assessment
        2. **Strategy Effectiveness**: Sentiment integration impact
        3. **Risk Assessment**: Drawdown and volatility analysis
        4. **Current Positioning**: Portfolio allocation rationale
        5. **Forward Outlook**: Based on current sentiment regime

        Provide actionable insights in a professional, data-driven tone. Limit response to 500 words.
        """
        
        response = client.chat.completions.create(
            model="google/gemini-2.0-flash-exp",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ Failed to generate AI analysis: {str(e)}"

# ==============================================================================
# MAIN APPLICATION LOGIC
# ==============================================================================

def main():
    """Main application logic with enhanced UI and functionality."""
    
    # Initialize components
    if not setup_nltk():
        st.stop()
    
    config = Config()
    session = create_requests_session()
    optimizer = PortfolioOptimizer()
    backtest_engine = BacktestEngine(optimizer, config)
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🎛️ AlphaSent Control Center")
        
        # API Status indicators
        st.markdown("### API Status")
        col1, col2 = st.columns(2)
        with col1:
            crypto_status = "🟢" if config.CRYPTOCOMPARE_API_KEY else "🔴"
            st.markdown(f"{crypto_status} CryptoCompare")
        with col2:
            ai_status = "🟢" if config.OPENROUTER_API_KEY else "🔴"
            st.markdown(f"{ai_status} OpenRouter AI")
        
        st.divider()
        
        # Strategy parameters
        st.markdown("### Strategy Parameters")
        sentiment_threshold = st.slider("Sentiment Threshold", -2.0, 2.0, config.SENTIMENT_THRESHOLDS['risk_on'], 0.1)
        max_position = st.slider("Max Position Size", 0.1, 0.5, config.MAX_POSITION_SIZE, 0.05)
        lookback_days = st.slider("Lookback Period (days)", 30, 180, config.LOOKBACK_PERIOD, 10)
        
        # Update config
        config.SENTIMENT_THRESHOLDS['risk_on'] = sentiment_threshold
        config.SENTIMENT_THRESHOLDS['risk_off'] = -sentiment_threshold
        config.MAX_POSITION_SIZE = max_position
        config.LOOKBACK_PERIOD = lookback_days
        
        st.divider()
        
        # Live news feed
        st.markdown("### 📰 Live News Feed")
        
        with st.spinner("Fetching latest news..."):
            live_news = fetch_live_news(session, config.CRYPTOCOMPARE_API_KEY)
        
        if not live_news.empty:
            for _, news in live_news.head(5).iterrows():
                sentiment_class = "risk-on" if news['compound'] > 0 else "risk-off"
                st.markdown(f"""
                <div class="news-item {sentiment_class}">
                    <strong>{news['source']}</strong><br>
                    <a href="{news['url']}" target="_blank">{news['title'][:60]}...</a><br>
                    <small>Sentiment: {news['sentiment_label']} ({news['compound']:.2f})</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("News feed temporarily unavailable")
    
    # Main content area
    st.markdown("---")
    
    # Action button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Run Enhanced Backtest Analysis", type="primary", use_container_width=True):
            run_full_analysis(config, backtest_engine)

def run_full_analysis(config: Config, backtest_engine: BacktestEngine):
    """Run the complete backtest analysis with enhanced reporting."""
    
    # Load data
    data = load_data(config.DATA_URL)
    if data.empty:
        st.error("Failed to load historical data. Please check the data source.")
        return
    
    # Prepare data
    prices_df = data.drop(columns=['compound'], errors='ignore')
    sentiment_data = data[['compound']].dropna()
    
    if sentiment_data.empty:
        st.error("No sentiment data available")
        return
    
    # Run backtest
    with st.spinner("Running comprehensive backtest analysis..."):
        strategy_returns, allocation_history, performance_metrics = backtest_engine.run_backtest(
            prices_df, sentiment_data
        )
    
    if strategy_returns.empty:
        st.error("Backtest failed to generate results")
        return
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Performance Dashboard", 
        "🎯 Asset Allocation", 
        "📈 Sentiment Analysis", 
        "🤖 AI Insights"
    ])
    
    with tab1:
        st.markdown("## Performance Dashboard")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            annual_ret = performance_metrics.get('annualized_return', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Annual Return</h3>
                <h2>{annual_ret:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sharpe = performance_metrics.get('sharpe_ratio', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Sharpe Ratio</h3>
                <h2>{sharpe:.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            max_dd = performance_metrics.get('max_drawdown', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Max Drawdown</h3>
                <h2>{max_dd:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            volatility = performance_metrics.get('annualized_volatility', 0)
            st.markdown(f"""
            <div class="metric-card">
                <h3>Volatility</h3>
                <h2>{volatility:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Performance charts
        fig_perf = create_performance_dashboard(strategy_returns, prices_df, performance_metrics)
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance metrics table
        if performance_metrics:
            st.markdown("### Detailed Performance Metrics")
            
            metrics_df = pd.DataFrame([
                ["Total Return", f"{performance_metrics.get('total_return', 0):.2%}"],
                ["Annualized Return", f"{performance_metrics.get('annualized_return', 0):.2%}"],
                ["Annualized Volatility", f"{performance_metrics.get('annualized_volatility', 0):.2%}"],
                ["Sharpe Ratio", f"{performance_metrics.get('sharpe_ratio', 0):.3f}"],
                ["Sortino Ratio", f"{performance_metrics.get('sortino_ratio', 0):.3f}"],
                ["Maximum Drawdown", f"{performance_metrics.get('max_drawdown', 0):.2%}"],
                ["Total Transaction Costs", f"{performance_metrics.get('total_transaction_costs', 0):.2%}"],
            ], columns=["Metric", "Value"])
            
            # Add benchmark comparison if available
            if 'btc_annual_return' in performance_metrics:
                benchmark_df = pd.DataFrame([
                    ["BTC Annual Return", f"{performance_metrics.get('btc_annual_return', 0):.2%}"],
                    ["BTC Sharpe Ratio", f"{performance_metrics.get('btc_sharpe_ratio', 0):.3f}"],
                    ["BTC Max Drawdown", f"{performance_metrics.get('btc_max_drawdown', 0):.2%}"],
                    ["Excess Return vs BTC", f"{performance_metrics.get('excess_return', 0):.2%}"],
                    ["Excess Sharpe vs BTC", f"{performance_metrics.get('excess_sharpe', 0):.3f}"],
                ], columns=["Benchmark Metric", "Value"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.dataframe(metrics_df, hide_index=True)
                with col2:
                    st.dataframe(benchmark_df, hide_index=True)
            else:
                st.dataframe(metrics_df, hide_index=True)
    
    with tab2:
        st.markdown("## Asset Allocation Analysis")
        
        if not allocation_history.empty:
            # Current allocation - FIXED VERSION
            latest_allocation = allocation_history.iloc[-1]
            
            # More robust extraction of weights
            non_meta_cols = ['date', 'regime', 'transaction_cost', 'method', 
                           'expected_return', 'volatility', 'sharpe_ratio', 'n_assets']
            
            # Convert latest_allocation to dict if it's not already
            if hasattr(latest_allocation, 'to_dict'):
                latest_dict = latest_allocation.to_dict()
            else:
                latest_dict = dict(latest_allocation)
            
            # Filter out metadata and ensure numeric values
            latest_weights_dict = {}
            for k, v in latest_dict.items():
                if k not in non_meta_cols:
                    try:
                        # Convert to float, skip if it fails
                        numeric_val = float(v) if v is not None else 0.0
                        if numeric_val > 0.01:  # Only include significant weights
                            latest_weights_dict[k] = numeric_val
                    except (ValueError, TypeError):
                        # Skip non-numeric values
                        continue
            
            latest_weights = pd.Series(latest_weights_dict).sort_values(ascending=False)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Current Portfolio Allocation")
                if not latest_weights.empty:
                    fig_pie = px.pie(
                        values=latest_weights.values,
                        names=latest_weights.index,
                        title="Current Asset Allocation"
                    )
                    fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("Current allocation is 100% cash/stable assets")
            
            with col2:
                st.markdown("### Allocation Details")
                if not latest_weights.empty:
                    allocation_df = pd.DataFrame({
                        'Asset': latest_weights.index,
                        'Weight': [f"{w:.1%}" for w in latest_weights.values],
                        'Value ($)': [f"${w * 100000:.0f}" for w in latest_weights.values]  # Assuming $100k portfolio
                    })
                    st.dataframe(allocation_df, hide_index=True)
                else:
                    st.info("No significant allocations to display")
            
            # Allocation over time - FIXED VERSION
            st.markdown("### Portfolio Evolution Over Time")
            
            # Create allocation history chart
            allocation_df_full = pd.DataFrame(allocation_history)
            
            try:
                if 'date' in allocation_df_full.columns:
                    allocation_df_full['date'] = pd.to_datetime(allocation_df_full['date'])
                    allocation_df_full = allocation_df_full.set_index('date')
                    
                    # Get asset columns more carefully
                    asset_cols = []
                    for col in allocation_df_full.columns:
                        if col not in non_meta_cols:
                            # Check if column contains numeric data
                            try:
                                pd.to_numeric(allocation_df_full[col], errors='coerce').mean()
                                asset_cols.append(col)
                            except:
                                continue
                    
                    if asset_cols:
                        # Convert asset columns to numeric, replacing errors with 0
                        for col in asset_cols:
                            allocation_df_full[col] = pd.to_numeric(allocation_df_full[col], errors='coerce').fillna(0)
                        
                        # Get top 5 assets by average allocation
                        asset_means = allocation_df_full[asset_cols].mean()
                        top_assets = asset_means.nlargest(5).index
                        
                        fig_allocation = go.Figure()
                        
                        for asset in top_assets:
                            fig_allocation.add_trace(go.Scatter(
                                x=allocation_df_full.index,
                                y=allocation_df_full[asset] * 100,
                                mode='lines',
                                stackgroup='one',
                                name=asset,
                                fill='tonexty'
                            ))
                        
                        fig_allocation.update_layout(
                            title='Portfolio Allocation Over Time',
                            xaxis_title='Date',
                            yaxis_title='Allocation %',
                            height=400
                        )
                        
                        st.plotly_chart(fig_allocation, use_container_width=True)
                    else:
                        st.warning("No valid asset allocation data found for charting")
                
            except Exception as e:
                st.warning(f"Could not create allocation chart: {str(e)}")
                st.write("Debug info:")
                st.write("Column dtypes:", allocation_df_full.dtypes)
                st.write("Sample data:", allocation_df_full.head())
            
            # Regime analysis
            st.markdown("### Regime Analysis")
            if 'regime' in allocation_df_full.columns:
                regime_counts = allocation_df_full['regime'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_regime = px.pie(
                        values=regime_counts.values,
                        names=regime_counts.index,
                        title="Time Spent in Each Regime",
                        color_discrete_map={
                            'risk_on': '#22c55e',
                            'neutral': '#f59e0b', 
                            'risk_off': '#ef4444'
                        }
                    )
                    st.plotly_chart(fig_regime, use_container_width=True)
                
                with col2:
                    st.markdown("#### Regime Statistics")
                    regime_stats = pd.DataFrame({
                        'Regime': regime_counts.index,
                        'Periods': regime_counts.values,
                        'Percentage': [f"{v/regime_counts.sum():.1%}" for v in regime_counts.values]
                    })
                    st.dataframe(regime_stats, hide_index=True)
        else:
            st.warning("No allocation history available")
    
    # Continue with tab3 and tab4 as before...
    with tab3:
        st.markdown("## Sentiment Analysis")
        
        # Sentiment overview
        latest_sentiment = sentiment_data['compound'].iloc[-7:].mean()  # Last 7 days average
        sentiment_zscore = ((sentiment_data['compound'] - sentiment_data['compound'].rolling(90).mean()) / 
                          sentiment_data['compound'].rolling(90).std()).iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            sentiment_color = "#22c55e" if latest_sentiment > 0.1 else "#ef4444" if latest_sentiment < -0.1 else "#f59e0b"
            st.markdown(f"""
            <div class="sentiment-gauge" style="background: linear-gradient(135deg, {sentiment_color}20, {sentiment_color}10);">
                <h3>Current Sentiment</h3>
                <h1 style="color: {sentiment_color};">{latest_sentiment:.2f}</h1>
                <p>{get_sentiment_label(latest_sentiment)}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            zscore_color = "#22c55e" if sentiment_zscore > 1 else "#ef4444" if sentiment_zscore < -1 else "#f59e0b"
            st.markdown(f"""
            <div class="sentiment-gauge" style="background: linear-gradient(135deg, {zscore_color}20, {zscore_color}10);">
                <h3>Sentiment Z-Score</h3>
                <h1 style="color: {zscore_color};">{sentiment_zscore:.2f}</h1>
                <p>{'Risk-On' if sentiment_zscore > 1 else 'Risk-Off' if sentiment_zscore < -1 else 'Neutral'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            regime = 'Risk-On' if sentiment_zscore > config.SENTIMENT_THRESHOLDS['risk_on'] else 'Risk-Off' if sentiment_zscore < config.SENTIMENT_THRESHOLDS['risk_off'] else 'Neutral'
            regime_color = "#22c55e" if regime == 'Risk-On' else "#ef4444" if regime == 'Risk-Off' else "#f59e0b"
            st.markdown(f"""
            <div class="sentiment-gauge" style="background: linear-gradient(135deg, {regime_color}20, {regime_color}10);">
                <h3>Current Regime</h3>
                <h1 style="color: {regime_color};">{regime}</h1>
                <p>Strategy Mode</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Sentiment charts
        fig_sentiment = create_sentiment_analysis_chart(sentiment_data, strategy_returns)
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Sentiment statistics
        st.markdown("### Sentiment Statistics")
        
        sentiment_stats = pd.DataFrame({
            'Metric': [
                'Mean Sentiment',
                'Sentiment Volatility', 
                'Positive Days %',
                'Negative Days %',
                'Extreme Sentiment Days %',
                'Current 30-Day Trend'
            ],
            'Value': [
                f"{sentiment_data['compound'].mean():.3f}",
                f"{sentiment_data['compound'].std():.3f}",
                f"{(sentiment_data['compound'] > 0.1).mean():.1%}",
                f"{(sentiment_data['compound'] < -0.1).mean():.1%}",
                f"{(abs(sentiment_data['compound']) > 0.5).mean():.1%}",
                f"{sentiment_data['compound'].tail(30).mean():.3f}"
            ]
        })
        
        st.dataframe(sentiment_stats, hide_index=True)
    
    with tab4:
        st.markdown("## 🤖 AI-Powered Analysis")
        
        with st.spinner("Generating comprehensive AI analysis..."):
            # Prepare regime history for AI analysis
            regime_df = pd.DataFrame()
            if not allocation_history.empty:
                regime_df = pd.DataFrame(allocation_history)[['date', 'regime']].copy()
                regime_df['date'] = pd.to_datetime(regime_df['date'])
            
            ai_analysis = generate_enhanced_ai_summary(
                performance_metrics=performance_metrics,
                latest_sentiment=latest_sentiment,
                latest_weights=latest_weights,
                regime_history=regime_df
            )
        
        st.markdown(ai_analysis)
        
        # Add interactive Q&A section
        st.markdown("---")
        st.markdown("### 💬 Interactive Analysis")
        
        user_question = st.text_input(
            "Ask a specific question about the strategy performance:",
            placeholder="e.g., How does the strategy perform during market downturns?"
        )
        
        if user_question and st.button("Get AI Response"):
            if Config.OPENROUTER_API_KEY:
                with st.spinner("Analyzing your question..."):
                    try:
                        client = OpenAI(
                            base_url="https://openrouter.ai/api/v1",
                            api_key=Config.OPENROUTER_API_KEY
                        )
                        
                        context_prompt = f"""
                        Based on the AlphaSent strategy backtest results:
                        Performance Metrics: {performance_metrics}
                        Latest Sentiment: {latest_sentiment}
                        Current Allocation: {latest_weights.head(5).to_dict()}
                        
                        User Question: {user_question}
                        
                        Provide a detailed, data-driven response referencing the specific metrics and results.
                        """
                        
                        response = client.chat.completions.create(
                            model="google/gemini-2.0-flash-exp",
                            messages=[{"role": "user", "content": context_prompt}],
                            temperature=0.2
                        )
                        
                        st.markdown("#### AI Response:")
                        st.markdown(response.choices[0].message.content)
                        
                    except Exception as e:
                        st.error(f"Failed to generate response: {e}")
            else:
                st.warning("Please configure OpenRouter API key for interactive analysis.")
        
        # Strategy recommendations
        st.markdown("---")
        st.markdown("### 📋 Strategy Recommendations")
        
        recommendations = []
        
        # Performance-based recommendations
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        max_dd = performance_metrics.get('max_drawdown', 0)
        
        if sharpe > 1.5:
            recommendations.append("✅ **Strong Performance**: The strategy shows excellent risk-adjusted returns.")
        elif sharpe < 1.0:
            recommendations.append("⚠️ **Performance Review**: Consider adjusting sentiment thresholds or position sizing.")
        
        if max_dd < -0.30:
            recommendations.append("🛡️ **Risk Management**: Consider implementing stricter stop-loss mechanisms.")
        
        # Sentiment-based recommendations
        if abs(sentiment_zscore) > 2:
            recommendations.append("🚨 **Extreme Sentiment**: Current market sentiment is at extreme levels - exercise caution.")
        
        # Allocation-based recommendations
        if latest_weights.max() > 0.4:
            recommendations.append("⚖️ **Diversification**: Consider reducing concentration in top holdings.")
        
        if len(latest_weights) < 3:
            recommendations.append("📊 **Portfolio Breadth**: Consider expanding to more assets for better diversification.")
        
        for rec in recommendations:
            st.markdown(rec)
        
        if not recommendations:
            st.success("🎯 **All Systems Green**: The strategy is performing well within acceptable parameters.")

# Initialize session state
if 'backtest_complete' not in st.session_state:
    st.session_state.backtest_complete = False

# Run main application
if __name__ == "__main__":
    main()

# ==============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ==============================================================================

def export_results_to_csv(strategy_returns: pd.Series, allocation_history: pd.DataFrame, performance_metrics: Dict):
    """Export backtest results to downloadable CSV files."""
    
    # Strategy returns export
    returns_df = strategy_returns.to_frame('strategy_returns')
    returns_df['cumulative_returns'] = (1 + strategy_returns).cumprod()
    returns_csv = returns_df.to_csv()
    
    # Allocation history export
    allocation_csv = allocation_history.to_csv(index=False)
    
    # Performance metrics export
    metrics_df = pd.DataFrame(list(performance_metrics.items()), columns=['Metric', 'Value'])
    metrics_csv = metrics_df.to_csv(index=False)
    
    return returns_csv, allocation_csv, metrics_csv

def create_risk_report(strategy_returns: pd.Series, prices_df: pd.DataFrame) -> Dict:
    """Generate comprehensive risk assessment report."""
    
    if strategy_returns.empty:
        return {}
    
    # VaR calculations
    var_95 = np.percentile(strategy_returns, 5)
    var_99 = np.percentile(strategy_returns, 1)
    
    # Conditional VaR (Expected Shortfall)
    cvar_95 = strategy_returns[strategy_returns <= var_95].mean()
    cvar_99 = strategy_returns[strategy_returns <= var_99].mean()
    
    # Tail ratios
    upside_returns = strategy_returns[strategy_returns > strategy_returns.median()]
    downside_returns = strategy_returns[strategy_returns < strategy_returns.median()]
    
    tail_ratio = (upside_returns.mean() / abs(downside_returns.mean())) if downside_returns.mean() != 0 else 0
    
    # Maximum consecutive losses
    losses = strategy_returns < 0
    max_consecutive_losses = 0
    current_consecutive = 0
    
    for loss in losses:
        if loss:
            current_consecutive += 1
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive)
        else:
            current_consecutive = 0
    
    # Correlation with Bitcoin (if available)
    btc_correlation = 0
    if 'BTC' in prices_df.columns:
        btc_returns = prices_df['BTC'].pct_change().dropna()
        common_dates = strategy_returns.index.intersection(btc_returns.index)
        if len(common_dates) > 30:
            btc_correlation = strategy_returns.loc[common_dates].corr(btc_returns.loc[common_dates])
    
    return {
        'var_95': var_95,
        'var_99': var_99,
        'cvar_95': cvar_95,
        'cvar_99': cvar_99,
        'tail_ratio': tail_ratio,
        'max_consecutive_losses': max_consecutive_losses,
        'btc_correlation': btc_correlation,
        'skewness': strategy_returns.skew(),
        'kurtosis': strategy_returns.kurtosis()
    }

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
    <p>AlphaSent v2.0 - Enhanced Sentiment-Driven Cryptocurrency Portfolio Management</p>
    <p>⚠️ <strong>Disclaimer:</strong> This is a research tool. Past performance does not guarantee future results. 
    Cryptocurrency investments carry significant risk.</p>
</div>
""", unsafe_allow_html=True)

# streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
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

# ------------------------------------------------------------------------------#
# Setup
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")
pio.templates.default = "plotly_white"

st.set_page_config(
    page_title="Project AlphaSent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main > div { padding-top: 1.25rem; }
    .news-item { border-left:4px solid #3498db; padding:0.75rem; margin:0.4rem 0; background:#f8f9fa; border-radius:0 10px 10px 0;}
    .risk-on { border-left-color:#22c55e!important; }
    .risk-off { border-left-color:#ef4444!important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Project AlphaSent")
st.caption("*A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation*")

# ------------------------------------------------------------------------------#
# Config
# ------------------------------------------------------------------------------#
class Config:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    CRYPTOCOMPARE_API_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
    DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
    SENTIMENT_THRESHOLDS = {"risk_on": 1.0, "risk_off": -1.0}  # Z of market sentiment
    TRANSACTION_COST = 0.0025  # 25 bps per side
    SLIPPAGE = 0.0005          # 5 bps slippage on turnover
    MAX_POSITION_SIZE = 0.30
    LOOKBACK_PERIOD = 90
    TURNOVER_CAP = 0.50        # 50% per rebalance (L1)
    SENTIMENT_TILT = 0.10      # beta for μ tilt via per-asset sentiment

@st.cache_resource
def setup_nltk():
    import nltk
    try:
        nltk.download("vader_lexicon", quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        return False

# ------------------------------------------------------------------------------#
# Requests + Data
# ------------------------------------------------------------------------------#
def create_requests_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504, 429])
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    return s

@st.cache_data(ttl=1800)
def load_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url, index_col=0, parse_dates=True)
        df.index.name = "time"
        if df.empty:
            raise ValueError("Loaded data is empty")

        # Drop all-NaN columns and ensure numeric for price columns
        df = df.dropna(axis=1, how="all")
        # prices: symbols with uppercase tickers; sentiments: 'compound_mkt', 'sent_<SYM>'
        for c in df.columns:
            if c == "compound_mkt" or c.startswith("sent_") or c.isupper():
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=300)
def get_sentiment_label(score: float) -> str:
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"

@st.cache_data(ttl=900)
def fetch_live_news(api_key: str) -> pd.DataFrame:
    """Fetch and VADER-score latest crypto news titles (display only)."""
    if not api_key:
        return pd.DataFrame()
    session = create_requests_session()
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"
    try:
        r = session.get(url, timeout=30); r.raise_for_status()
        data = r.json().get("Data", [])
        if not data:
            return pd.DataFrame()
        df = pd.DataFrame(data).head(20)
        # Sentiment on the fly (titles)
        analyzer = SentimentIntensityAnalyzer()
        df["compound"] = df["title"].fillna("").apply(lambda t: analyzer.polarity_scores(t)["compound"])
        df["sentiment_label"] = df["compound"].apply(get_sentiment_label)
        df["impact_score"] = df["compound"].abs()
        return df[["title", "source", "compound", "sentiment_label", "impact_score", "url"]]
    except Exception:
        return pd.DataFrame()

# ------------------------------------------------------------------------------#
# Utilities
# ------------------------------------------------------------------------------#
def compute_returns_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """Detect prices vs returns and compute safe daily returns."""
    df = df.copy()
    looks_like_prices = (df.min(numeric_only=True).min() > 0) and (df.median(numeric_only=True).median() > 1)
    if looks_like_prices:
        rets = df.pct_change()
    else:
        rets = df
    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.95)
    return rets

def extract_asset_universe(prices_df: pd.DataFrame) -> list:
    """Heuristic: assets are all-uppercase tickers (BTC, ETH, …)."""
    return [c for c in prices_df.columns if c.isupper()]

def extract_per_asset_sentiment(sent_df: pd.DataFrame, date) -> pd.Series:
    """Pull per-asset sentiment on or before 'date' (columns: sent_<SYM>)."""
    cols = [c for c in sent_df.columns if c.startswith("sent_")]
    if not cols:
        return pd.Series(dtype=float)
    s = sent_df.loc[:date, cols].tail(1)
    if s.empty:
        return pd.Series(dtype=float)
    s = s.squeeze()
    s.index = [c.replace("sent_", "") for c in s.index]
    return s

def get_market_sent_z(sent_df: pd.DataFrame, date, window: int = 90) -> float:
    """Compute market sentiment Z of 'compound_mkt' up to 'date'."""
    if "compound_mkt" not in sent_df.columns:
        return 0.0
    s = sent_df.loc[:date, "compound_mkt"].dropna()
    if len(s) < 30:
        return 0.0
    z = (s - s.rolling(window).mean()) / s.rolling(window).std()
    z = z.replace([np.inf, -np.inf], np.nan).dropna()
    return float(z.iloc[-1]) if len(z) else 0.0

# ------------------------------------------------------------------------------#
# Optimizers
# ------------------------------------------------------------------------------#
    """Add this function to debug model selection issues"""
    st.sidebar.markdown("**Debug Info**")
    if 'last_model_used' in st.session_state:
        st.sidebar.text(f"Last model: {st.session_state.last_model_used}")
    if 'last_weights_sum' in st.session_state:
        st.sidebar.text(f"Weights sum: {st.session_state.last_weights_sum:.4f}")
    if 'optimization_count' in st.session_state:
        st.sidebar.text(f"Optimizations: {st.session_state.optimization_count}")

# CACHING FIX: Update your caching decorators

class PortfolioOptimizer:
    def __init__(self, transaction_cost: float = 0.0025, max_weight: float = 0.30, slippage: float = 0.0005):
        self.transaction_cost = transaction_cost
        self.max_weight = max_weight
        self.slippage = slippage

    def clean_price_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Loosen NaN handling so partial histories are allowed.
        Returns an empty DataFrame instead of raising if no valid assets remain.
        """
        min_obs = max(30, len(prices) // 10)

        # Require min_obs valid points; allow forward-fill for partial histories
        valid_cols = prices.count()[lambda x: x >= min_obs].index
        cleaned = prices[valid_cols].ffill().dropna(how="all")

        # Remove assets with near-zero volatility
        returns = cleaned.pct_change().dropna()
        valid_assets = returns.std()[lambda x: x > 1e-6].index
        cleaned = cleaned[valid_assets]

        return cleaned  # may be empty

    def get_optimized_weights(
        self,
        prices: pd.DataFrame,
        model: str = "max_sharpe",
        sentiment_scores: Optional[pd.Series] = None,
        market_regime: str = "neutral",
        turnover_cap: Optional[float] = None,
        last_weights: Optional[pd.Series] = None,
        base_max_weight: float = 0.30,
        beta_sent: float = 0.10,
    ) -> Tuple[pd.Series, Dict]:
        try:
            clean_prices = self.clean_price_data(prices)

            # Handle case: no valid or too few assets after cleaning
            if clean_prices.empty or len(clean_prices.columns) < 2:
                if last_weights is not None and not last_weights.empty:
                    # Carry forward previous allocation
                    return last_weights.reindex(prices.columns, fill_value=0.0), {
                        "method": "carry_forward", "reason": "no_valid_assets"
                    }
                else:
                    # Fall back to equal-weight allocation over whatever assets exist in prices
                    return self._fallback_weights(list(prices.columns)), {
                        "method": "equal_weight", "reason": "no_valid_assets"
                    }

            assets = list(clean_prices.columns)

            # Bounds adjust by regime
            if market_regime == "risk_off":
                max_w = min(0.20, base_max_weight)
            elif market_regime == "risk_on":
                max_w = min(0.40, max(0.30, base_max_weight))
            else:
                max_w = base_max_weight

            mu = expected_returns.ema_historical_return(clean_prices, frequency=365)
            S = risk_models.CovarianceShrinkage(clean_prices).ledoit_wolf()

            # Sentiment tilt
            if (sentiment_scores is not None) and (len(sentiment_scores) > 0):
                aligned = sentiment_scores.reindex(mu.index).fillna(0.0)
                if aligned.sum() == 0:
                    st.warning(f"No sentiment match for assets: {list(mu.index)}")
                tilt = np.clip(beta_sent * aligned, -0.2, 0.2)
                mu = mu * (1.0 + tilt)

            # ===== FIX 1: Ensure model selection actually works =====
            w = None
            
            if model == "max_sharpe":
                ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
                ef.add_objective(L2_reg, gamma=0.01)
                ef.max_sharpe()
                w = pd.Series(ef.clean_weights(cutoff=0.005), dtype=float)
                
            elif model == "min_variance":
                ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
                ef.add_objective(L2_reg, gamma=0.01)
                ef.min_volatility()
                w = pd.Series(ef.clean_weights(cutoff=0.005), dtype=float)
                
            elif model == "max_qu":  # ← FIXED: This was the main issue
                ef = EfficientFrontier(mu, S, weight_bounds=(0, max_w))
                ef.add_objective(L2_reg, gamma=0.01)
                ef.max_quadratic_utility(risk_aversion=1.0)  # Fixed method name
                w = pd.Series(ef.clean_weights(cutoff=0.005), dtype=float)
                
            elif model == "erc":
                w = self._erc_weights(S, max_w=max_w)
                
            elif model == "equal_weight":
                w = pd.Series(1.0 / len(assets), index=assets, dtype=float)
                
            else:
                st.error(f"❌ Unknown model: {model}")
                return self._fallback_weights(list(prices.columns)), {
                    "method": "equal_weight", "reason": f"unknown_model_{model}"
                }
            
            # ===== FIX 2: Ensure normalization =====
            if w is None or w.empty:
                w = self._fallback_weights(assets)
            else:
                w = w / w.sum() if w.sum() > 0 else w

            # ===== FIX 3: Apply turnover cap correctly =====
            if turnover_cap is not None and last_weights is not None and not last_weights.empty:
                w = self._apply_turnover_cap(last_weights.reindex(w.index, fill_value=0.0), w, cap=turnover_cap)

            # ===== FIX 4: Ensure portfolio stats are calculated correctly =====
            try:
                port_ret = float(w.dot(mu))
                port_vol = float(np.sqrt(w.T @ S @ w))
                sharpe = port_ret / port_vol if port_vol > 0 else 0.0
            except Exception as e:
                st.warning(f"Portfolio stats calculation failed: {e}")
                port_ret = port_vol = sharpe = 0.0

            meta = {
                "method": model,
                "expected_return": port_ret,
                "volatility": port_vol,
                "sharpe_ratio": sharpe,
                "n_assets": int((w > 0).sum()),
                "max_weight": max_w,
                "market_regime": market_regime,
            }

            return w.reindex(prices.columns, fill_value=0.0), meta

        except (OptimizationError, ValueError, cp.SolverError) as e:
            st.error(f"Optimization failed for assets {list(prices.columns)}: {e}")
            return self._fallback_weights(list(prices.columns)), {
                "method": "equal_weight", "reason": str(e)
            }

    def _fallback_weights(self, asset_names: list) -> pd.Series:
        """Equal weight allocation for the given asset list."""
        if not asset_names:
            return pd.Series(dtype=float)
        return pd.Series(1.0 / len(asset_names), index=asset_names, dtype=float)
    
    def _erc_weights(self, cov_matrix: pd.DataFrame, max_w: float = 0.30) -> pd.Series:
        """Equal Risk Contribution portfolio optimization using CVXPY."""
        try:
            import cvxpy as cp
            n = len(cov_matrix)
            w = cp.Variable(n, nonneg=True)
            
            # Risk contributions should be equal
            risk_contrib = cp.multiply(w, cov_matrix @ w)
            
            # Minimize the sum of squared differences from equal risk contribution
            target_contrib = cp.sum(risk_contrib) / n
            obj = cp.sum_squares(risk_contrib - target_contrib)
            
            constraints = [
                cp.sum(w) == 1.0,  # weights sum to 1
                w <= max_w,       # max position size
                w >= 0.0          # long-only
            ]
            
            prob = cp.Problem(cp.Minimize(obj), constraints)
            prob.solve(solver=cp.ECOS)
            
            if w.value is None:
                raise ValueError("ERC optimization failed")
                
            weights = pd.Series(w.value, index=cov_matrix.index)
            return weights / weights.sum()  # normalize
            
        except Exception as e:
            st.warning(f"ERC failed: {e}, falling back to equal weight")
            return pd.Series(1.0/len(cov_matrix), index=cov_matrix.index)

    def _apply_turnover_cap(self, last_weights: pd.Series, target_weights: pd.Series, cap: float) -> pd.Series:
        """Apply turnover constraint by scaling back changes from last weights."""
        if last_weights.empty:
            return target_weights
        
        # Align indices
        last_w = last_weights.reindex(target_weights.index, fill_value=0.0)
        
        # Calculate proposed changes
        changes = target_weights - last_w
        total_turnover = changes.abs().sum()
        
        if total_turnover <= cap:
            return target_weights
        
        # Scale back changes to meet turnover cap
        scale_factor = cap / total_turnover
        adjusted_changes = changes * scale_factor
        adjusted_weights = last_w + adjusted_changes
        
        # Ensure non-negative and normalized
        adjusted_weights = adjusted_weights.clip(lower=0.0)
        return adjusted_weights / adjusted_weights.sum() if adjusted_weights.sum() > 0 else target_weights


# ===== FIX 5: Ensure the backtest engine processes model choice correctly =====

class BacktestEngine:
    def __init__(self, optimizer: PortfolioOptimizer, config: Config):
        self.optimizer = optimizer
        self.config = config

    def run_backtest(self, prices_df: pd.DataFrame, sent_df: pd.DataFrame, model_choice: str, beta_sent: float) -> Tuple[pd.Series, pd.DataFrame, Dict]:
        st.info("🚀 Initiating Enhanced Backtest Engine...")

        assets = extract_asset_universe(prices_df)
        px = prices_df[assets].dropna(how="all")
        daily_returns = px.pct_change().fillna(0.0)

        # Monthly end rebal dates present in index
        rebalance_dates = px.resample("ME").last().index
        rebalance_dates = [d for d in rebalance_dates if d in px.index]
        if len(rebalance_dates) < 3:
            st.error("Insufficient data for backtesting")
            return pd.Series(dtype=float), pd.DataFrame(), {}

        portfolio_returns = []
        allocation_history = []
        transaction_costs = []
        last_weights = pd.Series(dtype=float)

        progress = st.progress(0)
        status = st.empty()

        for i, (cur, nxt) in enumerate(zip(rebalance_dates[:-1], rebalance_dates[1:])):
            progress.progress((i + 1) / (len(rebalance_dates) - 1))
            status.text(f"Processing {cur.strftime('%Y-%m')}")

            # Determine regime via market sentiment z
            z = get_market_sent_z(sent_df, cur, window=self.config.LOOKBACK_PERIOD)
            if z > self.config.SENTIMENT_THRESHOLDS["risk_on"]:
                regime = "risk_on"
            elif z < self.config.SENTIMENT_THRESHOLDS["risk_off"]:
                regime = "risk_off"
            else:
                regime = "neutral"

            # Lookback window for optimization
            hist_start = cur - timedelta(days=self.config.LOOKBACK_PERIOD)
            hist_prices = px.loc[hist_start:cur].dropna(how="all")
            if len(hist_prices) < 30:
                continue

            # Pull per-asset sentiment snapshot at cur
            per_asset_sent = extract_per_asset_sentiment(sent_df, cur).reindex(assets).fillna(0.0)

            # ===== FIX 6: Properly handle model selection =====
            if model_choice == "auto":
                model = {"risk_on": "max_sharpe", "risk_off": "min_variance", "neutral": "max_qu"}[regime]
            else:
                model = model_choice  # Use the exact model selected by user

            # Debug info (only show on first iteration to avoid spam)
            if i == 0:
                st.info(f"🎯 Using model: {model} (selected: {model_choice}, regime: {regime})")

            target_w, meta = self.optimizer.get_optimized_weights(
                hist_prices,
                model=model,
                sentiment_scores=per_asset_sent,
                market_regime=regime,
                turnover_cap=self.config.TURNOVER_CAP,
                last_weights=last_weights,
                base_max_weight=self.config.MAX_POSITION_SIZE,
                beta_sent=beta_sent,
            )

            # Turnover & trading costs
            if not last_weights.empty:
                turnover = (target_w - last_weights.reindex(target_w.index, fill_value=0)).abs().sum()
                txn = turnover * (self.config.TRANSACTION_COST + self.config.SLIPPAGE)
                transaction_costs.append(float(txn))
            else:
                transaction_costs.append(0.0)

            # Realize period returns (apply txn cost on first day)
            period_rets = daily_returns.loc[cur:nxt]
            if period_rets is None or len(period_rets) < 2:
                continue
            
            p = (period_rets * target_w).sum(axis=1).astype(float)
            
            # apply transaction cost on the first day of the period
            p.iloc[0] = p.iloc[0] - float(transaction_costs[-1])
            
            # guard against inf/NaN
            p = pd.to_numeric(p, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if p.empty:
                continue

            portfolio_returns.append(p)

            rec = target_w.to_dict()
            rec.update({
                "date": cur, "regime": regime, **meta, "transaction_cost": transaction_costs[-1],
                "sent_mkt_z": z
            })
            allocation_history.append(rec)

            last_weights = target_w

        progress.empty(); status.empty()

        if not portfolio_returns:
            st.error("No valid returns generated")
            return pd.Series(dtype=float), pd.DataFrame(), {}

        strategy_returns = pd.concat(portfolio_returns).astype(float)
        allocation_df = pd.DataFrame(allocation_history)

        metrics = self._calculate_performance_metrics(strategy_returns, px, transaction_costs)

        st.success("✅ Backtest completed successfully!")
        return strategy_returns, allocation_df, metrics


    def _calculate_performance_metrics(self, returns: pd.Series, prices_df: pd.DataFrame, transaction_costs: list) -> Dict:
        if returns.empty: 
            return {"error": "No returns to analyze"}
            
        r = pd.to_numeric(returns, errors="coerce")
        r = r.replace([np.inf, -np.inf], np.nan).dropna()
    
        if len(r) < 10:  # Need minimum observations
            return {"error": "Insufficient return observations"}
            
        # Safer calculations with bounds checking
        try:
            cum_ret = np.prod(1.0 + r.clip(-0.99, 3.0))  # Clip extreme returns
            periods_per_year = 365.0
            years = len(r) / periods_per_year
            
            ann_ret = (cum_ret ** (1.0 / years) - 1.0) if cum_ret > 0 and years > 0 else 0.0
            ann_vol = r.std() * np.sqrt(periods_per_year) if len(r) > 1 else 0.0
            
            # Safer Sharpe calculation
            sharpe = (ann_ret / ann_vol) if (ann_vol > 1e-6 and np.isfinite(ann_ret) and np.isfinite(ann_vol)) else 0.0
            
            # Safer drawdown calculation
            cum_curve = (1 + r).cumprod()
            running_max = cum_curve.expanding().max()
            drawdown = (cum_curve - running_max) / running_max
            max_dd = drawdown.min()
            
            # Benchmark comparison (BTC if available)
            bench = {}
            if "BTC" in prices_df.columns:
                btc = compute_returns_from_data(prices_df[["BTC"]])["BTC"]
                # align BTC exactly to the strategy period
                btc = btc.reindex(r.index).dropna()
                btc = pd.to_numeric(btc, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(btc) > 0:
                    cum_btc = float(np.prod(1.0 + btc.values))
                    btc_ann_ret = (cum_btc ** (365.0 / len(btc)) - 1.0) if cum_btc > 0 else np.nan
                    btc_vol = float(np.nan_to_num(btc.std(), nan=0.0)) * np.sqrt(365.0)
                    btc_sharpe = (btc_ann_ret / btc_vol) if (btc_vol > 0 and np.isfinite(btc_ann_ret)) else 0.0
                    btc_curve = (1 + btc).cumprod()
                    btc_dd = (btc_curve - btc_curve.cummax()) / btc_curve.cummax()
                    bench = {
                        "btc_annual_return": btc_ann_ret,
                        "btc_volatility": btc_vol,
                        "btc_sharpe_ratio": btc_sharpe,
                        "btc_max_drawdown": float(btc_dd.min()),
                        "excess_return": (ann_ret - btc_ann_ret) if np.isfinite(ann_ret) and np.isfinite(btc_ann_ret) else np.nan,
                        "excess_sharpe": sharpe - btc_sharpe,
                    }

            # Transaction cost analysis
            total_txn = float(sum(transaction_costs))
            avg_annual_txn = total_txn * (365.0 / len(r)) if len(r) > 0 else 0.0

            # Downside risk metrics
            downside = r[r < 0]
            down_vol = (downside.std() * np.sqrt(365.0)) if len(downside) > 1 else 0.0
            sortino = (ann_ret / down_vol) if (down_vol > 0 and np.isfinite(ann_ret)) else 0.0

            return {
                "total_return": cum_ret - 1.0,
                "annualized_return": ann_ret,
                "annualized_volatility": ann_vol,
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "max_drawdown": max_dd,
                "total_transaction_costs": total_txn,
                "avg_annual_transaction_cost": avg_annual_txn,
                "observation_count": len(r),
                "years_analyzed": years,
                **bench,
            }
            
        except Exception as e:
            st.error(f"❌ Performance metrics calculation failed: {e}")
            return {"error": f"Metrics calculation failed: {e}"}

# ------------------------------------------------------------------------------#
# Charts
# ------------------------------------------------------------------------------#
def performance_dashboard(strategy_returns: pd.Series, prices_df: pd.DataFrame):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Cumulative Performance", "Rolling Sharpe (90d)", "Drawdown", "Monthly Returns Heatmap"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]],
        vertical_spacing=0.12,
    )
    strat_cum = (1 + strategy_returns).cumprod()
    fig.add_trace(go.Scatter(x=strat_cum.index, y=strat_cum.values, name="AlphaSent Strategy", line=dict(width=3)), row=1, col=1)

    if "BTC" in prices_df.columns:
        btc = compute_returns_from_data(prices_df[["BTC"]])["BTC"]
        btc_cum = (1 + btc).cumprod().reindex(strat_cum.index, method="ffill")
        fig.add_trace(go.Scatter(x=btc_cum.index, y=btc_cum.values, name="Bitcoin", line=dict(width=2, dash="dash")), row=1, col=1)

    roll_sharpe = strategy_returns.rolling(90).apply(lambda x: (x.mean() / x.std() * np.sqrt(365)) if x.std() > 0 else 0.0)
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe.values, name="90-Day Rolling Sharpe"), row=1, col=2)

    rolling_max = strat_cum.cummax()
    dd = (strat_cum - rolling_max) / rolling_max * 100.0
    fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown %", fill="tozeroy"), row=2, col=1)

    monthly = strategy_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    heat = monthly.to_frame("ret")
    heat["Year"] = heat.index.year.astype(str)
    heat["Month"] = heat.index.strftime("%b")
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun","Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot = heat.pivot(index="Year", columns="Month", values="ret").reindex(columns=order)
    if not pivot.empty:
        fig.add_trace(
            go.Heatmap(
                z=pivot.values * 100.0,
                x=pivot.columns,
                y=pivot.index,
                colorscale="RdYlGn",
                colorbar_title="%",
                hovertemplate="%{y} %{x}: %{z:.2f}%<extra></extra>",
                name="Monthly Returns %",
            ),
            row=2, col=2,
        )
    fig.update_yaxes(title_text="Cumulative Return", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
    fig.update_layout(height=800, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def sentiment_chart(sent_df: pd.DataFrame, strategy_returns: pd.Series):
    if "compound_mkt" not in sent_df.columns:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Market Sentiment Z-Score", "Sentiment vs Strategy"),
                        vertical_spacing=0.12, specs=[[{}],[{"secondary_y":True}]])
    s = sent_df["compound_mkt"].fillna(0.0)
    z = (s - s.rolling(90).mean()) / s.rolling(90).std()
    z = z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    fig.add_trace(go.Scatter(x=z.index, y=z.values, name="Sentiment Z"), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color="green", annotation_text="Risk-On", row=1, col=1)
    fig.add_hline(y=-1.0, line_dash="dash", line_color="red", annotation_text="Risk-Off", row=1, col=1)

    if not strategy_returns.empty:
        cum = (1 + strategy_returns).cumprod()
        fig.add_trace(go.Scatter(x=cum.index, y=cum.values, name="Strategy"), row=2, col=1, secondary_y=True)

    fig.update_yaxes(title_text="Z-Score", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative Return", row=2, col=1, secondary_y=True)
    fig.update_layout(height=600)
    return fig

# ------------------------------------------------------------------------------#
# AI Summary
# ------------------------------------------------------------------------------#
def generate_ai_summary(perf: Dict, latest_sentiment: float, latest_weights: pd.Series, regime_history: pd.DataFrame) -> str:
    if not Config.OPENROUTER_API_KEY:
        return "⚠️ Add your OpenRouter API key in `st.secrets` to enable AI insights."
    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=Config.OPENROUTER_API_KEY)
        top = latest_weights.nlargest(5) if not latest_weights.empty else pd.Series(dtype=float)
        reg_stats = regime_history["regime"].value_counts() if not regime_history.empty else pd.Series(dtype=float)
        prompt = f"""
Act as a quantitative crypto PM. Analyze this backtest:

Annual Return: {perf.get('annualized_return', 0):.2%}
Sharpe: {perf.get('sharpe_ratio', 0):.2f}
Max DD: {perf.get('max_drawdown', 0):.2%}
Volatility: {perf.get('annualized_volatility', 0):.2%}
Excess vs BTC: {perf.get('excess_return', 0):.2%}
Excess Sharpe vs BTC: {perf.get('excess_sharpe', 0):.2f}

Latest Market Sentiment Z: {latest_sentiment:.2f}
Top 5 holdings: {top.to_dict()}
Regime distribution: {reg_stats.to_dict()}

Provide: (1) executive summary, (2) effect of sentiment, (3) risk, (4) positioning rationale, (5) forward outlook. <= 500 words.
"""
        out = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        return out.choices[0].message.content
    except Exception as e:
        return f"❌ AI analysis failed: {e}"

# ------------------------------------------------------------------------------#
# Main app
# ------------------------------------------------------------------------------#
def fmt_pct(x):
    return "—" if (x is None or not np.isfinite(x)) else f"{x:.1%}"

def fmt_num(x, digs=2):
    return "—" if (x is None or not np.isfinite(x)) else f"{x:.{digs}f}"

def main():
    if not setup_nltk():
        st.stop()

    config = Config()
    optimizer = PortfolioOptimizer(max_weight=config.MAX_POSITION_SIZE, transaction_cost=config.TRANSACTION_COST, slippage=config.SLIPPAGE)
    engine = BacktestEngine(optimizer, config)
    

    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Control Center")
        col1, col2 = st.columns(2)
        col1.caption(("🟢" if config.CRYPTOCOMPARE_API_KEY else "🔴") + " CryptoCompare")
        col2.caption(("🟢" if config.OPENROUTER_API_KEY else "🔴") + " OpenRouter AI")
        st.divider()

        st.markdown("**Strategy Parameters**")
        t = st.slider("Sentiment Z Threshold (abs)", 0.5, 2.0, 1.0, 0.1)
        config.SENTIMENT_THRESHOLDS["risk_on"] = t
        config.SENTIMENT_THRESHOLDS["risk_off"] = -t

        config.MAX_POSITION_SIZE = st.slider("Max Position Size", 0.10, 0.50, config.MAX_POSITION_SIZE, 0.05)
        config.LOOKBACK_PERIOD = st.slider("Lookback Window (days)", 30, 180, config.LOOKBACK_PERIOD, 10)
        config.TURNOVER_CAP = st.slider("Turnover Cap per Rebalance (L1)", 0.10, 1.00, config.TURNOVER_CAP, 0.05)
        beta_sent = st.slider("Per-Asset Sentiment Tilt (β)", 0.0, 0.25, Config.SENTIMENT_TILT, 0.01)

        model_choice = st.selectbox("Optimization Model",
            options=[("Auto (regime-based)", "auto"),
                     ("Max Sharpe", "max_sharpe"),
                     ("Min Variance", "min_variance"),
                     ("Max Quadratic Utility", "max_qu"),
                     ("Equal Risk Contribution (ERC)", "erc"),
                     ("Equal Weight", "equal_weight")],
            format_func=lambda x: x[0]
        )[1]
        st.divider()

        st.markdown("**📰 Live News (titles only)**")
        news = fetch_live_news(config.CRYPTOCOMPARE_API_KEY)
        if not news.empty:
            for _, row in news.head(5).iterrows():
                cls = "risk-on" if row["compound"] > 0 else "risk-off"
                st.markdown(
                    f"""<div class="news-item {cls}">
                        <strong>{row['source']}</strong><br>
                        <a href="{row['url']}" target="_blank">{row['title'][:80]}...</a><br>
                        <small>Sentiment: {row['sentiment_label']} ({row['compound']:.2f})</small>
                    </div>""",
                    unsafe_allow_html=True,
                )
        else:
            st.info("News unavailable")

    st.divider()
    center = st.container()
    with center:
        run = st.button("🚀 Run Sentiment-Enhanced Backtest", type="primary", use_container_width=True)
    if not run:
        st.stop()

    data = load_data(config.DATA_URL)
    if data.empty:
        st.error("Failed to load data.")
        st.stop()

    # Split prices vs sentiment
    price_cols = [c for c in data.columns if c.isupper()]
    if not price_cols:
        st.error("No price columns detected.")
        st.stop()
    prices_df = data[price_cols]

    sent_cols = [c for c in data.columns if c == "compound_mkt" or c.startswith("sent_")]
    if not sent_cols:
        st.error("No sentiment columns found (expected 'compound_mkt' and 'sent_<SYMBOL>').")
        st.stop()
    sentiment_df = data[sent_cols].dropna(how="all")

    with st.spinner("Running backtest..."):
        strategy_returns, allocation_history, metrics = engine.run_backtest(prices_df, sentiment_df, model_choice=model_choice, beta_sent=beta_sent)

    if strategy_returns.empty:
        st.error("Backtest failed.")
        st.stop()

    # ======== Summary KPIs ======== #
    st.header("📊 Recommended Portfolio")
    valid_sr = pd.to_numeric(strategy_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if valid_sr.empty:
        strat_ret_annual = np.nan
        strat_vol_annual = np.nan
        strat_sharpe = 0.0
    else:
        # crypto trades daily; 365 is fine too — pick one and be consistent
        ann_factor = 365.0
        strat_ret_annual = valid_sr.mean() * ann_factor
        strat_vol_annual = valid_sr.std() * np.sqrt(ann_factor)
        strat_sharpe = (strat_ret_annual / strat_vol_annual) if (strat_vol_annual and strat_vol_annual > 0) else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Expected Annual Return (%)", f"{strat_ret_annual*100:.2f}")
    c2.metric("Annual Volatility (%)", f"{strat_vol_annual*100:.2f}")
    c3.metric("Sharpe Ratio", f"{strat_sharpe:.2f}")

    alloc_df = allocation_history.copy()
    if not alloc_df.empty:
        st.subheader("Current Asset Allocation")
        exclude = {"date","regime","transaction_cost","method","expected_return","volatility","sharpe_ratio","n_assets","max_weight","market_regime","sent_mkt_z"}
        latest_alloc = alloc_df.iloc[-1].drop(labels=exclude, errors="ignore")
        latest_alloc = pd.to_numeric(latest_alloc, errors="coerce").fillna(0.0)
        latest_alloc = latest_alloc[latest_alloc > 0] * 100
        if not latest_alloc.empty:
            fig = px.pie(values=latest_alloc.values, names=latest_alloc.index, title="Portfolio Allocation (%)", hole=0.3)
            fig.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active positions in the latest allocation.")

    st.subheader("Back Test Performance Dashboard")

    # ======== Tabs ======== #
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Performance", "🎯 Allocation", "📈 Sentiment", "🤖 AI Insights"])

    with tab1:
        st.subheader("Performance Dashboard")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Annual Return", fmt_pct(metrics.get("annualized_return")))
        k2.metric("Sharpe Ratio", fmt_num(metrics.get("sharpe_ratio")))
        k3.metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown")))
        k4.metric("Volatility", fmt_pct(metrics.get("annualized_volatility")))
        st.plotly_chart(performance_dashboard(strategy_returns, prices_df), use_container_width=True)

        st.markdown("##### Detailed Metrics")
        left, right = st.columns(2)
        base_df = pd.DataFrame(
            [
                ["Total Return", fmt_pct(metrics.get("total_return"))],
                ["Annualized Return", fmt_pct(metrics.get("annualized_return"))],
                ["Annualized Volatility", fmt_pct(metrics.get("annualized_volatility"))],
                ["Sharpe Ratio", fmt_num(metrics.get("sharpe_ratio"), 3)],
                ["Sortino Ratio", fmt_num(metrics.get("sortino_ratio"), 3)],
                ["Maximum Drawdown", fmt_pct(metrics.get("max_drawdown"))],
                ["Avg Annual Transaction Cost", fmt_pct(metrics.get("avg_annual_transaction_cost"))],
            ],
            columns=["Metric", "Value"],
        )
        left.dataframe(base_df, hide_index=True, use_container_width=True)

        if "btc_annual_return" in metrics:
            bench_df = pd.DataFrame(
                [
                    ["BTC Annual Return", fmt_pct(metrics.get("btc_annual_return"))],
                    ["BTC Sharpe Ratio", fmt_num(metrics.get("btc_sharpe_ratio"), 3)],
                    ["BTC Max Drawdown", fmt_pct(metrics.get("btc_max_drawdown"))],
                    ["Excess Return vs BTC", fmt_pct(metrics.get("excess_return"))],
                    ["Excess Sharpe vs BTC", fmt_num(metrics.get("excess_sharpe"), 3)],
                ],
                columns=["Benchmark Metric", "Value"],
            )
            right.dataframe(bench_df, hide_index=True, use_container_width=True)

    with tab2:
        st.subheader("Asset Allocation")
        if not alloc_df.empty:
            latest = pd.Series(alloc_df.iloc[-1])
            non_meta = {"date","regime","transaction_cost","method","expected_return","volatility","sharpe_ratio","n_assets","max_weight","market_regime","sent_mkt_z"}
            weights = latest[~latest.index.isin(non_meta)].apply(pd.to_numeric, errors="coerce").dropna()
            weights = weights[weights > 0.01].sort_values(ascending=False)

            colA, colB = st.columns(2)
            with colA:
                st.markdown("**Current Portfolio Allocation**")
                if not weights.empty:
                    if len(weights) <= 8:
                        fig = px.pie(values=weights.values, names=weights.index, title=None)
                        fig.update_traces(textposition="inside", textinfo="percent+label")
                    else:
                        fig = px.bar(x=weights.index, y=(weights.values * 100.0), labels={"x":"Asset","y":"Weight %"})
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Current allocation is effectively cash.")
            with colB:
                st.markdown("**Allocation Details**")
                if not weights.empty:
                    st.dataframe(
                        pd.DataFrame({"Asset": weights.index, "Weight": [f"{w:.1%}" for w in weights.values]}),
                        hide_index=True, use_container_width=True
                    )

            st.markdown("**Portfolio Evolution Over Time**")
            df = alloc_df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            asset_cols = [c for c in df.columns if c not in non_meta]
            for c in asset_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            if asset_cols:
                means = df[asset_cols].mean().nlargest(5).index
                area = go.Figure()
                for a in means:
                    area.add_trace(go.Scatter(x=df.index, y=df[a]*100, mode="lines", stackgroup="one", name=a))
                area.update_layout(yaxis_title="Allocation %", height=420, legend=dict(orientation="h"))
                st.plotly_chart(area, use_container_width=True)

            st.markdown("**Regime Analysis**")
            if "regime" in df.columns:
                counts = df["regime"].value_counts()
                if not counts.empty:
                    st.plotly_chart(px.pie(values=counts.values, names=counts.index, title="Time in Each Regime"),
                                    use_container_width=True)

    with tab3:
        st.subheader("Sentiment")
        if "compound_mkt" in sentiment_df.columns:
            latest_sent = sentiment_df["compound_mkt"].tail(7).mean()
            zscore = get_market_sent_z(sentiment_df, sentiment_df.index.max(), window=config.LOOKBACK_PERIOD)
        else:
            latest_sent, zscore = 0.0, 0.0
        c1, c2, c3 = st.columns(3)
        c1.metric("Current Market Sentiment", f"{latest_sent:.2f}", help=get_sentiment_label(latest_sent))
        c2.metric("Sentiment Z-Score", f"{zscore:.2f}")
        regime_now = "Risk-On" if zscore > config.SENTIMENT_THRESHOLDS["risk_on"] else "Risk-Off" if zscore < config.SENTIMENT_THRESHOLDS["risk_off"] else "Neutral"
        c3.metric("Current Regime", regime_now)
        st.plotly_chart(sentiment_chart(sentiment_df, strategy_returns), use_container_width=True)

        # Per-asset snapshot table if present
        snap = sentiment_df[[c for c in sentiment_df.columns if c.startswith("sent_")]].tail(1)
        if not snap.empty:
            out = snap.T.reset_index()
            out.columns = ["AssetSentCol", "Score"]
            out["Asset"] = out["AssetSentCol"].str.replace("sent_", "", regex=False)
            out = out[["Asset", "Score"]].sort_values("Score", ascending=False)
            st.dataframe(out, hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("🤖 AI-Powered Analysis")
        try:
            regime_df = pd.DataFrame(allocation_history)[["date","regime"]]
            regime_df["date"] = pd.to_datetime(regime_df["date"])
        except Exception:
            regime_df = pd.DataFrame(columns=["date","regime"])

        latest_weights = pd.Series(dtype=float)
        if not allocation_history.empty:
            last = pd.Series(allocation_history.iloc[-1])
            exclude = {"date","regime","transaction_cost","method","expected_return","volatility","sharpe_ratio","n_assets","max_weight","market_regime","sent_mkt_z"}
            latest_weights = last[~last.index.isin(exclude)].apply(pd.to_numeric, errors="coerce").dropna()

        ai_text = generate_ai_summary(metrics, zscore, latest_weights, regime_df)
        st.markdown(ai_text)

        st.divider()
        q = st.text_input("Ask a question about the results:")
        if q and Config.OPENROUTER_API_KEY:
            try:
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=Config.OPENROUTER_API_KEY)
                ctx = f"Metrics: {metrics}\nLatest Sentiment Z: {zscore:.2f}\n"
                resp = client.chat.completions.create(
                    model="google/gemini-2.0-flash-exp",
                    messages=[{"role": "user", "content": ctx + "\nQuestion: " + q}],
                    temperature=0.2,
                )
                st.markdown("**AI Response**")
                st.write(resp.choices[0].message.content)
            except Exception as e:
                st.warning(f"AI response failed: {e}")
        elif q:
            st.info("Add your OpenRouter API key to enable AI Q&A.")

if __name__ == "__main__":
    main()

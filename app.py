
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

# ------------------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------------------
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
    .risk-on { border-left-color:#2ecc71!important; }
    .risk-off { border-left-color:#e74c3c!important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📈 Project AlphaSent")
st.caption("*A Sentiment-Enhanced Framework for Systematic Cryptocurrency Allocation*")

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
class Config:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    CRYPTOCOMPARE_API_KEY = st.secrets.get("CRYPTOCOMPARE_API_KEY", "")
    DATA_URL = "https://raw.githubusercontent.com/planet0512/crypto/main/final_app_data.csv"
    SENTIMENT_THRESHOLDS = {"risk_on": 1.0, "risk_off": -1.0}
    TRANSACTION_COST = 0.0025  # 25 bps per side
    MAX_POSITION_SIZE = 0.30
    LOOKBACK_PERIOD = 90

@st.cache_resource
def setup_nltk():
    import nltk
    try:
        nltk.download("vader_lexicon", quiet=True)
        return True
    except Exception as e:
        st.error(f"Failed to download NLTK data: {e}")
        return False

# ------------------------------------------------------------------------------
# Requests + Data
# ------------------------------------------------------------------------------
@st.cache_data(ttl=300)
def get_sentiment_label(score: float) -> str:
    """
    Convert a VADER compound sentiment score into a human-readable label.
    """
    if score > 0.05:
        return "Positive"
    elif score < -0.05:
        return "Negative"
    else:
        return "Neutral"


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

        # Drop all-NaN columns and ensure numeric
        df = df.dropna(axis=1, how="all")
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan).dropna(how="all")
        return df
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=900)
def fetch_live_news(api_key: str) -> pd.DataFrame:
    """Fetch and analyze live crypto news. Safe for cache_data (no unhashables)."""
    if not api_key:
        return pd.DataFrame()

    session = create_requests_session()  # this is cache_resource'd
    url = f"https://min-api.cryptocompare.com/data/v2/news/?lang=EN&api_key={api_key}"

    try:
        r = session.get(url, timeout=30)
        r.raise_for_status()
        data = r.json().get("Data", [])
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data).head(20)

        # Sentiment
        analyzer = SentimentIntensityAnalyzer()
        df["compound"] = df["title"].fillna("").apply(lambda t: analyzer.polarity_scores(t)["compound"])
        df["sentiment_label"] = df["compound"].apply(get_sentiment_label)
        df["impact_score"] = df["compound"].abs()

        return df[["title", "source", "compound", "sentiment_label", "impact_score", "url"]]
    except Exception:
        return pd.DataFrame()



# ------------------------------------------------------------------------------
# Robust returns handling
# ------------------------------------------------------------------------------
def compute_returns_from_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect whether df looks like prices or returns and compute safe daily returns.
    Cleans inf/NaN and clips extreme negatives to avoid compounding blowups.
    """
    df = df.copy()

    looks_like_prices = (
        df.min(numeric_only=True).min() > 0
        and df.median(numeric_only=True).median() > 1
    )

    if looks_like_prices:
        rets = df.pct_change()
    else:
        rets = df

    rets = rets.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=-0.95)
    return rets


# ------------------------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------------------------
class PortfolioOptimizer:
    def __init__(self, transaction_cost: float = 0.0025, max_weight: float = 0.30):
        self.transaction_cost = transaction_cost
        self.max_weight = max_weight

    def clean_price_data(self, prices: pd.DataFrame) -> pd.DataFrame:
        min_obs = max(30, len(prices) // 10)
        valid_cols = prices.dropna().count()[lambda x: x >= min_obs].index
        if len(valid_cols) < 2:
            raise ValueError("Insufficient valid price data")
        cleaned = prices[valid_cols].ffill().dropna(how="all")
        returns = cleaned.pct_change().dropna()
        valid_assets = returns.std()[lambda x: x > 1e-6].index
        return cleaned[valid_assets]

    def get_optimized_weights(
        self,
        prices: pd.DataFrame,
        model: str = "max_sharpe",
        sentiment_scores: Optional[pd.Series] = None,
    ) -> Tuple[pd.Series, Dict]:

        try:
            clean_prices = self.clean_price_data(prices)
            if len(clean_prices.columns) < 2:
                return self._fallback_weights(prices.columns), {"method":"equal_weight","reason":"insufficient_assets"}

            mu = expected_returns.ema_historical_return(clean_prices, frequency=365)
            if sentiment_scores is not None and not sentiment_scores.empty:
                mu = self._adjust_returns_for_sentiment(mu, sentiment_scores)
            S = risk_models.CovarianceShrinkage(clean_prices).ledoit_wolf()

            ef = EfficientFrontier(mu, S, weight_bounds=(0, self.max_weight))
            ef.add_objective(L2_reg, gamma=0.01)

            if model == "max_sharpe":
                ef.max_sharpe()
            elif model == "min_variance":
                ef.min_volatility()
            elif model == "max_quadratic_utility":
                ef.max_quadratic_utility(risk_aversion=1.0)
            else:
                raise ValueError(f"Unknown optimization model: {model}")

            cleaned_weights = ef.clean_weights(cutoff=0.01)
            weights_series = pd.Series(cleaned_weights, dtype=float)

            # Re-normalize to be fully invested
            total = weights_series.sum()
            if total > 0:
                weights_series = weights_series / total

            port_ret = float(weights_series.dot(mu))
            port_vol = float(np.sqrt(weights_series.T @ S @ weights_series))
            sharpe = port_ret / port_vol if port_vol > 0 else 0.0

            metadata = {
                "method": model,
                "expected_return": port_ret,
                "volatility": port_vol,
                "sharpe_ratio": sharpe,
                "n_assets": int((weights_series > 0).sum()),
            }
            return weights_series.reindex(prices.columns, fill_value=0.0), metadata

        except (OptimizationError, ValueError, cp.error.SolverError) as e:
            st.warning(f"Optimization failed: {e}")
            return self._fallback_weights(prices.columns), {"method":"equal_weight","reason":str(e)}

    def _adjust_returns_for_sentiment(self, mu: pd.Series, scores: pd.Series) -> pd.Series:
        common = mu.index.intersection(scores.index)
        if len(common) == 0: return mu
        out = mu.copy()
        for a in common:
            factor = np.clip(scores[a] * 0.1, -0.2, 0.2)
            out[a] = out[a] * (1 + factor)
        return out

    def _fallback_weights(self, asset_names) -> pd.Series:
        return pd.Series(1.0 / max(1, len(asset_names)), index=asset_names, dtype=float)
    # --- Portfolio Analysis Tab ---

    def portfolio_recommendation_tab(strategy_returns: pd.Series):
        st.header("📊 Recommended Portfolio")
    
        if strategy_returns.empty:
            st.warning("No strategy return data available.")
            return
    
        # Annualized stats
        strat_ret_annual = strategy_returns.mean() * 252
        strat_vol_annual = strategy_returns.std() * np.sqrt(252)
        strat_sharpe = strat_ret_annual / strat_vol_annual if strat_vol_annual > 0 else 0
    
        st.subheader("Suggested Allocation")
        st.write("100% AlphaSent Strategy")
    
        st.metric("Expected Annual Return (%)", f"{strat_ret_annual*100:.2f}")
        st.metric("Annual Volatility (%)", f"{strat_vol_annual*100:.2f}")
        st.metric("Sharpe Ratio", f"{strat_sharpe:.2f}")

# ------------------------------------------------------------------------------
# Backtest
# ------------------------------------------------------------------------------
class BacktestEngine:
    def __init__(self, optimizer: PortfolioOptimizer, config: Config):
        self.optimizer = optimizer
        self.config = config

    def run_backtest(self, prices_df: pd.DataFrame, sentiment_data: pd.DataFrame):
        st.info("🚀 Initiating Enhanced Backtest Engine...")

        daily_returns = prices_df.pct_change().fillna(0)


        rebalance_dates = prices_df.resample("ME").last().index
        rebalance_dates = [d for d in rebalance_dates if d in prices_df.index]
        if len(rebalance_dates) < 3:
            st.error("Insufficient data for backtesting")
            return pd.Series(dtype=float), pd.DataFrame(), {}

        portfolio_returns = []
        allocation_history = []
        transaction_costs = []
        last_weights = pd.Series(dtype=float)

        sentiment_z = self._calculate_sentiment_zscore(sentiment_data)

        progress = st.progress(0)
        status = st.empty()

        for i, (cur, nxt) in enumerate(zip(rebalance_dates[:-1], rebalance_dates[1:])):
            progress.progress((i + 1) / (len(rebalance_dates) - 1))
            status.text(f"Processing {cur.strftime('%Y-%m')}")

            regime = self._get_regime(sentiment_z, cur)

            hist_start = cur - timedelta(days=self.config.LOOKBACK_PERIOD)
            hist_prices = prices_df.loc[hist_start:cur].dropna(how="all")
            if len(hist_prices) < 30:
                continue

            target_w, meta = self._optimize_for_regime(hist_prices, regime, sentiment_z)

            if not last_weights.empty:
                turnover = (target_w - last_weights.reindex(target_w.index, fill_value=0)).abs().sum()
                txn = turnover * self.config.TRANSACTION_COST / 2
                transaction_costs.append(float(txn))
            else:
                transaction_costs.append(0.0)

            period = self._calculate_period_returns(daily_returns, target_w, cur, nxt, transaction_costs[-1])
            if not period.empty:
                portfolio_returns.append(period)

            rec = target_w.to_dict()
            rec.update({"date": cur, "regime": regime, **meta, "transaction_cost": transaction_costs[-1]})
            allocation_history.append(rec)

            last_weights = target_w

        progress.empty()
        status.empty()

        if not portfolio_returns:
            st.error("No valid returns generated")
            return pd.Series(dtype=float), pd.DataFrame(), {}

        strategy_returns = pd.concat(portfolio_returns).astype(float)
        allocation_df = pd.DataFrame(allocation_history)

        metrics = self._calculate_performance_metrics(strategy_returns, prices_df, transaction_costs)

        st.success("✅ Backtest completed successfully!")
        return strategy_returns, allocation_df, metrics

    def _calculate_sentiment_zscore(self, sentiment_data: pd.DataFrame, window: int = 90) -> pd.Series:
        if "compound" not in sentiment_data.columns:
            return pd.Series(dtype=float)
        s = sentiment_data["compound"].fillna(0.0)
        mean = s.rolling(window=window, min_periods=30).mean()
        std = s.rolling(window=window, min_periods=30).std().replace(0, np.nan)
        return ((s - mean) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _get_regime(self, z: pd.Series, date) -> str:
        if z.empty: return "neutral"
        recent = z.loc[z.index <= date]
        if recent.empty: return "neutral"
        x = float(recent.iloc[-1])
        if x > self.config.SENTIMENT_THRESHOLDS["risk_on"]: return "risk_on"
        if x < self.config.SENTIMENT_THRESHOLDS["risk_off"]: return "risk_off"
        return "neutral"

    def _optimize_for_regime(self, prices: pd.DataFrame, regime: str, _z: pd.Series):
        if regime == "risk_on":
            w, meta = self.optimizer.get_optimized_weights(prices, model="max_sharpe")
        elif regime == "risk_off":
            w, meta = self.optimizer.get_optimized_weights(prices, model="min_variance")
        else:
            w, meta = self.optimizer.get_optimized_weights(prices, model="max_quadratic_utility")
        meta["regime"] = regime
        return w, meta

    def _calculate_period_returns(self, daily_returns, weights, start_date, end_date, txn_cost):
        period_returns = daily_returns.loc[start_date:end_date]
        if period_returns.empty:
            return pd.Series(dtype=float)

        common = weights.index.intersection(period_returns.columns)
        w = weights.reindex(common, fill_value=0.0).astype(float)
        r = period_returns[common].astype(float)
        r = r.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        p = (r * w).sum(axis=1)
        if not p.empty:
            p.iloc[0] -= float(txn_cost)
        return p.astype(float)

    def _calculate_performance_metrics(self, returns: pd.Series, prices_df: pd.DataFrame, transaction_costs: list) -> Dict:
        if returns.empty: return {}

        r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if r.empty: return {}

        n = len(r)
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        cum = np.prod(1.0 + returns.values)
        annualized_return = cum ** (365.0 / len(returns)) - 1.0 if cum > 0 else 0.0
        annualized_vol = float(np.nan_to_num(r.std(), nan=0.0)) * np.sqrt(365.0)

        ann_ret = cum ** (365.0 / len(returns)) - 1.0 if cum > 0 else 0.0
                
        ann_vol = float(np.nan_to_num(r.std(), nan=0.0)) * np.sqrt(365.0)
        sharpe = (annualized_return / annualized_vol) if (annualized_vol > 0 and np.isfinite(annualized_return)) else 0.0


        cum_curve = (1 + r).cumprod()
        roll_max = cum_curve.cummax()
        drawdown = (cum_curve - roll_max) / roll_max
        max_dd = float(drawdown.min())

        downside = r[r < 0]
        down_vol = (downside.std() * np.sqrt(365.0)) if len(downside) else 0.0
        sortino = (ann_ret / down_vol) if (down_vol > 0 and np.isfinite(ann_ret)) else 0.0

        bench = {}
        if "BTC" in prices_df.columns:
            btc = compute_returns_from_data(prices_df[["BTC"]])["BTC"]
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

        total_txn = float(sum(transaction_costs))
        avg_annual_txn = total_txn * (365.0 / n) if n > 0 else 0.0

        return {
            "total_return": cum - 1.0 if np.isfinite(cum) else np.nan,
            "annualized_return": ann_ret,
            "annualized_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_dd,
            "total_transaction_costs": total_txn,
            "avg_annual_transaction_cost": avg_annual_txn,
            **bench,
        }

# ------------------------------------------------------------------------------
# Charts
# ------------------------------------------------------------------------------
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

    # Monthly heatmap (clean labels)
    monthly = strategy_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)
    heat = monthly.to_frame("ret")
    heat["Year"] = heat.index.year.astype(str)  # Convert to string for categorical axis
    heat["Month"] = heat.index.strftime("%b")
    
    # Ensure months are ordered
    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
             "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    
    # Pivot table
    pivot = heat.pivot(index="Year", columns="Month", values="ret").reindex(columns=order)
    
    if not pivot.empty:
        fig.add_trace(
            go.Heatmap(
                z=pivot.values * 100.0,
                x=pivot.columns,
                y=pivot.index,  # Strings now → categorical axis
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

def sentiment_chart(sentiment_data: pd.DataFrame, strategy_returns: pd.Series):
    if "compound" not in sentiment_data.columns:
        return go.Figure()
    fig = make_subplots(rows=2, cols=1, subplot_titles=("Market Sentiment Z-Score", "Sentiment vs Strategy"), vertical_spacing=0.12, specs=[[{}],[{"secondary_y":True}]])
    s = sentiment_data["compound"].fillna(0.0)
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

# ------------------------------------------------------------------------------
# AI Summary
# ------------------------------------------------------------------------------
def generate_ai_summary(perf: Dict, latest_sentiment: float, latest_weights: pd.Series, regime_history: pd.DataFrame) -> str:
    if not Config.OPENROUTER_API_KEY:
        return "⚠️ Add your OpenRouter API key in `st.secrets` to enable AI insights."

    try:
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=Config.OPENROUTER_API_KEY)
        top = latest_weights.nlargest(5)
        reg_stats = regime_history["regime"].value_counts() if not regime_history.empty else pd.Series(dtype=float)
        prompt = f"""
Act as a quantitative crypto PM. Analyze this backtest:

Annual Return: {perf.get('annualized_return', 0):.2%}
Sharpe: {perf.get('sharpe_ratio', 0):.2f}
Max DD: {perf.get('max_drawdown', 0):.2%}
Volatility: {perf.get('annualized_volatility', 0):.2%}
Excess vs BTC: {perf.get('excess_return', 0):.2%}
Excess Sharpe vs BTC: {perf.get('excess_sharpe', 0):.2f}

Latest Sentiment Z: {latest_sentiment:.2f}
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

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def fmt_pct(x):
    return "—" if (x is None or not np.isfinite(x)) else f"{x:.1%}"

def fmt_num(x, digs=2):
    return "—" if (x is None or not np.isfinite(x)) else f"{x:.{digs}f}"

# --------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------

def portfolio_recommendation_tab(strategy_returns: pd.Series):
    st.header("📊 Recommended Portfolio")

    if strategy_returns.empty:
        st.warning("No strategy return data available.")
        return

    # Annualized stats
    strat_ret_annual = strategy_returns.mean() * 252
    strat_vol_annual = strategy_returns.std() * np.sqrt(252)
    strat_sharpe = strat_ret_annual / strat_vol_annual if strat_vol_annual > 0 else 0

    st.subheader("Suggested Allocation")
    st.write("100% AlphaSent Strategy")

    st.metric("Expected Annual Return (%)", f"{strat_ret_annual*100:.2f}")
    st.metric("Annual Volatility (%)", f"{strat_vol_annual*100:.2f}")
    st.metric("Sharpe Ratio", f"{strat_sharpe:.2f}")

# --------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------

def main():
    if not setup_nltk():
        st.stop()

    config = Config()
    session = create_requests_session()
    optimizer = PortfolioOptimizer(max_weight=config.MAX_POSITION_SIZE, transaction_cost=config.TRANSACTION_COST)
    engine = BacktestEngine(optimizer, config)

    # Sidebar
    with st.sidebar:
        st.subheader("🎛️ Control Center")
        col1, col2 = st.columns(2)
        col1.caption(("🟢" if config.CRYPTOCOMPARE_API_KEY else "🔴") + " CryptoCompare")
        col2.caption(("🟢" if config.OPENROUTER_API_KEY else "🔴") + " OpenRouter AI")
        st.divider()

        st.markdown("**Strategy Parameters**")
        t = st.slider("Sentiment Threshold", -2.0, 2.0, config.SENTIMENT_THRESHOLDS["risk_on"], 0.1)
        config.SENTIMENT_THRESHOLDS["risk_on"] = t
        config.SENTIMENT_THRESHOLDS["risk_off"] = -t
        config.MAX_POSITION_SIZE = st.slider("Max Position Size", 0.1, 0.5, config.MAX_POSITION_SIZE, 0.05)
        config.LOOKBACK_PERIOD = st.slider("Lookback (days)", 30, 180, config.LOOKBACK_PERIOD, 10)
        st.divider()

        st.markdown("**📰 Live News**")
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
        run = st.button("🚀 Run Enhanced Backtest Analysis", type="primary", use_container_width=True)

    if not run:
        st.stop()

    data = load_data(config.DATA_URL)
    if data.empty:
        st.error("Failed to load data.")
        st.stop()

    prices_df = data.drop(columns=["compound"], errors="ignore")
    sentiment_df = data[["compound"]].dropna()
    if sentiment_df.empty:
        st.error("No sentiment data available.")
        st.stop()

    with st.spinner("Running backtest..."):
        strategy_returns, allocation_history, metrics = engine.run_backtest(prices_df, sentiment_df)

    if strategy_returns.empty:
        st.error("Backtest failed.")
        st.stop()
    # Show Portfolio Recommendation before dashboards
    st.header("📊 Recommended Portfolio")
    
    if strategy_returns.empty:
        st.warning("Run the backtest to generate portfolio stats.")
    else:
        # Annualized stats
        strat_ret_annual = strategy_returns.mean() * 252
        strat_vol_annual = strategy_returns.std() * np.sqrt(252)
        strat_sharpe = strat_ret_annual / strat_vol_annual if strat_vol_annual > 0 else 0
    
        st.metric("Expected Annual Return (%)", f"{strat_ret_annual*100:.2f}")
        st.metric("Annual Volatility (%)", f"{strat_vol_annual*100:.2f}")
        st.metric("Sharpe Ratio", f"{strat_sharpe:.2f}")
    
        # Show latest allocation
        if not allocation_history.empty:
            st.subheader("Current Asset Allocation")
            latest_alloc = allocation_history.iloc[-1].select_dtypes(include=[np.number]) * 100

            st.bar_chart(latest_alloc)
    
    st.subheader("Performance Dashboard") 


    # Tabs
    tab0, tab1, tab2, tab3, tab4 = st.tabs(["Portfolio Analysis","📊 Performance", "🎯 Allocation", "📈 Sentiment", "🤖 AI Insights"])

    with tab0:
        portfolio_recommendation_tab(strategy_returns)
    with tab1:
        st.subheader("Performance Dashboard")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Annual Return", fmt_pct(metrics.get("annualized_return")))
        c2.metric("Sharpe Ratio", fmt_num(metrics.get("sharpe_ratio")))
        c3.metric("Max Drawdown", fmt_pct(metrics.get("max_drawdown")))
        c4.metric("Volatility", fmt_pct(metrics.get("annualized_volatility")))

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
                ["Total Transaction Costs", fmt_pct(metrics.get("total_transaction_costs"))],
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

        if allocation_history.empty:
            st.info("No allocation history.")
        else:
            latest = pd.Series(allocation_history.iloc[-1])
            non_meta = {"date","regime","transaction_cost","method","expected_return","volatility","sharpe_ratio","n_assets"}
            weights = (
                latest[~latest.index.isin(non_meta)]
                .apply(pd.to_numeric, errors="coerce")
                .dropna()
            )
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
            alloc_df = pd.DataFrame(allocation_history)
            alloc_df["date"] = pd.to_datetime(alloc_df["date"])
            alloc_df = alloc_df.set_index("date")

            asset_cols = [c for c in alloc_df.columns if c not in non_meta]
            for c in asset_cols:
                alloc_df[c] = pd.to_numeric(alloc_df[c], errors="coerce").fillna(0.0)

            if asset_cols:
                means = alloc_df[asset_cols].mean().nlargest(5).index
                area = go.Figure()
                for a in means:
                    area.add_trace(go.Scatter(x=alloc_df.index, y=alloc_df[a]*100, mode="lines", stackgroup="one", name=a))
                area.update_layout(yaxis_title="Allocation %", height=420, legend=dict(orientation="h"))
                st.plotly_chart(area, use_container_width=True)

            st.markdown("**Regime Analysis**")
            if "regime" in alloc_df.columns:
                counts = alloc_df["regime"].value_counts()
                if not counts.empty:
                    st.plotly_chart(px.pie(values=counts.values, names=counts.index, title="Time in Each Regime",
                                           color_discrete_map={"risk_on":"#22c55e","neutral":"#f59e0b","risk_off":"#ef4444"}),
                                    use_container_width=True)

    with tab3:
        st.subheader("Sentiment")
        latest_sent = sentiment_df["compound"].tail(7).mean()
        zscore = ((sentiment_df["compound"] - sentiment_df["compound"].rolling(90).mean()) /
                  sentiment_df["compound"].rolling(90).std()).replace([np.inf, -np.inf], np.nan).iloc[-1]
        zscore = float(zscore) if np.isfinite(zscore) else 0.0

        c1, c2, c3 = st.columns(3)
        c1.metric("Current Sentiment", f"{latest_sent:.2f}", help=get_sentiment_label(latest_sent))
        c2.metric("Sentiment Z-Score", f"{zscore:.2f}")
        regime_now = "Risk-On" if zscore > config.SENTIMENT_THRESHOLDS["risk_on"] else "Risk-Off" if zscore < config.SENTIMENT_THRESHOLDS["risk_off"] else "Neutral"
        c3.metric("Current Regime", regime_now)

        st.plotly_chart(sentiment_chart(sentiment_df, strategy_returns), use_container_width=True)

        stats = pd.DataFrame({
            "Metric":[
                "Mean Sentiment","Sentiment Volatility","Positive Days %","Negative Days %","Extreme Sentiment Days %","Current 30-Day Trend"
            ],
            "Value":[
                f"{sentiment_df['compound'].mean():.3f}",
                f"{sentiment_df['compound'].std():.3f}",
                f"{(sentiment_df['compound'] > 0.1).mean():.1%}",
                f"{(sentiment_df['compound'] < -0.1).mean():.1%}",
                f"{(abs(sentiment_df['compound']) > 0.5).mean():.1%}",
                f"{sentiment_df['compound'].tail(30).mean():.3f}",
            ],
        })
        st.dataframe(stats, hide_index=True, use_container_width=True)

    with tab4:
        st.subheader("🤖 AI-Powered Analysis")
        try:
            regime_df = pd.DataFrame(allocation_history)[["date","regime"]]
            regime_df["date"] = pd.to_datetime(regime_df["date"])
        except Exception:
            regime_df = pd.DataFrame(columns=["date","regime"])

        ai_text = generate_ai_summary(metrics, zscore, weights if "weights" in locals() else pd.Series(dtype=float), regime_df)
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

# ------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

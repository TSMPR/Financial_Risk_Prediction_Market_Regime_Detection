"""Leakage-aware time-series feature engineering and deterministic regime labels."""
from __future__ import annotations
import numpy as np
import pandas as pd

def build_features(prices: pd.DataFrame, volatility_window: int = 20, momentum_window: int = 20) -> pd.DataFrame:
    """Features at t use returns through t; labels are future risk regime at t+1."""
    returns = prices.pct_change()
    features = pd.DataFrame(index=prices.index)
    for col in prices.columns:
        r = returns[col]
        features[f"{col}_return_1d"] = r
        features[f"{col}_momentum"] = prices[col].pct_change(momentum_window)
        features[f"{col}_volatility"] = r.rolling(volatility_window, min_periods=volatility_window).std() * np.sqrt(252)
    # Broad market proxy from equal-weight asset returns.
    market_ret = returns.mean(axis=1)
    features["market_return_1d"] = market_ret
    features["market_momentum"] = (1 + market_ret).rolling(momentum_window).apply(np.prod, raw=True) - 1
    features["market_volatility"] = market_ret.rolling(volatility_window, min_periods=volatility_window).std() * np.sqrt(252)
    features["market_drawdown"] = (prices.div(prices.cummax()).sub(1)).mean(axis=1)
    return features.replace([np.inf, -np.inf], np.nan)

def make_regime_labels(prices: pd.DataFrame, volatility_window: int = 20,
                       momentum_window: int = 20, vol_quantile: float = 0.70) -> pd.Series:
    """Reproducible labels based on forward 20-day realized vol and current trend.

    HIGH_RISK when forward realized volatility exceeds the training-independent
    historical 70th percentile; otherwise BULLISH/BEARISH/SIDEWAYS by trailing
    momentum thresholds. This is a rule-defined target, not an externally observed truth.
    """
    market_ret = prices.pct_change().mean(axis=1)
    fwd_vol = market_ret.shift(-1).rolling(volatility_window).std().shift(-(volatility_window-1)) * np.sqrt(252)
    # Threshold uses expanding historical quantile at each date to avoid future-distribution leakage.
    trailing_vol = market_ret.rolling(volatility_window).std() * np.sqrt(252)
    threshold = trailing_vol.expanding(min_periods=max(60, volatility_window*2)).quantile(vol_quantile)
    momentum = (1 + market_ret).rolling(momentum_window).apply(np.prod, raw=True) - 1
    labels = pd.Series(index=prices.index, dtype="object")
    high = fwd_vol > threshold
    labels.loc[high.fillna(False)] = "HIGH_RISK"
    remaining = labels.isna() & momentum.notna()
    labels.loc[remaining & (momentum > 0.03)] = "BULLISH"
    labels.loc[remaining & (momentum < -0.03)] = "BEARISH"
    labels.loc[remaining & (momentum.between(-0.03, 0.03))] = "SIDEWAYS"
    return labels

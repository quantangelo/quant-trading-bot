from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RiskConfig, StrategyConfig

TRADING_DAYS = 252


def build_weights(closes: pd.DataFrame, strategy: StrategyConfig, risk: RiskConfig) -> pd.DataFrame:
    if strategy.name == "trend_volatility":
        weights = trend_volatility_weights(closes, strategy, risk)
    elif strategy.name == "dual_momentum":
        weights = dual_momentum_weights(closes, strategy, risk)
    else:
        raise ValueError(f"Unsupported strategy: {strategy.name}")
    return apply_regime_filter(weights, closes, strategy, risk)


def trend_volatility_weights(closes: pd.DataFrame, strategy: StrategyConfig, risk: RiskConfig) -> pd.DataFrame:
    if strategy.short_window >= strategy.long_window:
        raise ValueError("short_window must be smaller than long_window")

    returns = closes.pct_change()
    short_ma = closes.rolling(strategy.short_window).mean()
    long_ma = closes.rolling(strategy.long_window).mean()
    trend = (short_ma > long_ma).astype(float)

    annual_vol = returns.rolling(strategy.vol_window).std() * np.sqrt(TRADING_DAYS)
    raw = trend * (strategy.target_annual_vol / annual_vol.replace(0, np.nan))
    weights = cap_weights(raw, risk)

    rebalanced = weights.resample(strategy.rebalance_frequency).last().reindex(closes.index).ffill().fillna(0.0)
    return rebalanced.shift(1).fillna(0.0)


def dual_momentum_weights(closes: pd.DataFrame, strategy: StrategyConfig, risk: RiskConfig) -> pd.DataFrame:
    returns = closes.pct_change()
    momentum = closes.pct_change(strategy.momentum_window)
    annual_vol = returns.rolling(strategy.vol_window).std() * np.sqrt(TRADING_DAYS)
    positive = momentum.where(momentum > 0.0)
    ranks = positive.rank(axis=1, ascending=False, method="first")
    selected = ranks.le(max(strategy.top_n, 1)).astype(float)
    raw = allocate_selected_assets(returns, annual_vol, selected, strategy)
    raw *= min(strategy.target_annual_vol / 0.10, 1.0)
    weights = cap_weights(raw, risk)
    rebalanced = weights.resample(strategy.rebalance_frequency).last().reindex(closes.index).ffill().fillna(0.0)
    return rebalanced.shift(1).fillna(0.0)


def allocate_selected_assets(
    returns: pd.DataFrame,
    annual_vol: pd.DataFrame,
    selected: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    if strategy.allocation_method == "inverse_vol":
        scores = selected * (1.0 / annual_vol.replace(0, np.nan))
        return scores.div(scores.sum(axis=1), axis=0).fillna(0.0)
    if strategy.allocation_method != "correlation_adjusted":
        raise ValueError(f"Unsupported allocation_method: {strategy.allocation_method}")
    if len(returns) > 750:
        return _fast_correlation_adjusted_allocation(returns, annual_vol, selected, strategy)

    rows = []
    for date in returns.index:
        active = selected.columns[selected.loc[date].fillna(0.0).gt(0.0)]
        row = pd.Series(0.0, index=returns.columns, name=date)
        if len(active) == 0:
            rows.append(row)
            continue
        inv_vol = (1.0 / annual_vol.loc[date, active].replace(0, np.nan)).fillna(0.0)
        corr_window = returns.loc[:date, active].tail(strategy.correlation_window).corr().fillna(0.0).abs()
        avg_corr = corr_window.replace(1.0, np.nan).mean().fillna(0.0)
        penalty = 1.0 + strategy.correlation_penalty * avg_corr
        scores = inv_vol / penalty
        if scores.sum() > 0:
            row.loc[active] = scores / scores.sum()
        rows.append(row)
    return pd.DataFrame(rows).reindex(index=returns.index, columns=returns.columns).fillna(0.0)


def _fast_correlation_adjusted_allocation(
    returns: pd.DataFrame,
    annual_vol: pd.DataFrame,
    selected: pd.DataFrame,
    strategy: StrategyConfig,
) -> pd.DataFrame:
    avg_corr = returns.rolling(strategy.correlation_window).corr().abs().groupby(level=0).mean()
    avg_corr = avg_corr.reindex(index=returns.index, columns=returns.columns).fillna(0.0)
    inv_vol = 1.0 / annual_vol.replace(0, np.nan)
    scores = selected * inv_vol / (1.0 + strategy.correlation_penalty * avg_corr)
    return scores.div(scores.sum(axis=1), axis=0).fillna(0.0)


def cap_weights(raw: pd.DataFrame, risk: RiskConfig) -> pd.DataFrame:
    spendable = max(0.0, min(risk.max_gross_exposure, 1.0 - risk.min_cash))
    capped = raw.clip(lower=0.0, upper=risk.max_symbol_weight).fillna(0.0)
    gross = capped.abs().sum(axis=1)
    scale = (spendable / gross).clip(upper=1.0).fillna(0.0)
    return capped.mul(scale, axis=0)


def apply_regime_filter(
    weights: pd.DataFrame,
    closes: pd.DataFrame,
    strategy: StrategyConfig,
    risk: RiskConfig,
) -> pd.DataFrame:
    if not strategy.regime_filter:
        return weights
    symbol = strategy.regime_symbol or closes.columns[0]
    if symbol not in closes.columns:
        raise ValueError(f"Regime symbol {symbol} is not present in market data")
    regime_ma = closes[symbol].rolling(strategy.regime_window).mean()
    risk_on = closes[symbol] > regime_ma
    filtered = weights.where(risk_on, 0.0)
    if strategy.defensive_assets:
        defensive = defensive_weights(closes, strategy, risk)
        filtered.loc[~risk_on] = defensive.loc[~risk_on]
    elif strategy.regime_risk_off_weight > 0 and symbol in filtered.columns:
        risk_off_weight = min(strategy.regime_risk_off_weight, risk.max_symbol_weight, 1.0 - risk.min_cash)
        filtered.loc[~risk_on, symbol] = risk_off_weight
    return filtered.fillna(0.0)


def defensive_weights(closes: pd.DataFrame, strategy: StrategyConfig, risk: RiskConfig) -> pd.DataFrame:
    assets = [symbol for symbol in (strategy.defensive_assets or []) if symbol in closes.columns]
    if not assets:
        return pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    momentum = closes.loc[:, assets].pct_change(strategy.defensive_momentum_window)
    returns = closes.loc[:, assets].pct_change()
    annual_vol = returns.rolling(strategy.vol_window).std() * np.sqrt(TRADING_DAYS)
    ranks = momentum.rank(axis=1, ascending=False, method="first")
    selected = ranks.le(max(strategy.defensive_top_n, 1)).astype(float)
    selected = selected.where(momentum > 0.0, 0.0)
    weights = allocate_selected_assets(returns, annual_vol, selected, strategy)
    spendable = max(0.0, min(risk.max_gross_exposure, 1.0 - risk.min_cash))
    allocation = strategy.regime_risk_off_weight if strategy.regime_risk_off_weight > 0 else spendable
    weights *= min(spendable, allocation)
    all_weights = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    all_weights.loc[:, assets] = weights
    return cap_weights(all_weights, risk)

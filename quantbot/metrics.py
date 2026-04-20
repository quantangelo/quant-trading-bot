from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Performance:
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    value_at_risk_95: float
    conditional_value_at_risk_95: float
    win_rate: float
    trade_count: int

    def as_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def summarize(equity: pd.Series, returns: pd.Series, trade_count: int) -> Performance:
    equity = equity.dropna()
    returns = returns.dropna()
    if equity.empty:
        raise ValueError("Cannot summarize an empty equity curve")
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / TRADING_DAYS)
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_volatility = returns.std(ddof=0) * np.sqrt(TRADING_DAYS)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    downside = returns[returns < 0].std(ddof=0) * np.sqrt(TRADING_DAYS)
    sortino = annual_return / downside if downside > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    max_drawdown = abs(float(drawdown.min()))
    calmar = annual_return / max_drawdown if max_drawdown > 0 else 0.0
    value_at_risk_95 = abs(float(returns.quantile(0.05))) if not returns.empty else 0.0
    tail = returns[returns <= returns.quantile(0.05)]
    conditional_value_at_risk_95 = abs(float(tail.mean())) if not tail.empty else 0.0
    win_rate = float((returns > 0).mean()) if not returns.empty else 0.0
    return Performance(
        total_return=float(total_return),
        annual_return=float(annual_return),
        annual_volatility=float(annual_volatility),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_drawdown,
        calmar=float(calmar),
        value_at_risk_95=value_at_risk_95,
        conditional_value_at_risk_95=conditional_value_at_risk_95,
        win_rate=win_rate,
        trade_count=int(trade_count),
    )

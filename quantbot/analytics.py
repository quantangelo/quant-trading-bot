from __future__ import annotations

import pandas as pd

from .backtest import BacktestResult


def monthly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("ME").prod() - 1


def yearly_returns(returns: pd.Series) -> pd.Series:
    return (1 + returns).resample("YE").prod() - 1


def drawdown_periods(equity: pd.Series, top_n: int = 5) -> pd.DataFrame:
    drawdown = equity / equity.cummax() - 1
    periods = []
    in_drawdown = False
    start = None
    trough = None
    trough_depth = 0.0
    for date, value in drawdown.items():
        if value < 0 and not in_drawdown:
            in_drawdown = True
            start = date
            trough = date
            trough_depth = float(value)
        elif value < 0 and in_drawdown:
            if value < trough_depth:
                trough = date
                trough_depth = float(value)
        elif value == 0 and in_drawdown:
            periods.append({"start": start, "trough": trough, "end": date, "depth": abs(trough_depth)})
            in_drawdown = False
    if in_drawdown:
        periods.append({"start": start, "trough": trough, "end": pd.NaT, "depth": abs(trough_depth)})
    if not periods:
        return pd.DataFrame(columns=["start", "trough", "end", "depth", "days"])
    frame = pd.DataFrame(periods)
    frame["days"] = (frame["end"].fillna(equity.index[-1]) - frame["start"]).dt.days
    return frame.sort_values("depth", ascending=False).head(top_n)


def exposure_summary(weights: pd.DataFrame) -> pd.DataFrame:
    gross = weights.abs().sum(axis=1)
    net = weights.sum(axis=1)
    return pd.DataFrame(
        {
            "avg_gross": [gross.mean()],
            "max_gross": [gross.max()],
            "avg_net": [net.mean()],
            "max_net": [net.max()],
        }
    )


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(0.0)


def benchmark_correlation(result: BacktestResult) -> float:
    aligned = pd.concat([result.returns, result.benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0
    return float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))


def write_analytics(result: BacktestResult, out_dir: str) -> dict[str, str]:
    from pathlib import Path

    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    files = {
        "monthly_returns": base / "monthly_returns.csv",
        "yearly_returns": base / "yearly_returns.csv",
        "drawdowns": base / "drawdowns.csv",
        "exposure": base / "exposure.csv",
        "turnover": base / "turnover.csv",
    }
    monthly_returns(result.returns).to_csv(files["monthly_returns"], header=["return"])
    yearly_returns(result.returns).to_csv(files["yearly_returns"], header=["return"])
    drawdown_periods(result.equity).to_csv(files["drawdowns"], index=False)
    exposure_summary(result.weights).to_csv(files["exposure"], index=False)
    turnover(result.weights).to_csv(files["turnover"], header=["turnover"])
    return {name: str(path) for name, path in files.items()}

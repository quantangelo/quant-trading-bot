from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def simulate_return_paths(
    returns: pd.Series,
    initial_cash: float,
    simulations: int = 1000,
    seed: int = 42,
    block_size: int = 20,
) -> pd.DataFrame:
    clean = returns.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        raise ValueError("Cannot run Monte Carlo on empty returns")
    rng = np.random.default_rng(seed)
    paths = []
    for simulation in range(simulations):
        sampled = _sample_blocks(clean, len(clean), block_size, rng)
        equity = initial_cash * np.cumprod(1 + sampled)
        paths.append(
            {
                "simulation": simulation,
                "total_return": float(equity[-1] / initial_cash - 1),
                "annual_return": _annual_return(float(equity[-1] / initial_cash - 1), len(sampled)),
                "annual_volatility": float(np.std(sampled) * np.sqrt(TRADING_DAYS)),
                "max_drawdown": _max_drawdown(equity),
                "final_equity": float(equity[-1]),
            }
        )
    return pd.DataFrame(paths)


def monte_carlo_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    metrics = ["total_return", "annual_return", "annual_volatility", "max_drawdown", "final_equity"]
    rows = []
    for metric in metrics:
        series = paths[metric]
        rows.append(
            {
                "metric": metric,
                "p05": float(series.quantile(0.05)),
                "p25": float(series.quantile(0.25)),
                "median": float(series.quantile(0.50)),
                "p75": float(series.quantile(0.75)),
                "p95": float(series.quantile(0.95)),
                "mean": float(series.mean()),
            }
        )
    rows.append(
        {
            "metric": "loss_probability",
            "p05": float((paths["total_return"] < 0).mean()),
            "p25": float((paths["total_return"] < 0).mean()),
            "median": float((paths["total_return"] < 0).mean()),
            "p75": float((paths["total_return"] < 0).mean()),
            "p95": float((paths["total_return"] < 0).mean()),
            "mean": float((paths["total_return"] < 0).mean()),
        }
    )
    return pd.DataFrame(rows)


def write_monte_carlo(paths: pd.DataFrame, summary: pd.DataFrame, out_dir: str) -> dict[str, str]:
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    paths_file = base / "monte_carlo_paths.csv"
    summary_file = base / "monte_carlo_summary.csv"
    paths.to_csv(paths_file, index=False)
    summary.to_csv(summary_file, index=False)
    return {"paths": str(paths_file), "summary": str(summary_file)}


def _sample_blocks(values: np.ndarray, length: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    if block_size <= 1 or block_size >= length:
        return rng.choice(values, size=length, replace=True)
    chunks = []
    while sum(len(chunk) for chunk in chunks) < length:
        start = int(rng.integers(0, length - block_size + 1))
        chunks.append(values[start : start + block_size])
    return np.concatenate(chunks)[:length]


def _annual_return(total_return: float, periods: int) -> float:
    years = max(periods / TRADING_DAYS, 1 / TRADING_DAYS)
    return float((1 + total_return) ** (1 / years) - 1)


def _max_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity / peaks - 1
    return abs(float(np.min(drawdowns)))

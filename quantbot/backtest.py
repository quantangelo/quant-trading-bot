from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .config import BotConfig, with_strategy_params
from .data import close_matrix, field_matrix
from .metrics import Performance, summarize
from .strategy import build_weights


@dataclass(frozen=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    trades: pd.DataFrame
    performance: Performance
    benchmark_symbol: str
    benchmark_equity: pd.Series
    benchmark_returns: pd.Series
    benchmark_performance: Performance
    passed_validation: bool
    validation_messages: list[str]


def run_backtest(data: dict[str, pd.DataFrame], config: BotConfig) -> BacktestResult:
    closes = close_matrix(data)
    volumes = field_matrix(data, "volume").reindex(closes.index).reindex(columns=closes.columns).fillna(0.0)
    execution_prices = _execution_prices(data, config).reindex(closes.index).reindex(columns=closes.columns).ffill()
    target_weights = build_weights(closes, config.strategy, config.risk)
    asset_returns = closes.pct_change().fillna(0.0)

    equity_values: list[float] = []
    daily_returns: list[float] = []
    actual_weights = []
    trades = []
    equity = config.initial_cash
    previous_weights = pd.Series(0.0, index=closes.columns)
    halted = False
    peak = equity

    for date in closes.index:
        desired = target_weights.loc[date].copy()
        if halted:
            desired[:] = 0.0
        desired = _apply_volume_caps(
            desired,
            previous_weights,
            equity,
            execution_prices.loc[date],
            volumes.loc[date],
            config.costs.volume_limit_pct,
        )

        turnover = (desired - previous_weights).abs().sum()
        cost_rate = _estimate_cost_rate(turnover, desired, previous_weights, equity, config)
        gross_return = float((previous_weights * asset_returns.loc[date]).sum())
        net_return = gross_return - cost_rate
        equity *= 1 + net_return

        peak = max(peak, equity)
        drawdown = 1 - equity / peak
        if drawdown >= config.risk.max_drawdown:
            halted = True

        if turnover > 1e-9:
            trade_row = (desired - previous_weights).rename("weight_delta").reset_index()
            trade_row.columns = ["symbol", "weight_delta"]
            trade_row["date"] = date
            trade_row["equity"] = equity
            trade_row["estimated_notional"] = trade_row["weight_delta"] * equity
            trade_row["side"] = trade_row["weight_delta"].map(lambda value: "BUY" if value > 0 else "SELL")
            trades.append(trade_row)

        equity_values.append(equity)
        daily_returns.append(net_return)
        actual_weights.append(desired.rename(date))
        previous_weights = desired

    equity_series = pd.Series(equity_values, index=closes.index, name="equity")
    returns_series = pd.Series(daily_returns, index=closes.index, name="returns")
    weights = pd.DataFrame(actual_weights).reindex(columns=closes.columns).fillna(0.0)
    trade_frame = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(columns=["symbol", "weight_delta", "date", "equity"])
    trade_count = int((trade_frame["weight_delta"].abs() > 1e-9).sum()) if not trade_frame.empty else 0
    performance = summarize(equity_series, returns_series, trade_count)
    benchmark_symbol, benchmark_returns = _benchmark_returns(closes, config)
    benchmark_equity = config.initial_cash * (1 + benchmark_returns).cumprod()
    benchmark_performance = summarize(benchmark_equity.rename("benchmark_equity"), benchmark_returns, 1)
    passed, messages = validate(performance, benchmark_performance, config)
    return BacktestResult(
        equity_series,
        returns_series,
        weights,
        trade_frame,
        performance,
        benchmark_symbol,
        benchmark_equity,
        benchmark_returns,
        benchmark_performance,
        passed,
        messages,
    )


def validate(performance: Performance, benchmark: Performance, config: BotConfig) -> tuple[bool, list[str]]:
    messages: list[str] = []
    if performance.sharpe < config.validation.min_sharpe:
        messages.append(f"Sharpe {performance.sharpe:.2f} is below gate {config.validation.min_sharpe:.2f}")
    if performance.max_drawdown > config.validation.max_drawdown:
        messages.append(f"Drawdown {performance.max_drawdown:.2%} exceeds gate {config.validation.max_drawdown:.2%}")
    if performance.trade_count < config.validation.min_trades:
        messages.append(f"Trade count {performance.trade_count} is below gate {config.validation.min_trades}")
    excess = performance.total_return - benchmark.total_return
    if excess < config.validation.min_benchmark_excess_return:
        messages.append(
            f"Benchmark excess return {excess:.2%} is below gate {config.validation.min_benchmark_excess_return:.2%}"
        )
    if config.validation.require_lower_drawdown_than_benchmark and performance.max_drawdown > benchmark.max_drawdown:
        messages.append(
            f"Drawdown {performance.max_drawdown:.2%} is worse than benchmark drawdown {benchmark.max_drawdown:.2%}"
        )
    return len(messages) == 0, messages


def _estimate_cost_rate(turnover: float, desired: pd.Series, previous: pd.Series, equity: float, config: BotConfig) -> float:
    variable_bps = config.costs.commission_bps + config.costs.slippage_bps + (config.costs.spread_bps / 2)
    variable_cost = turnover * variable_bps / 10_000
    changed_orders = int(((desired - previous).abs() > 1e-9).sum())
    minimum_cost = (changed_orders * config.costs.min_commission / equity) if equity > 0 else 0.0
    return variable_cost + minimum_cost


def _execution_prices(data: dict[str, pd.DataFrame], config: BotConfig) -> pd.DataFrame:
    field = config.costs.execution_price.lower()
    if field not in {"open", "close"}:
        raise ValueError("costs.execution_price must be 'open' or 'close'")
    return field_matrix(data, field)


def _benchmark_returns(closes: pd.DataFrame, config: BotConfig) -> tuple[str, pd.Series]:
    if config.benchmark.weights:
        missing = sorted(set(config.benchmark.weights) - set(closes.columns))
        if missing:
            raise ValueError(f"Benchmark symbols missing from market data: {missing}")
        weights = pd.Series(config.benchmark.weights, dtype=float)
        if weights.sum() <= 0:
            raise ValueError("Benchmark weights must sum to a positive value")
        weights = weights / weights.sum()
        returns = closes.loc[:, weights.index].pct_change().fillna(0.0).mul(weights, axis=1).sum(axis=1)
        label = "+".join(f"{weight:.0%}{symbol}" for symbol, weight in weights.items())
        return label, returns.rename("benchmark_returns")
    symbol = config.benchmark.symbol or closes.columns[0]
    if symbol not in closes.columns:
        raise ValueError(f"Benchmark symbol {symbol} is not present in market data")
    return symbol, closes[symbol].pct_change().fillna(0.0).rename("benchmark_returns")


def _apply_volume_caps(
    desired: pd.Series,
    previous: pd.Series,
    equity: float,
    prices: pd.Series,
    volumes: pd.Series,
    volume_limit_pct: float,
) -> pd.Series:
    if volume_limit_pct <= 0:
        return desired
    adjusted = desired.copy()
    for symbol in desired.index:
        price = float(prices.get(symbol, 0.0))
        volume = float(volumes.get(symbol, 0.0))
        if price <= 0 or volume <= 0 or equity <= 0:
            continue
        max_weight_delta = (volume * volume_limit_pct * price) / equity
        delta = float(desired[symbol] - previous.get(symbol, 0.0))
        if abs(delta) > max_weight_delta:
            adjusted[symbol] = float(previous.get(symbol, 0.0)) + (max_weight_delta if delta > 0 else -max_weight_delta)
    return adjusted


def optimize(
    data: dict[str, pd.DataFrame],
    config: BotConfig,
    short_windows: list[int],
    long_windows: list[int],
    vol_targets: list[float],
) -> pd.DataFrame:
    rows = []
    for short in short_windows:
        for long in long_windows:
            if short >= long:
                continue
            for vol_target in vol_targets:
                candidate = with_strategy_params(
                    config,
                    short_window=short,
                    long_window=long,
                    target_annual_vol=vol_target,
                )
                result = run_backtest(data, candidate)
                rows.append(
                    {
                        "short_window": short,
                        "long_window": long,
                        "strategy": candidate.strategy.name,
                        "target_annual_vol": vol_target,
                        "passed": result.passed_validation,
                        **result.performance.as_dict(),
                    }
                )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["passed", "sharpe", "calmar"], ascending=[False, False, False])


def parameter_stability(grid: pd.DataFrame) -> pd.DataFrame:
    if grid.empty:
        return pd.DataFrame()
    top = grid.sort_values("sharpe", ascending=False).head(max(1, len(grid) // 5))
    return pd.DataFrame(
        [
            {
                "candidates": len(grid),
                "pass_rate": float(grid["passed"].mean()),
                "median_sharpe": float(grid["sharpe"].median()),
                "sharpe_std": float(grid["sharpe"].std(ddof=0)),
                "median_drawdown": float(grid["max_drawdown"].median()),
                "top_sharpe_min": float(top["sharpe"].min()),
                "top_sharpe_max": float(top["sharpe"].max()),
            }
        ]
    )


def select_stable_candidate(grid: pd.DataFrame, neighbor_radius: float = 1.0) -> pd.Series:
    if grid.empty:
        raise ValueError("Cannot select from an empty optimization grid")
    scored = grid.copy()
    scored["cluster_score"] = scored.apply(lambda row: _cluster_score(row, grid, neighbor_radius), axis=1)
    scored["risk_adjusted_score"] = (
        scored["cluster_score"]
        + scored["sharpe"].fillna(0.0)
        + scored["calmar"].fillna(0.0) * 0.25
        - scored["max_drawdown"].fillna(0.0)
    )
    return scored.sort_values(["passed", "risk_adjusted_score", "cluster_score", "sharpe"], ascending=[False, False, False, False]).iloc[0]


def _cluster_score(row: pd.Series, grid: pd.DataFrame, neighbor_radius: float) -> float:
    numeric = ["short_window", "long_window", "target_annual_vol"]
    distances = pd.Series(0.0, index=grid.index)
    for column in numeric:
        scale = max(float(grid[column].max() - grid[column].min()), 1.0)
        distances += ((grid[column] - row[column]) / scale) ** 2
    neighbors = grid.loc[distances.pow(0.5) <= neighbor_radius]
    if neighbors.empty:
        neighbors = grid
    if len(neighbors) == 1 and len(grid) > 1:
        return -1_000_000.0
    return float(
        neighbors["sharpe"].median()
        + neighbors["calmar"].median() * 0.25
        - neighbors["max_drawdown"].median()
        + neighbors["passed"].mean() * 0.25
    )


def walk_forward(data: dict[str, pd.DataFrame], config: BotConfig, train_years: int = 3, test_years: int = 1) -> pd.DataFrame:
    return walk_forward_optimized(
        data,
        config,
        train_years=train_years,
        test_years=test_years,
        short_windows=[config.strategy.short_window],
        long_windows=[config.strategy.long_window],
        vol_targets=[config.strategy.target_annual_vol],
        max_windows=None,
    )


def walk_forward_optimized(
    data: dict[str, pd.DataFrame],
    config: BotConfig,
    train_years: int,
    test_years: int,
    short_windows: list[int],
    long_windows: list[int],
    vol_targets: list[float],
    max_windows: int | None = None,
    selection_method: str = "best",
) -> pd.DataFrame:
    closes = close_matrix(data)
    start = closes.index.min()
    end = closes.index.max()
    rows = []
    cursor = start
    while cursor + pd.DateOffset(years=train_years + test_years) <= end:
        if max_windows is not None and len(rows) >= max_windows:
            break
        train_end = cursor + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(years=test_years)
        train_data = {symbol: frame.loc[cursor:train_end] for symbol, frame in data.items()}
        test_data = {symbol: frame.loc[train_end:test_end] for symbol, frame in data.items()}
        grid = optimize(
            train_data,
            config,
            short_windows=short_windows,
            long_windows=long_windows,
            vol_targets=vol_targets,
        )
        if grid.empty:
            break
        if selection_method == "best":
            best = grid.iloc[0]
        elif selection_method == "stable":
            best = select_stable_candidate(grid)
        else:
            raise ValueError("selection_method must be 'best' or 'stable'")
        candidate = with_strategy_params(
            config,
            short_window=int(best["short_window"]),
            long_window=int(best["long_window"]),
            target_annual_vol=float(best["target_annual_vol"]),
        )
        result = run_backtest(test_data, candidate)
        rows.append(
            {
                "train_start": cursor.date().isoformat(),
                "train_end": train_end.date().isoformat(),
                "test_end": test_end.date().isoformat(),
                "selected_short_window": int(best["short_window"]),
                "selected_long_window": int(best["long_window"]),
                "selected_target_annual_vol": float(best["target_annual_vol"]),
                "train_sharpe": float(best["sharpe"]),
                "train_cluster_score": float(best.get("cluster_score", 0.0)),
                "selection_method": selection_method,
                "train_passed": bool(best["passed"]),
                **result.performance.as_dict(),
                "passed": result.passed_validation,
            }
        )
        cursor = cursor + pd.DateOffset(years=test_years)
    return pd.DataFrame(rows)

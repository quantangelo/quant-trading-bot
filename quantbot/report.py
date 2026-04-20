from __future__ import annotations

from pathlib import Path

import pandas as pd

from .backtest import BacktestResult
from .config import BotConfig
from .metrics import Performance
from .analytics import benchmark_correlation, drawdown_periods, exposure_summary, turnover


def write_markdown_report(
    result: BacktestResult,
    config: BotConfig,
    output: str,
    monte_carlo: pd.DataFrame | None = None,
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quant Bot Report",
        "",
        f"Strategy: `{config.strategy.name}`",
        f"Symbols: `{', '.join(config.data.symbols)}`",
        f"Benchmark: buy-and-hold `{result.benchmark_symbol}`",
        f"Validation: `{'PASS' if result.passed_validation else 'FAIL'}`",
        "",
        "## Performance",
        "",
        "| Metric | Strategy | Benchmark |",
        "| --- | ---: | ---: |",
    ]
    lines.extend(_performance_rows(result.performance, result.benchmark_performance))
    exposure = exposure_summary(result.weights).iloc[0]
    avg_turnover = turnover(result.weights).mean()
    lines.extend(
        [
            "",
            "## Portfolio Diagnostics",
            "",
            f"- Benchmark correlation: {benchmark_correlation(result):.2f}",
            f"- Average gross exposure: {exposure['avg_gross']:.2%}",
            f"- Max gross exposure: {exposure['max_gross']:.2%}",
            f"- Average daily turnover: {avg_turnover:.2%}",
            "",
            "## Worst Drawdowns",
            "",
            "| Start | Trough | End | Depth | Days |",
            "| --- | --- | --- | ---: | ---: |",
        ]
    )
    drawdowns = drawdown_periods(result.equity)
    if drawdowns.empty:
        lines.append("| None | None | None | 0.00% | 0 |")
    else:
        for row in drawdowns.itertuples(index=False):
            end = "" if pd_isna(row.end) else row.end.date().isoformat()
            lines.append(
                f"| {row.start.date().isoformat()} | {row.trough.date().isoformat()} | {end} | {row.depth:.2%} | {int(row.days)} |"
            )
    lines.extend(
        [
            "",
            "## Validation Messages",
            "",
        ]
    )
    if result.validation_messages:
        lines.extend(f"- {message}" for message in result.validation_messages)
    else:
        lines.append("- None")
    if monte_carlo is not None and not monte_carlo.empty:
        lines.extend(
            [
                "",
                "## Monte Carlo Robustness",
                "",
                "| Metric | P05 | Median | P95 | Mean |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in monte_carlo.itertuples(index=False):
            lines.append(
                f"| {row.metric} | {_format_mc(row.metric, row.p05)} | {_format_mc(row.metric, row.median)} | {_format_mc(row.metric, row.p95)} | {_format_mc(row.metric, row.mean)} |"
            )
    lines.extend(
        [
            "",
            "## Risk Settings",
            "",
            f"- Max symbol weight: {config.risk.max_symbol_weight:.2%}",
            f"- Max gross exposure: {config.risk.max_gross_exposure:.2%}",
            f"- Min cash reserve: {config.risk.min_cash:.2%}",
            f"- Max drawdown halt: {config.risk.max_drawdown:.2%}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _performance_rows(strategy: Performance, benchmark: Performance) -> list[str]:
    rows = [
        ("Total return", _pct(strategy.total_return), _pct(benchmark.total_return)),
        ("Annual return", _pct(strategy.annual_return), _pct(benchmark.annual_return)),
        ("Annual volatility", _pct(strategy.annual_volatility), _pct(benchmark.annual_volatility)),
        ("Sharpe", f"{strategy.sharpe:.2f}", f"{benchmark.sharpe:.2f}"),
        ("Sortino", f"{strategy.sortino:.2f}", f"{benchmark.sortino:.2f}"),
        ("Max drawdown", _pct(strategy.max_drawdown), _pct(benchmark.max_drawdown)),
        ("Daily VaR 95", _pct(strategy.value_at_risk_95), _pct(benchmark.value_at_risk_95)),
        ("Daily CVaR 95", _pct(strategy.conditional_value_at_risk_95), _pct(benchmark.conditional_value_at_risk_95)),
        ("Win rate", _pct(strategy.win_rate), _pct(benchmark.win_rate)),
        ("Trades", str(strategy.trade_count), str(benchmark.trade_count)),
    ]
    return [f"| {name} | {left} | {right} |" for name, left, right in rows]


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _format_mc(metric: str, value: float) -> str:
    if metric == "final_equity":
        return f"{value:,.2f}"
    return _pct(value)


def pd_isna(value) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return value != value
    return bool(pd.isna(value))

from __future__ import annotations

from pathlib import Path

from .backtest import BacktestResult


def write_plots(result: BacktestResult, out_dir: str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to generate plots") from exc

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    equity_path = output / "equity_curve.png"
    ax = result.equity.plot(title="Equity Curve", figsize=(10, 5))
    result.benchmark_equity.plot(ax=ax, label=f"Benchmark {result.benchmark_symbol}")
    ax.set_ylabel("Account Value")
    ax.legend()
    ax.figure.tight_layout()
    ax.figure.savefig(equity_path)
    plt.close(ax.figure)
    paths.append(equity_path)

    drawdown_path = output / "drawdown.png"
    drawdown = result.equity / result.equity.cummax() - 1
    ax = drawdown.plot(title="Drawdown", figsize=(10, 4))
    ax.set_ylabel("Drawdown")
    ax.figure.tight_layout()
    ax.figure.savefig(drawdown_path)
    plt.close(ax.figure)
    paths.append(drawdown_path)

    exposure_path = output / "exposure.png"
    ax = result.weights.abs().sum(axis=1).plot(title="Gross Exposure", figsize=(10, 4))
    ax.set_ylabel("Gross Exposure")
    ax.figure.tight_layout()
    ax.figure.savefig(exposure_path)
    plt.close(ax.figure)
    paths.append(exposure_path)
    return paths

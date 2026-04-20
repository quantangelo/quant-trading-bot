from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    source: str
    path: str
    symbols: list[str]
    start: str | None = None
    end: str | None = None
    cache_path: str = "data/raw"


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    short_window: int
    long_window: int
    vol_window: int
    target_annual_vol: float
    rebalance_frequency: str
    momentum_window: int = 126
    top_n: int = 2
    regime_filter: bool = False
    regime_symbol: str | None = None
    regime_window: int = 200
    regime_risk_off_weight: float = 0.0
    defensive_assets: list[str] | None = None
    defensive_top_n: int = 1
    defensive_momentum_window: int = 63
    allocation_method: str = "inverse_vol"
    correlation_window: int = 63
    correlation_penalty: float = 1.0


@dataclass(frozen=True)
class RiskConfig:
    max_symbol_weight: float
    max_gross_exposure: float
    max_drawdown: float
    min_cash: float


@dataclass(frozen=True)
class CostConfig:
    commission_bps: float
    slippage_bps: float
    spread_bps: float = 0.0
    min_commission: float = 0.0
    volume_limit_pct: float = 0.10
    execution_price: str = "close"


@dataclass(frozen=True)
class ValidationConfig:
    min_sharpe: float
    max_drawdown: float
    min_trades: int
    min_benchmark_excess_return: float = 0.0
    require_lower_drawdown_than_benchmark: bool = False


@dataclass(frozen=True)
class BenchmarkConfig:
    symbol: str | None = None
    weights: dict[str, float] | None = None


@dataclass(frozen=True)
class BotConfig:
    initial_cash: float
    data: DataConfig
    strategy: StrategyConfig
    risk: RiskConfig
    costs: CostConfig
    validation: ValidationConfig
    benchmark: BenchmarkConfig = BenchmarkConfig()


def load_config(path: str | Path) -> BotConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return BotConfig(
        initial_cash=float(raw["initial_cash"]),
        data=DataConfig(**raw["data"]),
        strategy=StrategyConfig(**raw["strategy"]),
        risk=RiskConfig(**raw["risk"]),
        costs=CostConfig(**raw["costs"]),
        validation=ValidationConfig(**raw["validation"]),
        benchmark=BenchmarkConfig(**raw.get("benchmark", {})),
    )


def with_strategy_params(config: BotConfig, **params: Any) -> BotConfig:
    strategy = StrategyConfig(**{**config.strategy.__dict__, **params})
    return BotConfig(
        initial_cash=config.initial_cash,
        data=config.data,
        strategy=strategy,
        risk=config.risk,
        costs=config.costs,
        validation=config.validation,
        benchmark=config.benchmark,
    )

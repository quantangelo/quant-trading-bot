# Quant Trading Bot

A research-first quant trading bot scaffold with:

- CSV-first market data loading, plus optional Yahoo Finance download support.
- A trend-following strategy with volatility targeting.
- Portfolio-level risk controls, fees, slippage, max drawdown guard, and position caps.
- Backtesting, parameter search, walk-forward validation, and paper order generation.
- Standard-library tests via `unittest`.

No software can guarantee profit. This project is built to avoid trading unless a strategy passes explicit validation gates.

## Quick Start

Generate deterministic sample data:

```powershell
python -m quantbot.cli make-demo-data --out data/demo
```

Download and cache real Yahoo data:

```powershell
python -m quantbot.cli download --symbols SPY QQQ TLT GLD --start 2005-01-01 --cache data/raw
```

Or use the built-in real-data universe helper:

```powershell
python -m quantbot.cli prepare-real-data
```

Check whether the real-data cache is ready:

```powershell
python -m quantbot.cli check-real-data
```

Run a backtest:

```powershell
python -m quantbot.cli backtest --config configs/example.json
```

Run a small parameter search:

```powershell
python -m quantbot.cli optimize --config configs/example.json --short 20 40 --long 100 160 --vol-target 0.08 0.12
```

Score parameter stability:

```powershell
python -m quantbot.cli stability --config configs/example.json --short 20 40 60 --long 100 160 220 --vol-target 0.08 0.10 0.12
```

Select a stable parameter cluster:

```powershell
python -m quantbot.cli stability --config configs/example.json --short 20 40 60 --long 100 160 220 --vol-target 0.08 0.10 0.12 --select-stable
```

Run walk-forward validation:

```powershell
python -m quantbot.cli walk-forward --config configs/example.json
```

Run optimized walk-forward validation:

```powershell
python -m quantbot.cli walk-forward --config configs/real_data.json --short 20 40 60 --long 100 160 220 --vol-target 0.06 0.08 0.10 0.12 --selection stable
```

For a quick smoke test, add `--max-windows 2`.

`configs/real_data.json` uses a blended `60% SPY / 40% IEF` benchmark and rotates risk-off exposure into the strongest defensive asset among `IEF`, `TLT`, and `GLD`.
It also uses correlation-adjusted allocation to reduce exposure to assets that are moving too similarly.

Run the alternate dual-momentum strategy:

```powershell
python -m quantbot.cli backtest --config configs/dual_momentum.json
```

Write a benchmark comparison report:

```powershell
python -m quantbot.cli report --config configs/example.json
```

Write analytics tables and plots:

```powershell
python -m quantbot.cli analytics --config configs/example.json
python -m quantbot.cli plots --config configs/example.json
python -m quantbot.cli dashboard --config configs/example.json
```

Check data quality and run Monte Carlo robustness:

```powershell
python -m quantbot.cli quality --config configs/example.json
python -m quantbot.cli monte-carlo --config configs/example.json --simulations 1000 --block-size 20
python -m quantbot.cli report --config configs/example.json --monte-carlo
```

Scan news risk from RSS feeds or a local headline file:

```powershell
python -m quantbot.cli news-risk --config configs/real_data.json
python -m quantbot.cli news-risk --config configs/real_data.json --news-file headlines.txt
```

Block broker submission when news risk is high:

```powershell
python -m quantbot.cli paper --config configs/real_data.json --broker alpaca-paper --news-risk-check --news-risk-threshold 0.75
```

Create current paper orders from the latest bar:

```powershell
python -m quantbot.cli paper --config configs/example.json
```

Update a persistent paper account and submit to the dry-run broker:

```powershell
python -m quantbot.cli paper --config configs/example.json --state orders/paper_state.json --submit-dry-run
```

Submit paper orders to Alpaca:

```powershell
$env:ALPACA_API_KEY="your-paper-key"
$env:ALPACA_SECRET_KEY="your-paper-secret"
$env:ALPACA_PAPER="true"
python -m quantbot.cli paper --config configs/real_data.json --broker alpaca-paper --max-order-notional 5000
```

Alpaca submission is paper-only and refuses to run unless `ALPACA_PAPER=true`.

Run tests:

```powershell
python -m unittest discover -s tests
```

## Data Format

Place one CSV per symbol in your data directory, for example `data/demo/SPY.csv`.
Required columns:

```text
date,open,high,low,close,volume
```

Column names are case-insensitive. Dates should be parseable by pandas.

## Live Trading

The code intentionally ships with a dry-run/paper broker only. Add a real broker adapter only after:

1. The strategy passes out-of-sample and walk-forward validation.
2. You have configured risk limits appropriate to your account.
3. You have manually reviewed generated orders for a paper-trading period.

## Safety Defaults

- Signals are shifted by one bar to reduce look-ahead bias.
- Position sizing is volatility-targeted and capped.
- Backtests include commission and slippage assumptions.
- Optional spread, minimum commission, and volume participation limits make execution less naive.
- Trading halts when the configured max drawdown limit is breached.
- Validation fails when Sharpe, drawdown, or trade count gates are not met.
- Benchmark-relative validation gates can reject strategies that fail to beat buy-and-hold.

The demo data is synthetic and useful for proving that the bot works end to end.
Passing a synthetic backtest is not evidence that a strategy is ready for live capital.

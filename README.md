# Quant Trading Bot

Research-first quantitative trading bot for ETF rotation, backtesting, risk analysis, paper trading, Alpaca paper order submission, news-risk monitoring, and static dashboard reporting.

This project is designed to help answer one question before any capital is risked:

> Does this strategy still look acceptable after costs, drawdowns, walk-forward validation, benchmark comparison, data-quality checks, Monte Carlo stress testing, and paper-trading review?

It is not a promise of profitability. No trading bot can guarantee profits. The code intentionally emphasizes validation gates, audit files, dry-run behavior, paper trading, and conservative safety checks.

## What It Does

- Loads OHLCV market data from CSV files, cached Yahoo Finance files, or optional Yahoo downloads.
- Generates deterministic demo data for development and testing.
- Backtests ETF strategies with transaction costs, spread assumptions, minimum commissions, slippage, volume participation limits, and drawdown halts.
- Supports trend-following and dual-momentum ETF rotation.
- Supports risk-on/risk-off regime filtering.
- Supports defensive fallback assets during risk-off regimes.
- Supports inverse-volatility and correlation-adjusted allocation.
- Compares strategy performance against explicit or blended benchmarks.
- Runs parameter optimization, parameter-stability scoring, and walk-forward validation.
- Exports analytics, plots, Monte Carlo robustness results, markdown reports, and a professional static HTML dashboard.
- Generates local paper orders and maintains a local paper account state.
- Submits paper orders to Alpaca using paper credentials only.
- Splits Alpaca notional orders into child orders under a configurable max notional cap.
- Can sync Alpaca paper order status, positions, and account equity.
- Can rebalance from actual Alpaca paper positions to avoid duplicate buying on repeated runs.
- Scans news headlines/RSS feeds for macro risk themes and can block broker submission when news risk is high.
- Includes standard-library tests via `unittest`.

## Important Safety Notes

- This is research and automation software, not financial advice.
- The demo data is synthetic. Passing demo-data tests proves the bot works mechanically, not that the strategy has real edge.
- Always run real-data validation before paper trading.
- Always paper trade before using real capital.
- Alpaca integration is paper-only by design and refuses to run unless `ALPACA_PAPER=true`.
- Live trading is intentionally not implemented.
- Generated folders such as `data/`, `reports/`, and `orders/` are ignored by git.
- API keys must be stored in environment variables, never committed.

## Repository Layout

```text
configs/
  example.json             Demo trend-volatility config
  dual_momentum.json       Demo dual-momentum config
  real_data.json           Real-data ETF rotation config

quantbot/
  alpaca_broker.py         Alpaca paper broker adapter
  analytics.py             Monthly returns, drawdowns, exposure, turnover
  backtest.py              Backtest engine, validation, optimization, walk-forward
  broker.py                Local paper broker/order/account abstractions
  cli.py                   Command-line interface
  config.py                Dataclass config loader
  dashboard.py             Static HTML dashboard generator
  data.py                  CSV/Yahoo data loading and caching
  metrics.py               Performance metrics
  monte_carlo.py           Block-bootstrap Monte Carlo robustness
  news_risk.py             News headline risk overlay
  plotting.py              PNG chart generation with matplotlib
  quality.py               Market-data quality checks
  report.py                Markdown report generator
  strategy.py              Strategy and allocation logic

tests/
  test_backtest.py         Unit/integration tests
  fixtures/headlines.txt   Local news-risk test fixture
```

## Installation

From the project root:

```powershell
cd C:\Users\donat\OneDrive\Documents\quant-trading-bot
python -m pip install -r requirements.txt
```

Core dependencies:

- `pandas`
- `numpy`
- `matplotlib`
- `yfinance`
- `alpaca-py`

If you only want to run the synthetic demo and tests, Yahoo and Alpaca credentials are not required.

## Data

### CSV Format

The bot expects one CSV per symbol. Required columns:

```text
date,open,high,low,close,volume
```

Example:

```text
data/demo/SPY.csv
data/demo/TLT.csv
data/demo/GLD.csv
```

Column names are case-insensitive. Dates must be parseable by pandas.

### Demo Data

Generate deterministic synthetic ETF-like data:

```powershell
python -m quantbot.cli make-demo-data --out data/demo
```

This creates demo data for:

```text
SPY, TLT, GLD
```

### Real Yahoo Data

Download a custom set:

```powershell
python -m quantbot.cli download --symbols SPY QQQ TLT GLD IEF EFA EEM VNQ DBC --start 2005-01-01 --cache data/raw
```

Or use the built-in real-data universe helper:

```powershell
python -m quantbot.cli prepare-real-data
```

Check that the real-data cache is ready:

```powershell
python -m quantbot.cli check-real-data
```

The real-data config reads from:

```text
data/raw
```

## Config Files

### `configs/example.json`

Demo strategy using:

- CSV demo data
- `trend_volatility`
- `SPY`, `TLT`, `GLD`
- SPY benchmark
- validation gates relaxed enough for demo workflow

### `configs/dual_momentum.json`

Demo dual-momentum strategy using:

- CSV demo data
- `dual_momentum`
- regime filter
- defensive fallback into `TLT` / `GLD`
- correlation-adjusted allocation

### `configs/real_data.json`

Real-data ETF rotation strategy using:

- cached Yahoo data
- symbols: `SPY`, `QQQ`, `TLT`, `GLD`, `IEF`, `EFA`, `EEM`, `VNQ`, `DBC`
- blended benchmark: `60% SPY / 40% IEF`
- dual momentum
- SPY 200-day regime filter
- risk-off defensive assets: `IEF`, `TLT`, `GLD`
- correlation-adjusted allocation
- stricter validation:
  - minimum Sharpe
  - max drawdown limit
  - minimum trade count
  - benchmark outperformance gate
  - optional lower-drawdown-than-benchmark gate

## Strategies

### Trend Volatility

Configured with:

```json
"name": "trend_volatility"
```

Logic:

- Computes short and long moving averages.
- Takes long exposure when short moving average is above long moving average.
- Sizes exposure using rolling volatility and target annual volatility.
- Caps symbol weight and total gross exposure.
- Shifts weights by one bar to reduce look-ahead bias.

Key settings:

```json
"short_window": 40,
"long_window": 160,
"vol_window": 20,
"target_annual_vol": 0.10,
"rebalance_frequency": "W-FRI"
```

### Dual Momentum

Configured with:

```json
"name": "dual_momentum"
```

Logic:

- Computes momentum over `momentum_window`.
- Keeps only assets with positive momentum.
- Ranks assets by momentum.
- Selects the top `top_n`.
- Allocates using either inverse volatility or correlation-adjusted allocation.
- Applies risk caps.
- Shifts weights by one bar.

Key settings:

```json
"momentum_window": 126,
"top_n": 3
```

## Allocation

### Inverse Volatility

Configured with:

```json
"allocation_method": "inverse_vol"
```

Selected assets receive weights based on inverse rolling volatility.

### Correlation Adjusted

Configured with:

```json
"allocation_method": "correlation_adjusted",
"correlation_window": 63,
"correlation_penalty": 1.0
```

This starts with inverse volatility, then penalizes assets that are highly correlated with the rest of the selected basket. The goal is to avoid loading too heavily into assets that move similarly.

## Regime Filter and Defensive Fallback

The regime filter can reduce risk when a key market proxy is below trend.

Example from `configs/real_data.json`:

```json
"regime_filter": true,
"regime_symbol": "SPY",
"regime_window": 200
```

When `SPY` is below its 200-day moving average, the bot enters risk-off mode.

Risk-off mode can rotate into defensive assets:

```json
"defensive_assets": ["IEF", "TLT", "GLD"],
"defensive_top_n": 1,
"defensive_momentum_window": 63,
"regime_risk_off_weight": 0.75
```

That means the bot ranks defensive assets by momentum and allocates to the strongest qualifying defensive asset, capped by risk settings.

## Risk Controls

Configured under `risk`:

```json
"max_symbol_weight": 0.35,
"max_gross_exposure": 1.00,
"max_drawdown": 0.20,
"min_cash": 0.02
```

Meaning:

- No symbol should exceed `max_symbol_weight`.
- Gross exposure is capped.
- Cash reserve is preserved.
- If backtest drawdown breaches `max_drawdown`, the strategy halts to cash.

## Cost and Execution Model

Configured under `costs`:

```json
"commission_bps": 0.5,
"slippage_bps": 1.0,
"spread_bps": 0.5,
"min_commission": 0.0,
"volume_limit_pct": 0.05,
"execution_price": "close"
```

The backtester estimates:

- commissions
- slippage
- half-spread cost
- optional minimum commissions
- volume participation limits

`volume_limit_pct` prevents unrealistic position changes that would exceed a fraction of daily traded volume.

## Benchmarking

Single-symbol benchmark:

```json
"benchmark": {
  "symbol": "SPY"
}
```

Blended benchmark:

```json
"benchmark": {
  "weights": {
    "SPY": 0.6,
    "IEF": 0.4
  }
}
```

The benchmark is used in reports and validation gates.

## Backtesting

Run a backtest:

```powershell
python -m quantbot.cli backtest --config configs/example.json
```

Outputs:

```text
reports/equity.csv
reports/trades.csv
```

Printed metrics include:

- total return
- annual return
- annual volatility
- Sharpe
- Sortino
- max drawdown
- daily VaR 95
- daily CVaR 95
- win rate
- trade count
- benchmark return and Sharpe
- validation pass/fail

## Optimization and Stability

Run a parameter grid:

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

Stable selection penalizes isolated parameter winners. A single sharp result with poor neighboring support is treated as less trustworthy than a broader region of decent results.

## Walk-Forward Validation

Fixed-parameter walk-forward:

```powershell
python -m quantbot.cli walk-forward --config configs/example.json
```

Optimized walk-forward:

```powershell
python -m quantbot.cli walk-forward --config configs/real_data.json --short 20 40 60 --long 100 160 220 --vol-target 0.06 0.08 0.10 0.12 --selection stable
```

Quick smoke test:

```powershell
python -m quantbot.cli walk-forward --config configs/example.json --short 20 --long 80 --vol-target 0.08 0.10 --selection stable --max-windows 2
```

Outputs:

```text
reports/walk_forward.csv
```

Walk-forward output includes:

- train window
- test window
- selected parameters
- train Sharpe
- train cluster score
- selection method
- out-of-sample metrics
- validation pass/fail

## Data Quality Checks

Run:

```powershell
python -m quantbot.cli quality --config configs/real_data.json
```

Output:

```text
reports/data_quality.csv
```

Checks include:

- missing OHLCV columns
- unsorted dates
- duplicate dates
- null values
- non-positive prices
- non-positive volume
- invalid high/low relationships
- missing business days
- stale repeated closes
- large daily return outliers

## Monte Carlo Robustness

Run:

```powershell
python -m quantbot.cli monte-carlo --config configs/real_data.json --simulations 1000 --block-size 20
```

Outputs:

```text
reports/monte_carlo/monte_carlo_paths.csv
reports/monte_carlo/monte_carlo_summary.csv
```

The Monte Carlo module uses block-bootstrap sampling of strategy returns. This preserves some return clustering compared with a simple one-day shuffle.

Summary metrics include:

- total return percentiles
- annual return percentiles
- volatility percentiles
- max drawdown percentiles
- final equity percentiles
- loss probability

## Reports

Generate a markdown report:

```powershell
python -m quantbot.cli report --config configs/real_data.json --out reports/real_data_report.md
```

Include Monte Carlo:

```powershell
python -m quantbot.cli report --config configs/real_data.json --monte-carlo --out reports/real_data_report.md
```

The markdown report includes:

- strategy and universe
- benchmark
- validation status
- performance table
- portfolio diagnostics
- worst drawdowns
- validation messages
- risk settings
- optional Monte Carlo table

## Analytics and Plots

Export analytics tables:

```powershell
python -m quantbot.cli analytics --config configs/real_data.json
```

Outputs:

```text
reports/analytics/monthly_returns.csv
reports/analytics/yearly_returns.csv
reports/analytics/drawdowns.csv
reports/analytics/exposure.csv
reports/analytics/turnover.csv
```

Generate PNG plots:

```powershell
python -m quantbot.cli plots --config configs/real_data.json
```

Outputs:

```text
reports/plots/equity_curve.png
reports/plots/drawdown.png
reports/plots/exposure.png
```

## Dashboard

Generate the static HTML dashboard:

```powershell
python -m quantbot.cli dashboard --config configs/real_data.json --out reports/dashboard.html
```

Open:

```text
reports/dashboard.html
```

Dashboard sections:

- executive snapshot
- equity curve vs benchmark
- cumulative PnL
- drawdown chart
- monthly return bars
- gross exposure chart
- exposure summary
- turnover
- worst drawdowns
- recent trades
- optional news-risk table
- optional Monte Carlo table
- optional data-quality table

The dashboard is self-contained static HTML with embedded SVG charts. It does not require a local server.

## News Risk Overlay

The news-risk module scans headlines and maps themes to affected assets. It is intended as a risk/context overlay, not a standalone trading signal.

Run using default RSS feeds:

```powershell
python -m quantbot.cli news-risk --config configs/real_data.json
```

Run from a local headline file:

```powershell
python -m quantbot.cli news-risk --config configs/real_data.json --news-file headlines.txt
```

Outputs:

```text
reports/news_risk.csv
reports/news_risk.summary.json
```

Detected themes currently include:

- Middle East conflict
- oil supply shock
- inflation/rates
- banking/credit stress
- China growth risk

Example: headlines about conflict around Iran or the Strait of Hormuz can raise `middle_east_conflict` and `oil_supply_shock` risk, which maps to assets such as `DBC`, `GLD`, `SPY`, `QQQ`, `TLT`, and `IEF`.

Use news risk as a paper-trading gate:

```powershell
python -m quantbot.cli paper --config configs/real_data.json --broker alpaca-paper --news-risk-check --news-risk-threshold 0.75
```

If max news-risk score is above the threshold, the bot writes paper orders but blocks broker submission.

## Local Paper Trading

Generate target paper orders:

```powershell
python -m quantbot.cli paper --config configs/real_data.json
```

Output:

```text
orders/paper_orders.csv
```

Update local paper account state:

```powershell
python -m quantbot.cli paper --config configs/real_data.json --state orders/paper_state.json
```

Outputs:

```text
orders/paper_orders.csv
orders/paper_state.json
```

Dry-run broker mode:

```powershell
python -m quantbot.cli paper --config configs/real_data.json --broker dry-run
```

Dry-run prints accepted orders but does not contact a broker.

## Alpaca Paper Trading

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Set paper credentials:

```powershell
$env:ALPACA_API_KEY="your-paper-key"
$env:ALPACA_SECRET_KEY="your-paper-secret"
$env:ALPACA_PAPER="true"
```

Submit Alpaca paper orders:

```powershell
python -m quantbot.cli paper --config configs/real_data.json --broker alpaca-paper --max-order-notional 5000
```

Receipts:

```text
orders/alpaca_submissions.csv
```

Important behavior:

- Alpaca adapter uses `TradingClient(..., paper=True)`.
- It refuses to run without API keys.
- It refuses to run unless `ALPACA_PAPER=true`.
- Orders are notional market orders.
- Alpaca calculates filled quantity after execution.
- Large target orders are split into child orders under `--max-order-notional`.
- During non-US market hours, Alpaca may show orders as `accepted` with filled quantity `0.00` until the US market opens.

Sync Alpaca paper order status, positions, and account details:

```powershell
python -m quantbot.cli alpaca-orders
```

Outputs:

```text
orders/alpaca_order_status.csv
orders/alpaca_positions.csv
orders/alpaca_account.csv
```

Use actual Alpaca paper positions as the current portfolio before generating new rebalance orders:

```powershell
python -m quantbot.cli paper --config configs/real_data.json --use-alpaca-positions --broker alpaca-paper --max-order-notional 5000
```

This mode pulls current Alpaca paper positions and account equity, converts the positions into current weights, then generates only the trades needed to move from the current paper portfolio to the latest target weights. This is the safer command to use after the first paper-trading run because it helps prevent buying the same target allocation repeatedly.

## Suggested Real-Data Research Workflow

Run this before any paper trading:

```powershell
python -m pip install -r requirements.txt
python -m quantbot.cli prepare-real-data
python -m quantbot.cli check-real-data
python -m quantbot.cli quality --config configs/real_data.json
python -m quantbot.cli backtest --config configs/real_data.json
python -m quantbot.cli stability --config configs/real_data.json --short 20 40 60 --long 100 160 220 --vol-target 0.06 0.08 0.10 0.12 --select-stable
python -m quantbot.cli walk-forward --config configs/real_data.json --short 20 40 60 --long 100 160 220 --vol-target 0.06 0.08 0.10 0.12 --selection stable
python -m quantbot.cli monte-carlo --config configs/real_data.json --simulations 1000
python -m quantbot.cli news-risk --config configs/real_data.json
python -m quantbot.cli report --config configs/real_data.json --monte-carlo --out reports/real_data_report.md
python -m quantbot.cli dashboard --config configs/real_data.json --out reports/dashboard.html
```

Only consider paper trading if:

- data quality is acceptable
- backtest passes validation
- walk-forward results are acceptable
- benchmark-relative performance is acceptable
- Monte Carlo drawdown/loss ranges are tolerable
- news risk is not blocking submission
- generated paper orders look sensible

## Command Reference

```text
make-demo-data      Generate deterministic synthetic CSV data
download            Download Yahoo data for custom symbols
prepare-real-data   Download the built-in real-data ETF universe
check-real-data     Check whether cached real-data symbols exist
backtest            Run a backtest and write equity/trades
optimize            Run a parameter grid
stability           Score parameter stability and optionally select stable candidate
walk-forward        Run fixed or optimized walk-forward validation
paper               Generate paper orders, update local paper state, or submit to broker
alpaca-orders       Sync Alpaca paper orders, positions, and account snapshot
news-risk           Scan headlines/RSS feeds for macro risk themes
report              Write markdown performance report
quality             Run market-data quality checks
monte-carlo         Run block-bootstrap Monte Carlo robustness
analytics           Export analytics CSV files
plots               Generate PNG plots
dashboard           Generate static HTML dashboard
```

Use command-specific help:

```powershell
python -m quantbot.cli paper --help
python -m quantbot.cli walk-forward --help
python -m quantbot.cli news-risk --help
```

## Testing

Run all tests:

```powershell
python -m unittest discover -s tests
```

The test suite covers:

- demo backtest execution
- shifted signals
- risk caps and cash reserve
- dual momentum
- regime filter
- defensive fallback
- explicit and blended benchmarks
- correlation-adjusted allocation
- stable parameter selection
- walk-forward output
- local paper account persistence
- data quality checks
- Monte Carlo summaries
- Alpaca paper guardrails and order splitting
- Alpaca status and position normalization
- news-risk detection
- dashboard generation

## Git and Generated Files

The `.gitignore` excludes generated or sensitive runtime artifacts:

```text
data/
reports/
orders/
.env
__pycache__/
*.py[cod]
.pytest_cache/
.venv/
venv/
```

This keeps source code and configs in git while excluding downloaded market data, reports, paper order logs, account state, and credentials.

## Current Limitations

- No real live-trading adapter is implemented.
- Alpaca integration is paper-only.
- News-risk scoring is rule-based and should be treated as a conservative overlay, not a predictive model.
- Yahoo data can contain survivorship, adjustment, or data-quality issues; always run `quality`.
- The backtester is daily-bar oriented, not intraday.
- Fill simulation is simplified and cannot reproduce real market microstructure.
- Tax, margin, borrow costs, and account-specific constraints are not modeled.

## Roadmap Ideas

- Filled quantity reconciliation into a persistent local broker ledger.
- Scheduled paper-trading runs.
- Email/Telegram/Discord alerts.
- Asset-class exposure constraints.
- Per-symbol slippage/spread assumptions.
- More strategy families, such as mean reversion and volatility breakout.
- HTML dashboard links to generated CSV artifacts.
- Real broker adapter behind stricter live-trading kill switches.

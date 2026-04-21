from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd

from .analytics import benchmark_correlation, drawdown_periods, exposure_summary, monthly_returns, turnover
from .backtest import BacktestResult
from .config import BotConfig


def write_dashboard(
    result: BacktestResult,
    config: BotConfig,
    output: str,
    news_risk_path: str | None = "reports/news_risk.csv",
    monte_carlo_path: str | None = "reports/monte_carlo_smoke/monte_carlo_summary.csv",
    quality_path: str | None = "reports/data_quality.csv",
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _page(
            "Quant Bot Dashboard",
            "\n".join(
                [
                    _hero(result, config),
                    _metrics(result),
                    _chart_grid(result),
                    _portfolio_section(result),
                    _table_section("Recent Trades", result.trades.tail(20)),
                    _optional_table("News Risk", news_risk_path),
                    _optional_table("Monte Carlo Summary", monte_carlo_path),
                    _optional_table("Data Quality", quality_path),
                    "</main>",
                ]
            ),
        ),
        encoding="utf-8",
    )
    return path


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #07110f;
      --panel: #0d1b18;
      --panel-2: #112420;
      --line: #24443d;
      --ink: #effaf7;
      --muted: #90aaa3;
      --accent: #1bd6a3;
      --accent-2: #5ab8ff;
      --warn: #f6b44b;
      --bad: #ff6470;
      --good: #34d399;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      letter-spacing: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 20% 0%, rgba(27, 214, 163, 0.14), transparent 30%),
        linear-gradient(135deg, #07110f 0%, #0a151b 45%, #10130f 100%);
    }}
    header, main {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 24px;
    }}
    header {{
      padding-top: 34px;
      padding-bottom: 10px;
    }}
    h1 {{
      margin: 0;
      font-size: 38px;
      font-weight: 760;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 16px;
      text-transform: uppercase;
      color: #cbe5df;
    }}
    .subtle {{ color: var(--muted); }}
    .hero-row {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 16px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--line);
      padding-bottom: 18px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(17, 36, 32, 0.8);
      font-weight: 700;
      color: var(--accent);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
    }}
    .chart-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
    }}
    .wide {{ grid-column: 1 / -1; }}
    .card, .chart-card {{
      background: linear-gradient(180deg, rgba(17, 36, 32, 0.94), rgba(13, 27, 24, 0.98));
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .metric-value {{
      font-size: 26px;
      font-weight: 760;
      margin-top: 7px;
    }}
    .metric-value.good {{ color: var(--good); }}
    .metric-value.bad {{ color: var(--bad); }}
    section {{ margin: 18px 0; }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .axis {{ stroke: #2f554d; stroke-width: 1; }}
    .gridline {{ stroke: rgba(47, 85, 77, 0.55); stroke-width: 1; }}
    .line-strategy {{ fill: none; stroke: var(--accent); stroke-width: 2.4; }}
    .line-benchmark {{ fill: none; stroke: var(--accent-2); stroke-width: 1.8; opacity: 0.9; }}
    .area-drawdown {{ fill: rgba(255, 100, 112, 0.22); stroke: var(--bad); stroke-width: 1.8; }}
    .bar-pos {{ fill: var(--good); }}
    .bar-neg {{ fill: var(--bad); }}
    .bar-neutral {{ fill: var(--accent-2); }}
    .legend {{
      display: flex;
      gap: 14px;
      color: var(--muted);
      font-size: 12px;
      margin-top: 8px;
    }}
    .dot {{
      width: 9px;
      height: 9px;
      border-radius: 50%;
      display: inline-block;
      margin-right: 6px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    th, td {{
      text-align: left;
      padding: 9px 10px;
      border-bottom: 1px solid rgba(36, 68, 61, 0.65);
      vertical-align: top;
    }}
    th {{
      background: #132c27;
      color: #cbe5df;
    }}
    tr:hover td {{ background: rgba(27, 214, 163, 0.06); }}
    .note {{
      margin-top: 8px;
      font-size: 12px;
      color: var(--muted);
    }}
    @media (max-width: 820px) {{
      .chart-grid {{ grid-template-columns: 1fr; }}
      h1 {{ font-size: 30px; }}
    }}
  </style>
</head>
<body>
{body}
</body>
</html>
"""


def _hero(result: BacktestResult, config: BotConfig) -> str:
    validation = "PASS" if result.passed_validation else "FAIL"
    return f"""<header>
  <div class="hero-row">
    <div>
      <h1>Quant Bot Dashboard</h1>
      <div class="subtle">Strategy: {escape(config.strategy.name)} | Benchmark: {escape(result.benchmark_symbol)}</div>
      <div class="subtle">Universe: {escape(", ".join(config.data.symbols))}</div>
    </div>
    <div class="badge">Validation {validation}</div>
  </div>
</header>
<main>"""


def _metrics(result: BacktestResult) -> str:
    perf = result.performance
    items = [
        ("Total Return", _pct(perf.total_return), perf.total_return),
        ("Annual Return", _pct(perf.annual_return), perf.annual_return),
        ("Sharpe", f"{perf.sharpe:.2f}", perf.sharpe),
        ("Sortino", f"{perf.sortino:.2f}", perf.sortino),
        ("Max Drawdown", _pct(perf.max_drawdown), -perf.max_drawdown),
        ("Benchmark Return", _pct(result.benchmark_performance.total_return), result.benchmark_performance.total_return),
        ("Benchmark Corr.", f"{benchmark_correlation(result):.2f}", benchmark_correlation(result)),
        ("Trades", str(perf.trade_count), 0.0),
    ]
    cards = []
    for label, value, sign in items:
        tone = "good" if sign > 0 else "bad" if sign < 0 else ""
        cards.append(f'<div class="card"><div class="metric-label">{label}</div><div class="metric-value {tone}">{value}</div></div>')
    return f'<section><h2>Executive Snapshot</h2><div class="grid">{"".join(cards)}</div></section>'


def _chart_grid(result: BacktestResult) -> str:
    pnl = result.equity - result.equity.iloc[0]
    drawdown = result.equity / result.equity.cummax() - 1
    monthly = monthly_returns(result.returns).tail(36)
    exposure = result.weights.abs().sum(axis=1)
    return f"""<section class="chart-grid">
  <div class="chart-card wide">
    <h2>Equity Curve vs Benchmark</h2>
    {_line_chart({"Strategy": result.equity, "Benchmark": result.benchmark_equity})}
    <div class="legend"><span><span class="dot" style="background: var(--accent)"></span>Strategy</span><span><span class="dot" style="background: var(--accent-2)"></span>Benchmark</span></div>
  </div>
  <div class="chart-card">
    <h2>Cumulative PnL</h2>
    {_line_chart({"PnL": pnl}, strategy_only=True)}
    <div class="note">Account value less starting capital.</div>
  </div>
  <div class="chart-card">
    <h2>Drawdown</h2>
    {_area_chart(drawdown)}
    <div class="note">Peak-to-trough decline from strategy equity highs.</div>
  </div>
  <div class="chart-card">
    <h2>Monthly Returns</h2>
    {_bar_chart(monthly)}
    <div class="note">Last {len(monthly)} months.</div>
  </div>
  <div class="chart-card">
    <h2>Gross Exposure</h2>
    {_line_chart({"Exposure": exposure}, strategy_only=True)}
    <div class="note">Absolute portfolio exposure over time.</div>
  </div>
</section>"""


def _portfolio_section(result: BacktestResult) -> str:
    exposure = exposure_summary(result.weights)
    drawdowns = drawdown_periods(result.equity)
    turnover_frame = pd.DataFrame({"avg_daily_turnover": [turnover(result.weights).mean()]})
    return "\n".join(
        [
            _table_section("Exposure Summary", exposure),
            _table_section("Turnover", turnover_frame),
            _table_section("Worst Drawdowns", drawdowns),
        ]
    )


def _line_chart(series_map: dict[str, pd.Series], strategy_only: bool = False) -> str:
    width, height, pad = 920, 280, 28
    aligned = pd.concat(series_map.values(), axis=1).dropna()
    if aligned.empty:
        return '<div class="card subtle">No chart data.</div>'
    aligned.columns = list(series_map.keys())
    ymin = float(aligned.min().min())
    ymax = float(aligned.max().max())
    if ymin == ymax:
        ymax = ymin + 1
    paths = []
    colors = ["line-strategy"] if strategy_only else ["line-strategy", "line-benchmark"]
    for idx, column in enumerate(aligned.columns):
        points = []
        values = aligned[column].to_list()
        for i, value in enumerate(values):
            x = pad + i / max(len(values) - 1, 1) * (width - 2 * pad)
            y = height - pad - (float(value) - ymin) / (ymax - ymin) * (height - 2 * pad)
            points.append((x, y))
        d = " ".join(("M" if i == 0 else "L") + f"{x:.2f},{y:.2f}" for i, (x, y) in enumerate(points))
        paths.append(f'<path class="{colors[min(idx, len(colors)-1)]}" d="{d}" />')
    return _svg_frame(width, height, "\n".join(paths), _format_axis(ymin), _format_axis(ymax))


def _area_chart(series: pd.Series) -> str:
    width, height, pad = 920, 280, 28
    clean = series.dropna()
    if clean.empty:
        return '<div class="card subtle">No drawdown data.</div>'
    ymin, ymax = float(clean.min()), 0.0
    points = []
    for i, value in enumerate(clean):
        x = pad + i / max(len(clean) - 1, 1) * (width - 2 * pad)
        y = height - pad - (float(value) - ymin) / (ymax - ymin if ymax != ymin else 1) * (height - 2 * pad)
        points.append((x, y))
    zero_y = pad
    d = f"M{points[0][0]:.2f},{zero_y:.2f} " + " ".join(f"L{x:.2f},{y:.2f}" for x, y in points) + f" L{points[-1][0]:.2f},{zero_y:.2f} Z"
    return _svg_frame(width, height, f'<path class="area-drawdown" d="{d}" />', _pct(ymin), "0.00%")


def _bar_chart(series: pd.Series) -> str:
    width, height, pad = 920, 280, 28
    clean = series.dropna()
    if clean.empty:
        return '<div class="card subtle">No monthly return data.</div>'
    max_abs = max(float(clean.abs().max()), 0.001)
    zero_y = height / 2
    bar_w = (width - 2 * pad) / len(clean) * 0.72
    parts = [f'<line class="axis" x1="{pad}" y1="{zero_y:.2f}" x2="{width-pad}" y2="{zero_y:.2f}" />']
    for i, value in enumerate(clean):
        x = pad + i / len(clean) * (width - 2 * pad)
        bar_h = abs(float(value)) / max_abs * (height / 2 - pad)
        y = zero_y - bar_h if value >= 0 else zero_y
        klass = "bar-pos" if value >= 0 else "bar-neg"
        parts.append(f'<rect class="{klass}" x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" rx="2" />')
    return _svg_frame(width, height, "\n".join(parts), _pct(-max_abs), _pct(max_abs))


def _svg_frame(width: int, height: int, body: str, y_min_label: str, y_max_label: str) -> str:
    grid = "\n".join(
        f'<line class="gridline" x1="28" y1="{y}" x2="{width-28}" y2="{y}" />'
        for y in [28, height / 2, height - 28]
    )
    labels = f"""
      <text x="30" y="20" fill="#90aaa3" font-size="11">{escape(y_max_label)}</text>
      <text x="30" y="{height-8}" fill="#90aaa3" font-size="11">{escape(y_min_label)}</text>
    """
    return f'<svg viewBox="0 0 {width} {height}" role="img">{grid}{labels}{body}</svg>'


def _optional_table(title: str, path: str | None) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        return f'<section><h2>{escape(title)}</h2><div class="card subtle">No {escape(title.lower())} file found at {escape(path)}.</div></section>'
    try:
        frame = pd.read_csv(file_path)
    except Exception as exc:
        return f'<section><h2>{escape(title)}</h2><div class="card subtle">Could not read {escape(path)}: {escape(str(exc))}</div></section>'
    return _table_section(title, frame.head(20))


def _table_section(title: str, frame: pd.DataFrame) -> str:
    return f"<section><h2>{escape(title)}</h2>{_html_table(frame)}</section>"


def _html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<div class="card subtle">No rows.</div>'
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_format_cell)
    return safe.to_html(index=False, escape=False)


def _format_cell(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return escape(str(value))


def _pct(value: float) -> str:
    return f"{value:.2%}"


def _format_axis(value: float) -> str:
    if abs(value) < 3:
        return f"{value:.2f}"
    return f"{value:,.0f}"

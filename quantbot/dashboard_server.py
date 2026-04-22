from __future__ import annotations

from dataclasses import asdict
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from .alpaca_broker import AlpacaPaperBroker
from .backtest import run_backtest
from .broker import Order, PaperBroker
from .config import load_config
from .dashboard import write_dashboard
from .data import load_market_data
from .news_risk import fetch_rss_news, score_news, should_block_trading, write_news_risk_report
from .quality import check_market_data, quality_summary, write_quality_report
from .strategy import build_weights


class DashboardTradingApp:
    def __init__(
        self,
        config_path: str,
        host: str = "127.0.0.1",
        port: int = 8765,
        max_order_notional: float = 5_000.0,
        news_risk_threshold: float = 0.75,
        dashboard_out: str = "reports/dashboard.html",
        require_confirm_phrase: str = "PAPER",
    ) -> None:
        self.config_path = config_path
        self.host = host
        self.port = port
        self.max_order_notional = max_order_notional
        self.news_risk_threshold = news_risk_threshold
        self.dashboard_out = dashboard_out
        self.require_confirm_phrase = require_confirm_phrase

    def serve(self) -> None:
        handler = self._handler()
        server = ThreadingHTTPServer((self.host, self.port), handler)
        print(f"Dashboard trading server running at http://{self.host}:{self.port}")
        print("Press Ctrl+C to stop.")
        server.serve_forever()

    def _handler(self):
        app = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_html(app.render_home())
                elif parsed.path == "/dashboard":
                    self._send_html(app.render_dashboard())
                elif parsed.path == "/orders":
                    self._send_html(app.render_orders())
                else:
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")

            def do_POST(self) -> None:
                parsed = urlparse(self.path)
                if parsed.path != "/submit":
                    self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                    return
                length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(length).decode("utf-8")
                data = parse_qs(body)
                phrase = data.get("confirm", [""])[0]
                if phrase != app.require_confirm_phrase:
                    self._send_html(app.render_orders(message="Submission blocked: type PAPER exactly to confirm."))
                    return
                try:
                    message = app.submit_orders()
                except Exception as exc:
                    message = f"Submission failed: {exc}"
                self._send_html(app.render_orders(message=message))

            def log_message(self, format: str, *args) -> None:
                return

            def _send_html(self, content: str) -> None:
                encoded = content.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

        return Handler

    def render_home(self) -> str:
        account, positions = self._alpaca_snapshot()
        return _page(
            "Quant Bot Trading Desk",
            "\n".join(
                [
                    _top_nav(),
                    "<main>",
                    _status_cards(account, positions),
                    _actions(),
                    _warning(),
                    "</main>",
                ]
            ),
        )

    def render_dashboard(self) -> str:
        path = self._refresh_dashboard()
        body = path.read_text(encoding="utf-8")
        insert = _control_strip()
        return body.replace("<body>", f"<body>{insert}", 1)

    def render_orders(self, message: str | None = None) -> str:
        plan = self.build_order_plan()
        order_frame = pd.DataFrame([asdict(order) for order in plan["orders"]])
        return _page(
            "Order Preview",
            "\n".join(
                [
                    _top_nav(),
                    "<main>",
                    f'<section><h2>Order Preview</h2>{_message(message)}{_risk_block(plan)}</section>',
                    _orders_table(order_frame),
                    _submit_form(plan),
                    "</main>",
                ]
            ),
        )

    def build_order_plan(self) -> dict:
        config = load_config(self.config_path)
        data = load_market_data(config.data)
        closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in data.items()}).ffill().dropna()
        weights = build_weights(closes, config.strategy, config.risk)
        target = weights.iloc[-1]

        alpaca = AlpacaPaperBroker(max_order_notional=self.max_order_notional)
        previous, equity, positions, account = alpaca.current_weights(target.index)
        open_orders = alpaca.get_orders("open", 100)
        orders = PaperBroker(equity).orders_from_weights(previous, target)
        news_hits, news_summary = self._news_risk(config.data.symbols)
        quality = quality_summary(check_market_data(data))
        news_blocked = should_block_trading(news_summary, self.news_risk_threshold)
        return {
            "orders": orders,
            "target": target,
            "previous": previous,
            "equity": equity,
            "positions": positions,
            "account": account,
            "open_orders": open_orders,
            "news_summary": news_summary,
            "news_hits": news_hits,
            "quality": quality,
            "news_blocked": news_blocked,
            "blocked": news_blocked or bool(open_orders),
        }

    def submit_orders(self) -> str:
        plan = self.build_order_plan()
        if plan["blocked"]:
            if plan["open_orders"]:
                return "Submission blocked because Alpaca already has open paper orders."
            return "Submission blocked by news-risk threshold."
        orders: list[Order] = plan["orders"]
        if not orders:
            return "No rebalance orders required."
        alpaca = AlpacaPaperBroker(max_order_notional=self.max_order_notional)
        receipts = alpaca.submit_orders(orders)
        receipts_path = alpaca.write_receipts(receipts, "orders/alpaca_dashboard_submissions.csv")
        return f"Submitted {len(receipts)} Alpaca paper child orders. Receipts written to {receipts_path}."

    def _refresh_dashboard(self) -> Path:
        config = load_config(self.config_path)
        result = run_backtest(load_market_data(config.data), config)
        return write_dashboard(result, config, self.dashboard_out)

    def _alpaca_snapshot(self) -> tuple[dict, list[dict]]:
        alpaca = AlpacaPaperBroker(max_order_notional=self.max_order_notional)
        account = alpaca.get_account()
        positions = alpaca.get_positions()
        alpaca.write_position_snapshot(positions, account, "orders/alpaca_positions.csv", "orders/alpaca_account.csv")
        return account, positions

    def _news_risk(self, symbols: list[str]) -> tuple[list, dict]:
        items = fetch_rss_news(None, limit=50)
        hits, summary = score_news(items, symbols)
        write_news_risk_report(hits, summary, "reports/news_risk.csv")
        return hits, summary


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>{_css()}</style>
</head>
<body>{body}</body>
</html>"""


def _css() -> str:
    return """
    :root { color-scheme: dark; --bg:#07110f; --panel:#0e1c19; --line:#28483f; --ink:#effaf7; --muted:#91aaa4; --accent:#1bd6a3; --bad:#ff6470; --warn:#f6b44b; }
    * { box-sizing: border-box; }
    body { margin:0; font-family: Inter, Segoe UI, Arial, sans-serif; letter-spacing:0; color:var(--ink); background:linear-gradient(135deg,#07110f,#0a151b 52%,#11150f); }
    header, main { max-width:1280px; margin:0 auto; padding:24px; }
    header { display:flex; justify-content:space-between; align-items:center; gap:16px; border-bottom:1px solid var(--line); }
    a { color:var(--accent); text-decoration:none; font-weight:700; }
    nav { display:flex; gap:14px; flex-wrap:wrap; }
    h1 { margin:0; font-size:32px; }
    h2 { margin:0 0 14px; font-size:15px; text-transform:uppercase; color:#cce5df; }
    section { margin:18px 0; }
    .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:12px; }
    .card { background:linear-gradient(180deg,rgba(18,38,34,.96),rgba(13,27,24,.98)); border:1px solid var(--line); border-radius:8px; padding:16px; }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; }
    .value { margin-top:7px; font-size:24px; font-weight:760; }
    .warn { color:var(--warn); }
    .bad { color:var(--bad); }
    table { width:100%; border-collapse:collapse; font-size:13px; background:var(--panel); border:1px solid var(--line); border-radius:8px; overflow:hidden; }
    th, td { text-align:left; padding:9px 10px; border-bottom:1px solid rgba(40,72,63,.7); }
    th { background:#132c27; color:#cbe5df; }
    button, input { border-radius:6px; border:1px solid var(--line); background:#132c27; color:var(--ink); padding:10px 12px; font:inherit; }
    button { background:var(--accent); color:#04110e; font-weight:800; cursor:pointer; }
    button:disabled { opacity:.45; cursor:not-allowed; }
    form { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
    .message { margin:12px 0; padding:12px; border:1px solid var(--line); background:#10221e; border-radius:8px; }
    .danger { border-color:rgba(255,100,112,.8); }
    .strip { position:sticky; top:0; z-index:5; background:#07110f; border-bottom:1px solid #28483f; padding:10px 20px; display:flex; justify-content:center; gap:18px; }
    @media (max-width:720px) { header { align-items:flex-start; flex-direction:column; } h1 { font-size:26px; } }
    """


def _top_nav() -> str:
    return """<header><div><h1>Quant Bot Trading Desk</h1><div class="label">Local Alpaca paper controls</div></div><nav><a href="/">Status</a><a href="/dashboard">Dashboard</a><a href="/orders">Order Preview</a></nav></header>"""


def _control_strip() -> str:
    return '<div class="strip"><a href="/">Trading Desk</a><a href="/orders">Preview Paper Orders</a></div>'


def _status_cards(account: dict, positions: list[dict]) -> str:
    cards = [
        ("Equity", _money(account.get("equity", 0))),
        ("Cash", _money(account.get("cash", 0))),
        ("Buying Power", _money(account.get("buying_power", 0))),
        ("Open Positions", str(len(positions))),
    ]
    return '<section><h2>Alpaca Paper Account</h2><div class="grid">' + "".join(
        f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'
        for label, value in cards
    ) + "</div></section>"


def _actions() -> str:
    return """<section><h2>Actions</h2><div class="grid">
      <a class="card" href="/dashboard">Open performance dashboard</a>
      <a class="card" href="/orders">Preview rebalance orders</a>
    </div></section>"""


def _warning() -> str:
    return '<section><div class="card warn">Paper trading only. The submit button requires an explicit PAPER confirmation and uses Alpaca paper credentials.</div></section>'


def _message(message: str | None) -> str:
    if not message:
        return ""
    klass = "message danger" if "failed" in message.lower() or "blocked" in message.lower() else "message"
    return f'<div class="{klass}">{escape(message)}</div>'


def _risk_block(plan: dict) -> str:
    summary = plan["news_summary"]
    quality = plan["quality"]
    risk = float(summary.get("max_score", 0.0))
    status = "Blocked" if plan["news_blocked"] else "Clear"
    quality_text = "No data quality issues" if quality.empty else f"{len(quality)} data quality rows"
    return f"""<div class="grid">
      <div class="card"><div class="label">Account Equity</div><div class="value">{_money(plan["equity"])}</div></div>
      <div class="card"><div class="label">News Risk</div><div class="value {'bad' if plan['news_blocked'] else ''}">{escape(status)} {risk:.2f}</div></div>
      <div class="card"><div class="label">Data Quality</div><div class="value">{escape(quality_text)}</div></div>
      <div class="card"><div class="label">Target Orders</div><div class="value">{len(plan["orders"])}</div></div>
      <div class="card"><div class="label">Open Alpaca Orders</div><div class="value {'bad' if plan['open_orders'] else ''}">{len(plan["open_orders"])}</div></div>
    </div>"""


def _orders_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return '<section><h2>Orders</h2><div class="card">No orders required.</div></section>'
    safe = frame.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(lambda value: f"{value:.4f}" if isinstance(value, float) else escape(str(value)))
    return f"<section><h2>Orders</h2>{safe.to_html(index=False, escape=False)}</section>"


def _submit_form(plan: dict) -> str:
    disabled = "disabled" if plan["blocked"] or not plan["orders"] else ""
    return f"""<section><h2>Submit To Alpaca Paper</h2>
      <div class="card">
        <form method="post" action="/submit">
          <label>Type PAPER to confirm</label>
          <input name="confirm" autocomplete="off" placeholder="PAPER">
          <button type="submit" {disabled}>Submit Paper Orders</button>
        </form>
      </div>
    </section>"""


def _money(value) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return "$0.00"

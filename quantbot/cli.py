from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .alpaca_broker import AlpacaPaperBroker, split_order
from .backtest import optimize, parameter_stability, run_backtest, select_stable_candidate, walk_forward, walk_forward_optimized
from .analytics import write_analytics
from .broker import DryRunBroker, PaperAccount, PaperBroker
from .config import load_config
from .dashboard import write_dashboard
from .data import download_yahoo_to_cache, load_market_data, make_demo_data, missing_cached_symbols
from .monte_carlo import monte_carlo_summary, simulate_return_paths, write_monte_carlo
from .news_risk import fetch_rss_news, load_local_news, score_news, should_block_trading, write_news_risk_report
from .plotting import write_plots
from .quality import check_market_data, quality_summary, write_quality_report
from .report import write_markdown_report
from .strategy import build_weights


def main() -> None:
    parser = argparse.ArgumentParser(prog="quantbot")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("make-demo-data")
    demo.add_argument("--out", default="data/demo")

    download = sub.add_parser("download")
    download.add_argument("--symbols", nargs="+", required=True)
    download.add_argument("--start", default=None)
    download.add_argument("--end", default=None)
    download.add_argument("--cache", default="data/raw")

    prep = sub.add_parser("prepare-real-data")
    prep.add_argument("--start", default="2005-01-01")
    prep.add_argument("--end", default=None)
    prep.add_argument("--cache", default="data/raw")

    check_real = sub.add_parser("check-real-data")
    check_real.add_argument("--config", default="configs/real_data.json")

    backtest = sub.add_parser("backtest")
    backtest.add_argument("--config", required=True)
    backtest.add_argument("--equity-out", default="reports/equity.csv")
    backtest.add_argument("--trades-out", default="reports/trades.csv")

    opt = sub.add_parser("optimize")
    opt.add_argument("--config", required=True)
    opt.add_argument("--short", nargs="+", type=int, required=True)
    opt.add_argument("--long", nargs="+", type=int, required=True)
    opt.add_argument("--vol-target", nargs="+", type=float, required=True)
    opt.add_argument("--out", default="reports/optimization.csv")

    stability = sub.add_parser("stability")
    stability.add_argument("--config", required=True)
    stability.add_argument("--short", nargs="+", type=int, required=True)
    stability.add_argument("--long", nargs="+", type=int, required=True)
    stability.add_argument("--vol-target", nargs="+", type=float, required=True)
    stability.add_argument("--grid-out", default="reports/stability_grid.csv")
    stability.add_argument("--summary-out", default="reports/stability_summary.csv")
    stability.add_argument("--select-stable", action="store_true")

    wf = sub.add_parser("walk-forward")
    wf.add_argument("--config", required=True)
    wf.add_argument("--out", default="reports/walk_forward.csv")
    wf.add_argument("--train-years", type=int, default=3)
    wf.add_argument("--test-years", type=int, default=1)
    wf.add_argument("--short", nargs="+", type=int, default=None)
    wf.add_argument("--long", nargs="+", type=int, default=None)
    wf.add_argument("--vol-target", nargs="+", type=float, default=None)
    wf.add_argument("--max-windows", type=int, default=None)
    wf.add_argument("--selection", choices=["best", "stable"], default="best")

    paper = sub.add_parser("paper")
    paper.add_argument("--config", required=True)
    paper.add_argument("--previous-weights", default=None)
    paper.add_argument("--out", default="orders/paper_orders.csv")
    paper.add_argument("--state", default=None)
    paper.add_argument("--submit-dry-run", action="store_true")
    paper.add_argument("--broker", choices=["none", "dry-run", "alpaca-paper"], default="none")
    paper.add_argument("--alpaca-receipts-out", default="orders/alpaca_submissions.csv")
    paper.add_argument("--alpaca-positions-out", default="orders/alpaca_positions.csv")
    paper.add_argument("--alpaca-account-out", default="orders/alpaca_account.csv")
    paper.add_argument("--use-alpaca-positions", action="store_true")
    paper.add_argument("--max-order-notional", type=float, default=10_000.0)
    paper.add_argument("--news-risk-check", action="store_true")
    paper.add_argument("--news-file", default=None)
    paper.add_argument("--news-risk-threshold", type=float, default=0.75)
    paper.add_argument("--news-risk-out", default="reports/news_risk.csv")

    news = sub.add_parser("news-risk")
    news.add_argument("--config", required=True)
    news.add_argument("--news-file", default=None)
    news.add_argument("--rss", nargs="*", default=None)
    news.add_argument("--limit", type=int, default=50)
    news.add_argument("--out", default="reports/news_risk.csv")

    alpaca_orders = sub.add_parser("alpaca-orders")
    alpaca_orders.add_argument("--status", choices=["open", "closed", "all"], default="all")
    alpaca_orders.add_argument("--limit", type=int, default=100)
    alpaca_orders.add_argument("--orders-out", default="orders/alpaca_order_status.csv")
    alpaca_orders.add_argument("--positions-out", default="orders/alpaca_positions.csv")
    alpaca_orders.add_argument("--account-out", default="orders/alpaca_account.csv")

    report = sub.add_parser("report")
    report.add_argument("--config", required=True)
    report.add_argument("--out", default="reports/report.md")
    report.add_argument("--monte-carlo", action="store_true")
    report.add_argument("--mc-simulations", type=int, default=1000)
    report.add_argument("--mc-block-size", type=int, default=20)

    quality = sub.add_parser("quality")
    quality.add_argument("--config", required=True)
    quality.add_argument("--out", default="reports/data_quality.csv")

    mc = sub.add_parser("monte-carlo")
    mc.add_argument("--config", required=True)
    mc.add_argument("--simulations", type=int, default=1000)
    mc.add_argument("--seed", type=int, default=42)
    mc.add_argument("--block-size", type=int, default=20)
    mc.add_argument("--out-dir", default="reports/monte_carlo")

    analytics = sub.add_parser("analytics")
    analytics.add_argument("--config", required=True)
    analytics.add_argument("--out-dir", default="reports/analytics")

    plots = sub.add_parser("plots")
    plots.add_argument("--config", required=True)
    plots.add_argument("--out-dir", default="reports/plots")

    dashboard = sub.add_parser("dashboard")
    dashboard.add_argument("--config", required=True)
    dashboard.add_argument("--out", default="reports/dashboard.html")
    dashboard.add_argument("--news-risk", default="reports/news_risk.csv")
    dashboard.add_argument("--monte-carlo", default="reports/monte_carlo_smoke/monte_carlo_summary.csv")
    dashboard.add_argument("--quality", default="reports/data_quality.csv")

    args = parser.parse_args()
    if args.command == "make-demo-data":
        make_demo_data(args.out)
        print(f"Demo data written to {args.out}")
    elif args.command == "download":
        paths = download_yahoo_to_cache(args.symbols, args.start, args.end, args.cache)
        print("Downloaded:")
        for path in paths:
            print(f"- {path}")
    elif args.command == "prepare-real-data":
        symbols = ["SPY", "QQQ", "TLT", "GLD", "IEF", "EFA", "EEM", "VNQ", "DBC"]
        paths = download_yahoo_to_cache(symbols, args.start, args.end, args.cache)
        print("Downloaded real-data universe:")
        for path in paths:
            print(f"- {path}")
    elif args.command == "check-real-data":
        config = load_config(args.config)
        missing = missing_cached_symbols(config.data.symbols, config.data.cache_path)
        if missing:
            print("Missing cached symbols:")
            for symbol in missing:
                print(f"- {symbol}")
            print("Run: python -m quantbot.cli prepare-real-data")
        else:
            print(f"All {len(config.data.symbols)} cached symbols are present in {config.data.cache_path}")
    elif args.command == "backtest":
        config = load_config(args.config)
        result = run_backtest(load_market_data(config.data), config)
        _write_series(result.equity, args.equity_out)
        _write_frame(result.trades, args.trades_out)
        _print_performance(result)
    elif args.command == "optimize":
        config = load_config(args.config)
        grid = optimize(load_market_data(config.data), config, args.short, args.long, args.vol_target)
        _write_frame(grid, args.out)
        print(grid.head(10).to_string(index=False))
    elif args.command == "stability":
        config = load_config(args.config)
        grid = optimize(load_market_data(config.data), config, args.short, args.long, args.vol_target)
        summary = parameter_stability(grid)
        _write_frame(grid, args.grid_out)
        _write_frame(summary, args.summary_out)
        print(summary.to_string(index=False))
        if args.select_stable and not grid.empty:
            selected = select_stable_candidate(grid)
            print("Stable candidate:")
            print(selected.to_frame().T.to_string(index=False))
    elif args.command == "walk-forward":
        config = load_config(args.config)
        data = load_market_data(config.data)
        if args.short or args.long or args.vol_target:
            wf_result = walk_forward_optimized(
                data,
                config,
                train_years=args.train_years,
                test_years=args.test_years,
                short_windows=args.short or [config.strategy.short_window],
                long_windows=args.long or [config.strategy.long_window],
                vol_targets=args.vol_target or [config.strategy.target_annual_vol],
                max_windows=args.max_windows,
                selection_method=args.selection,
            )
        else:
            wf_result = walk_forward(data, config, train_years=args.train_years, test_years=args.test_years)
            if args.max_windows is not None:
                wf_result = wf_result.head(args.max_windows)
        _write_frame(wf_result, args.out)
        print(wf_result.to_string(index=False))
    elif args.command == "paper":
        config = load_config(args.config)
        data = load_market_data(config.data)
        closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in data.items()}).ffill().dropna()
        weights = build_weights(closes, config.strategy, config.risk)
        target = weights.iloc[-1]
        alpaca = None
        account_equity = config.initial_cash
        if args.use_alpaca_positions:
            alpaca = AlpacaPaperBroker(max_order_notional=args.max_order_notional)
            previous, account_equity, positions, account = alpaca.current_weights(target.index)
            positions_path, account_path = alpaca.write_position_snapshot(
                positions,
                account,
                args.alpaca_positions_out,
                args.alpaca_account_out,
            )
            print(f"Alpaca positions snapshot written to {positions_path}")
            print(f"Alpaca account snapshot written to {account_path}")
            print(f"Using Alpaca paper equity {account_equity:.2f} and live paper positions as current weights")
        else:
            previous = _read_previous_weights(args.previous_weights, target.index)
        broker = PaperBroker(account_equity)
        orders = broker.orders_from_weights(previous, target)
        path = broker.write_orders(orders, args.out)
        print(f"Paper orders written to {path}")
        _print_order_summary(orders, args.max_order_notional)
        if args.news_risk_check:
            hits, summary = _run_news_risk(config.data.symbols, args.news_file, None, 50, args.news_risk_out)
            print(f"News risk: {summary['risk_level']} score {float(summary['max_score']):.2f}, action {summary['action']}")
            if should_block_trading(summary, args.news_risk_threshold):
                print(f"Broker submission blocked by news risk threshold {args.news_risk_threshold:.2f}")
                return
        if args.state:
            prices = closes.iloc[-1]
            account = PaperAccount(args.state, config.initial_cash)
            fills = account.rebalance_to_weights(target, prices)
            print(f"Paper account updated: {len(fills)} fills, equity {account.equity(prices):.2f}")
        if args.submit_dry_run or args.broker == "dry-run":
            receipts = DryRunBroker().submit_orders(orders)
            print(f"Dry-run broker accepted {len(receipts)} orders")
        if args.broker == "alpaca-paper":
            alpaca = alpaca or AlpacaPaperBroker(max_order_notional=args.max_order_notional)
            receipts = alpaca.submit_orders(orders)
            receipts_path = alpaca.write_receipts(receipts, args.alpaca_receipts_out)
            print(f"Alpaca paper broker submitted {len(receipts)} child orders from {len(orders)} target orders")
            print(f"Alpaca receipts written to {receipts_path}")
    elif args.command == "alpaca-orders":
        alpaca = AlpacaPaperBroker()
        orders = alpaca.get_orders(args.status, args.limit)
        positions = alpaca.get_positions()
        account = alpaca.get_account()
        orders_path = alpaca.write_order_status(orders, args.orders_out)
        positions_path, account_path = alpaca.write_position_snapshot(positions, account, args.positions_out, args.account_out)
        print(f"Alpaca orders written to {orders_path} ({len(orders)} rows)")
        print(f"Alpaca positions written to {positions_path} ({len(positions)} rows)")
        print(f"Alpaca account written to {account_path}")
    elif args.command == "news-risk":
        config = load_config(args.config)
        hits, summary = _run_news_risk(config.data.symbols, args.news_file, args.rss, args.limit, args.out)
        print(f"News risk report written to {args.out}")
        print(f"Headlines: {summary['headline_count']}, hits: {summary['hit_count']}, max score: {float(summary['max_score']):.2f}")
        print(f"Risk level: {summary['risk_level']}, action: {summary['action']}")
        if hits:
            print(pd.DataFrame([hit.__dict__ for hit in hits]).head(10).to_string(index=False))
    elif args.command == "report":
        config = load_config(args.config)
        result = run_backtest(load_market_data(config.data), config)
        mc_summary = None
        if args.monte_carlo:
            mc_paths = simulate_return_paths(
                result.returns,
                config.initial_cash,
                simulations=args.mc_simulations,
                block_size=args.mc_block_size,
            )
            mc_summary = monte_carlo_summary(mc_paths)
        path = write_markdown_report(result, config, args.out, monte_carlo=mc_summary)
        print(f"Report written to {path}")
    elif args.command == "quality":
        config = load_config(args.config)
        issues = check_market_data(load_market_data(config.data))
        path = write_quality_report(issues, args.out)
        print(f"Data quality report written to {path}")
        frame = quality_summary(issues)
        if frame.empty:
            print("No data quality issues found")
        else:
            print(frame.to_string(index=False))
    elif args.command == "monte-carlo":
        config = load_config(args.config)
        result = run_backtest(load_market_data(config.data), config)
        paths = simulate_return_paths(
            result.returns,
            config.initial_cash,
            simulations=args.simulations,
            seed=args.seed,
            block_size=args.block_size,
        )
        summary = monte_carlo_summary(paths)
        files = write_monte_carlo(paths, summary, args.out_dir)
        print("Monte Carlo outputs written:")
        for name, path in files.items():
            print(f"- {name}: {path}")
        print(summary.to_string(index=False))
    elif args.command == "analytics":
        config = load_config(args.config)
        result = run_backtest(load_market_data(config.data), config)
        files = write_analytics(result, args.out_dir)
        print("Analytics written:")
        for name, path in files.items():
            print(f"- {name}: {path}")
    elif args.command == "plots":
        config = load_config(args.config)
        result = run_backtest(load_market_data(config.data), config)
        paths = write_plots(result, args.out_dir)
        print("Plots written:")
        for path in paths:
            print(f"- {path}")
    elif args.command == "dashboard":
        config = load_config(args.config)
        result = run_backtest(load_market_data(config.data), config)
        path = write_dashboard(result, config, args.out, args.news_risk, args.monte_carlo, args.quality)
        print(f"Dashboard written to {path}")


def _print_performance(result) -> None:
    perf = result.performance
    print(f"Total return:      {perf.total_return:.2%}")
    print(f"Annual return:     {perf.annual_return:.2%}")
    print(f"Annual volatility: {perf.annual_volatility:.2%}")
    print(f"Sharpe:            {perf.sharpe:.2f}")
    print(f"Sortino:           {perf.sortino:.2f}")
    print(f"Max drawdown:      {perf.max_drawdown:.2%}")
    print(f"Daily VaR 95:      {perf.value_at_risk_95:.2%}")
    print(f"Daily CVaR 95:     {perf.conditional_value_at_risk_95:.2%}")
    print(f"Win rate:          {perf.win_rate:.2%}")
    print(f"Trades:            {perf.trade_count}")
    print(f"Benchmark:         {result.benchmark_symbol} Sharpe {result.benchmark_performance.sharpe:.2f}, return {result.benchmark_performance.total_return:.2%}")
    print(f"Validation:        {'PASS' if result.passed_validation else 'FAIL'}")
    for message in result.validation_messages:
        print(f"- {message}")


def _write_series(series: pd.Series, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(output, header=True)


def _write_frame(frame: pd.DataFrame, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)


def _read_previous_weights(path: str | None, symbols: pd.Index) -> pd.Series:
    if not path:
        return pd.Series(0.0, index=symbols)
    frame = pd.read_csv(path)
    if not {"symbol", "weight"}.issubset(frame.columns):
        raise ValueError("previous weights file must include symbol,weight columns")
    return frame.set_index("symbol")["weight"].reindex(symbols).fillna(0.0)


def _print_order_summary(orders, max_order_notional: float) -> None:
    buys = sum(abs(order.estimated_notional) for order in orders if order.side == "BUY")
    sells = sum(abs(order.estimated_notional) for order in orders if order.side == "SELL")
    child_count = sum(len(split_order(order, max_order_notional)) for order in orders)
    print(f"Order summary: {len(orders)} target orders, {child_count} broker child orders")
    print(f"Buy notional: {buys:.2f}; sell notional: {sells:.2f}")


def _run_news_risk(symbols: list[str], news_file: str | None, rss: list[str] | None, limit: int, output: str):
    items = load_local_news(news_file) if news_file else fetch_rss_news(rss or None, limit=limit)
    hits, summary = score_news(items, symbols)
    write_news_risk_report(hits, summary, output)
    return hits, summary


if __name__ == "__main__":
    main()

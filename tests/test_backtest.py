import unittest
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quantbot.alpaca_broker import AlpacaPaperBroker, split_order
from quantbot.backtest import run_backtest, select_stable_candidate, walk_forward_optimized
from quantbot.broker import PaperAccount
from quantbot.broker import Order
from quantbot.config import BotConfig, CostConfig, DataConfig, RiskConfig, StrategyConfig, ValidationConfig
from quantbot.config import BenchmarkConfig
from quantbot.dashboard import write_dashboard
from quantbot.data import load_csv_data, make_demo_data
from quantbot.monte_carlo import monte_carlo_summary, simulate_return_paths
from quantbot.news_risk import NewsItem, score_news, should_block_trading
from quantbot.quality import check_market_data
from quantbot.strategy import build_weights


class BacktestTests(unittest.TestCase):
    def config(self, path: str) -> BotConfig:
        return BotConfig(
            initial_cash=100_000.0,
            data=DataConfig("csv", path, ["SPY", "TLT", "GLD"]),
            strategy=StrategyConfig("trend_volatility", 20, 80, 20, 0.10, "W-FRI"),
            risk=RiskConfig(0.40, 1.0, 0.25, 0.02),
            costs=CostConfig(0.5, 1.0),
            validation=ValidationConfig(0.0, 0.50, 1),
        )

    def test_demo_backtest_runs(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = self.config(tmp)
            data = load_csv_data(tmp, config.data.symbols, None, None)
            result = run_backtest(data, config)
            self.assertGreater(result.equity.iloc[-1], 0)
            self.assertEqual(set(result.weights.columns), {"SPY", "TLT", "GLD"})
            self.assertFalse(result.trades.empty)
            self.assertEqual(result.benchmark_symbol, "SPY")

    def test_signals_are_shifted(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = self.config(tmp)
            data = load_csv_data(tmp, config.data.symbols, None, None)
            result = run_backtest(data, config)
            first_invested = result.weights.abs().sum(axis=1).gt(0)
            self.assertGreater(first_invested.idxmax(), pd.Timestamp("2014-01-02"))

    def test_min_cash_caps_gross_exposure(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = self.config(tmp)
            data = load_csv_data(tmp, config.data.symbols, None, None)
            closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in data.items()}).ffill().dropna()
            weights = build_weights(closes, config.strategy, config.risk)
            self.assertLessEqual(weights.abs().sum(axis=1).max(), 0.98 + 1e-9)

    def test_dual_momentum_strategy_runs(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = BotConfig(
                initial_cash=100_000.0,
                data=DataConfig("csv", tmp, ["SPY", "TLT", "GLD"]),
                strategy=StrategyConfig("dual_momentum", 20, 80, 20, 0.10, "W-FRI", momentum_window=63, top_n=2),
                risk=RiskConfig(0.50, 1.0, 0.25, 0.02),
                costs=CostConfig(0.5, 1.0),
                validation=ValidationConfig(0.0, 0.50, 1),
            )
            data = load_csv_data(tmp, config.data.symbols, None, None)
            result = run_backtest(data, config)
            self.assertGreater(result.equity.iloc[-1], 0)
            self.assertIn("sortino", result.performance.as_dict())

    def test_regime_filter_reduces_exposure(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            base = self.config(tmp)
            no_filter = base
            filtered = BotConfig(
                initial_cash=base.initial_cash,
                data=base.data,
                strategy=StrategyConfig(
                    "trend_volatility",
                    20,
                    80,
                    20,
                    0.10,
                    "W-FRI",
                    regime_filter=True,
                    regime_symbol="SPY",
                    regime_window=200,
                ),
                risk=base.risk,
                costs=base.costs,
                validation=base.validation,
            )
            data = load_csv_data(tmp, base.data.symbols, None, None)
            closes = pd.DataFrame({symbol: frame["close"] for symbol, frame in data.items()}).ffill().dropna()
            unfiltered_weights = build_weights(closes, no_filter.strategy, no_filter.risk)
            filtered_weights = build_weights(closes, filtered.strategy, filtered.risk)
            self.assertLessEqual(filtered_weights.abs().sum(axis=1).mean(), unfiltered_weights.abs().sum(axis=1).mean())

    def test_defensive_fallback_allocates_when_risk_off(self):
        dates = pd.bdate_range("2024-01-01", periods=260)
        spy = pd.Series(100.0 - np.arange(260) * 0.10, index=dates)
        tlt = pd.Series(100.0 + np.arange(260) * 0.05, index=dates)
        gld = pd.Series(100.0, index=dates)
        closes = pd.DataFrame({"SPY": spy, "TLT": tlt, "GLD": gld})
        strategy = StrategyConfig(
            "trend_volatility",
            5,
            20,
            10,
            0.10,
            "W-FRI",
            regime_filter=True,
            regime_symbol="SPY",
            regime_window=50,
            regime_risk_off_weight=0.50,
            defensive_assets=["TLT", "GLD"],
            defensive_top_n=1,
            defensive_momentum_window=20,
        )
        weights = build_weights(closes, strategy, RiskConfig(0.60, 1.0, 0.25, 0.02))
        self.assertGreater(weights["TLT"].iloc[-1], 0)
        self.assertEqual(weights["SPY"].iloc[-1], 0)

    def test_explicit_and_blended_benchmark(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = self.config(tmp)
            data = load_csv_data(tmp, config.data.symbols, None, None)
            explicit = BotConfig(
                config.initial_cash,
                config.data,
                config.strategy,
                config.risk,
                config.costs,
                config.validation,
                BenchmarkConfig(symbol="TLT"),
            )
            explicit_result = run_backtest(data, explicit)
            self.assertEqual(explicit_result.benchmark_symbol, "TLT")
            blended = BotConfig(
                config.initial_cash,
                config.data,
                config.strategy,
                config.risk,
                config.costs,
                config.validation,
                BenchmarkConfig(weights={"SPY": 0.6, "TLT": 0.4}),
            )
            blended_result = run_backtest(data, blended)
            self.assertIn("SPY", blended_result.benchmark_symbol)
            self.assertIn("TLT", blended_result.benchmark_symbol)

    def test_correlation_adjusted_allocator_runs(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = BotConfig(
                initial_cash=100_000.0,
                data=DataConfig("csv", tmp, ["SPY", "TLT", "GLD"]),
                strategy=StrategyConfig(
                    "dual_momentum",
                    20,
                    80,
                    20,
                    0.10,
                    "W-FRI",
                    momentum_window=63,
                    top_n=2,
                    allocation_method="correlation_adjusted",
                    correlation_window=20,
                ),
                risk=RiskConfig(0.50, 1.0, 0.25, 0.02),
                costs=CostConfig(0.5, 1.0),
                validation=ValidationConfig(0.0, 0.50, 1),
            )
            data = load_csv_data(tmp, config.data.symbols, None, None)
            result = run_backtest(data, config)
            self.assertGreater(result.equity.iloc[-1], 0)
            self.assertLessEqual(result.weights.abs().sum(axis=1).max(), 0.98 + 1e-9)

    def test_stable_candidate_selection_prefers_cluster(self):
        grid = pd.DataFrame(
            [
                {"short_window": 10, "long_window": 100, "target_annual_vol": 0.10, "passed": True, "sharpe": 3.0, "calmar": 0.5, "max_drawdown": 0.30},
                {"short_window": 40, "long_window": 160, "target_annual_vol": 0.08, "passed": True, "sharpe": 1.1, "calmar": 1.0, "max_drawdown": 0.10},
                {"short_window": 45, "long_window": 165, "target_annual_vol": 0.08, "passed": True, "sharpe": 1.0, "calmar": 1.0, "max_drawdown": 0.10},
                {"short_window": 50, "long_window": 170, "target_annual_vol": 0.08, "passed": True, "sharpe": 1.0, "calmar": 0.9, "max_drawdown": 0.11},
            ]
        )
        selected = select_stable_candidate(grid, neighbor_radius=0.25)
        self.assertNotEqual(int(selected["short_window"]), 10)

    def test_optimized_walk_forward_records_selected_parameters(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = self.config(tmp)
            data = load_csv_data(tmp, config.data.symbols, None, None)
            result = walk_forward_optimized(
                data,
                config,
                train_years=3,
                test_years=1,
                short_windows=[20],
                long_windows=[80],
                vol_targets=[0.08, 0.10],
                selection_method="stable",
            )
            self.assertIn("selected_short_window", result.columns)
            self.assertIn("train_sharpe", result.columns)
            self.assertIn("selection_method", result.columns)

    def test_paper_account_persists_state(self):
        with TemporaryDirectory() as tmp:
            state_path = f"{tmp}/paper_state.json"
            account = PaperAccount(state_path, 10_000.0)
            target = pd.Series({"SPY": 0.50, "TLT": 0.25})
            prices = pd.Series({"SPY": 100.0, "TLT": 50.0})
            fills = account.rebalance_to_weights(target, prices)
            reloaded = PaperAccount(state_path, 10_000.0)
            self.assertEqual(len(fills), 2)
            self.assertGreater(reloaded.positions["SPY"], 0)
            self.assertLess(reloaded.cash, 10_000.0)

    def test_quality_checks_flag_bad_ohlc(self):
        dates = pd.bdate_range("2024-01-01", periods=3)
        frame = pd.DataFrame(
            {
                "open": [10.0, 10.5, 11.0],
                "high": [10.2, 10.4, 10.9],
                "low": [9.8, 10.1, 10.7],
                "close": [10.1, 10.6, 11.2],
                "volume": [1000.0, 0.0, 1200.0],
            },
            index=dates,
        )
        issues = check_market_data({"BAD": frame})
        checks = {issue.check for issue in issues}
        self.assertIn("ohlc", checks)
        self.assertIn("volume", checks)

    def test_monte_carlo_summary_runs(self):
        returns = pd.Series([0.01, -0.005, 0.002, 0.004, -0.003] * 20)
        paths = simulate_return_paths(returns, 100_000.0, simulations=25, seed=1, block_size=5)
        summary = monte_carlo_summary(paths)
        self.assertEqual(len(paths), 25)
        self.assertIn("loss_probability", set(summary["metric"]))

    def test_alpaca_requires_paper_flag(self):
        old_key = __import__("os").environ.get("ALPACA_API_KEY")
        old_secret = __import__("os").environ.get("ALPACA_SECRET_KEY")
        old_paper = __import__("os").environ.get("ALPACA_PAPER")
        os = __import__("os")
        try:
            os.environ["ALPACA_API_KEY"] = "key"
            os.environ["ALPACA_SECRET_KEY"] = "secret"
            os.environ["ALPACA_PAPER"] = "false"
            with self.assertRaises(RuntimeError):
                AlpacaPaperBroker()
        finally:
            _restore_env("ALPACA_API_KEY", old_key)
            _restore_env("ALPACA_SECRET_KEY", old_secret)
            _restore_env("ALPACA_PAPER", old_paper)

    def test_alpaca_order_splitting(self):
        order = Order("DBC", 0.35, 0.35, 35_000.0, "BUY")
        chunks = split_order(order, 5_000.0)
        self.assertEqual(len(chunks), 7)
        self.assertEqual(sum(abs(chunk.estimated_notional) for chunk in chunks), 35_000.0)
        self.assertTrue(all(abs(chunk.estimated_notional) <= 5_000.0 for chunk in chunks))

    def test_news_risk_detects_oil_geopolitics(self):
        items = [
            NewsItem("Iran conflict raises fears over Strait of Hormuz oil supply disruption", "test"),
            NewsItem("Retail earnings calendar remains quiet", "test"),
        ]
        hits, summary = score_news(items, ["SPY", "QQQ", "GLD", "DBC", "TLT"])
        themes = {hit.theme for hit in hits}
        self.assertIn("middle_east_conflict", themes)
        self.assertIn("oil_supply_shock", themes)
        self.assertTrue(should_block_trading(summary, 0.5))

    def test_dashboard_writes_html(self):
        with TemporaryDirectory() as tmp:
            make_demo_data(tmp)
            config = self.config(tmp)
            data = load_csv_data(tmp, config.data.symbols, None, None)
            result = run_backtest(data, config)
            path = write_dashboard(result, config, f"{tmp}/dashboard.html", None, None, None)
            content = path.read_text(encoding="utf-8")
            self.assertIn("Quant Bot Dashboard", content)
            self.assertIn("Executive Snapshot", content)
            self.assertIn("Equity Curve vs Benchmark", content)


def _restore_env(name: str, value: str | None) -> None:
    os = __import__("os")
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Protocol

import pandas as pd


@dataclass(frozen=True)
class Order:
    symbol: str
    target_weight: float
    weight_delta: float
    estimated_notional: float
    side: str = "BUY"


class Broker(Protocol):
    def submit_orders(self, orders: list[Order]) -> list[dict]:
        ...


class DryRunBroker:
    def submit_orders(self, orders: list[Order]) -> list[dict]:
        return [
            {
                "symbol": order.symbol,
                "side": order.side,
                "estimated_notional": order.estimated_notional,
                "status": "dry_run",
            }
            for order in orders
        ]


class LiveBrokerDisabled:
    def submit_orders(self, orders: list[Order]) -> list[dict]:
        raise RuntimeError(
            "Live broker submission is intentionally disabled. Implement a broker adapter, paper trade it, "
            "and add explicit credentials/kill-switch checks before using real capital."
        )


class PaperBroker:
    def __init__(self, account_equity: float, min_order_notional: float = 50.0) -> None:
        self.account_equity = account_equity
        self.min_order_notional = min_order_notional

    def orders_from_weights(self, previous: pd.Series, target: pd.Series) -> list[Order]:
        symbols = sorted(set(previous.index) | set(target.index))
        orders: list[Order] = []
        for symbol in symbols:
            old = float(previous.get(symbol, 0.0))
            new = float(target.get(symbol, 0.0))
            delta = new - old
            notional = delta * self.account_equity
            if abs(notional) >= self.min_order_notional:
                orders.append(Order(symbol, new, delta, notional, "BUY" if delta > 0 else "SELL"))
        return orders

    def write_orders(self, orders: list[Order], output: str) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([order.__dict__ for order in orders])
        if frame.empty:
            frame = pd.DataFrame(columns=["symbol", "target_weight", "weight_delta", "estimated_notional"])
        frame.to_csv(path, index=False)
        return path


class PaperAccount:
    def __init__(self, state_path: str, initial_cash: float) -> None:
        self.state_path = Path(state_path)
        self.initial_cash = initial_cash
        self.state = self._load()

    @property
    def cash(self) -> float:
        return float(self.state["cash"])

    @property
    def positions(self) -> dict[str, float]:
        return {symbol: float(shares) for symbol, shares in self.state["positions"].items()}

    def equity(self, prices: pd.Series) -> float:
        position_value = sum(shares * float(prices.get(symbol, 0.0)) for symbol, shares in self.positions.items())
        return self.cash + position_value

    def rebalance_to_weights(self, target: pd.Series, prices: pd.Series, min_order_notional: float = 50.0) -> list[dict]:
        equity = self.equity(prices)
        fills = []
        for symbol, weight in target.items():
            price = float(prices.get(symbol, 0.0))
            if price <= 0:
                continue
            current_shares = self.positions.get(symbol, 0.0)
            current_value = current_shares * price
            target_value = float(weight) * equity
            delta_value = target_value - current_value
            if abs(delta_value) < min_order_notional:
                continue
            shares_delta = delta_value / price
            if delta_value > self.cash:
                shares_delta = self.cash / price
                delta_value = shares_delta * price
            self.state["cash"] = float(self.state["cash"]) - delta_value
            self.state["positions"][symbol] = current_shares + shares_delta
            fills.append(
                {
                    "symbol": symbol,
                    "shares": shares_delta,
                    "price": price,
                    "notional": delta_value,
                    "side": "BUY" if shares_delta > 0 else "SELL",
                }
            )
        self.state["history"].extend(fills)
        self.save()
        return fills

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def _load(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {"cash": self.initial_cash, "positions": {}, "history": []}

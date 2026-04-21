from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

from .broker import Order


class AlpacaPaperBroker:
    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        max_order_notional: float = 10_000.0,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.secret_key = secret_key or os.getenv("ALPACA_SECRET_KEY")
        self.paper_flag = os.getenv("ALPACA_PAPER", "true").lower()
        self.max_order_notional = max_order_notional
        self._client = None
        self._sdk: dict[str, Any] | None = None
        self._validate_environment()

    def submit_orders(self, orders: list[Order]) -> list[dict]:
        client = self._client_instance()
        sdk = self._sdk_classes()
        receipts = []
        for order in orders:
            for child in split_order(order, self.max_order_notional):
                request = sdk["MarketOrderRequest"](
                    symbol=child.symbol,
                    notional=round(abs(child.estimated_notional), 2),
                    side=sdk["OrderSide"].BUY if child.side == "BUY" else sdk["OrderSide"].SELL,
                    time_in_force=sdk["TimeInForce"].DAY,
                )
                submitted = client.submit_order(order_data=request)
                receipts.append(_normalize_submission(submitted, child))
        return receipts

    def write_receipts(self, receipts: list[dict], output: str) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        columns = ["symbol", "side", "notional", "status", "id", "client_order_id"]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for receipt in receipts:
                writer.writerow({column: receipt.get(column, "") for column in columns})
        return path

    def _validate_environment(self) -> None:
        if not self.api_key or not self.secret_key:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for Alpaca paper trading")
        if self.paper_flag not in {"true", "1", "yes"}:
            raise RuntimeError("Refusing Alpaca submission unless ALPACA_PAPER=true")

    def _client_instance(self):
        if self._client is None:
            sdk = self._sdk_classes()
            self._client = sdk["TradingClient"](self.api_key, self.secret_key, paper=True)
        return self._client

    def _sdk_classes(self) -> dict[str, Any]:
        if self._sdk is None:
            try:
                from alpaca.trading.client import TradingClient
                from alpaca.trading.enums import OrderSide, TimeInForce
                from alpaca.trading.requests import MarketOrderRequest
            except ImportError as exc:
                raise RuntimeError("Install alpaca-py first: python -m pip install alpaca-py") from exc
            self._sdk = {
                "TradingClient": TradingClient,
                "OrderSide": OrderSide,
                "TimeInForce": TimeInForce,
                "MarketOrderRequest": MarketOrderRequest,
            }
        return self._sdk


def _normalize_submission(submitted: Any, order: Order) -> dict:
    return {
        "symbol": getattr(submitted, "symbol", order.symbol),
        "side": str(getattr(submitted, "side", order.side)),
        "notional": abs(order.estimated_notional),
        "status": str(getattr(submitted, "status", "submitted")),
        "id": str(getattr(submitted, "id", "")),
        "client_order_id": str(getattr(submitted, "client_order_id", "")),
    }


def split_order(order: Order, max_order_notional: float) -> list[Order]:
    if max_order_notional <= 0:
        raise ValueError("max_order_notional must be positive")
    notional = abs(order.estimated_notional)
    if notional <= max_order_notional:
        return [order]
    chunks = []
    remaining = notional
    sign = 1 if order.estimated_notional >= 0 else -1
    while remaining > 0:
        chunk_notional = min(max_order_notional, remaining)
        chunk_delta = order.weight_delta * (chunk_notional / notional)
        chunks.append(
            Order(
                symbol=order.symbol,
                target_weight=order.target_weight,
                weight_delta=chunk_delta,
                estimated_notional=sign * chunk_notional,
                side=order.side,
            )
        )
        remaining = round(remaining - chunk_notional, 10)
    return chunks

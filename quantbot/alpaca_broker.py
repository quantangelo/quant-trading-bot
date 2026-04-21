from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import pandas as pd

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

    def get_orders(self, status: str = "all", limit: int = 100) -> list[dict]:
        client = self._client_instance()
        sdk = self._sdk_classes()
        status_key = status.lower()
        if status_key not in {"open", "closed", "all"}:
            raise ValueError("status must be one of: open, closed, all")
        request = sdk["GetOrdersRequest"](
            status={
                "open": sdk["QueryOrderStatus"].OPEN,
                "closed": sdk["QueryOrderStatus"].CLOSED,
                "all": sdk["QueryOrderStatus"].ALL,
            }[status_key],
            limit=limit,
        )
        return [normalize_order_status(order) for order in client.get_orders(filter=request)]

    def get_positions(self) -> list[dict]:
        client = self._client_instance()
        return [normalize_position(position) for position in client.get_all_positions()]

    def get_account(self) -> dict:
        client = self._client_instance()
        account = client.get_account()
        return {
            "id": _clean(getattr(account, "id", "")),
            "status": _clean(getattr(account, "status", "")),
            "currency": _clean(getattr(account, "currency", "")),
            "cash": _float_value(account, "cash"),
            "buying_power": _float_value(account, "buying_power"),
            "equity": _float_value(account, "equity"),
            "portfolio_value": _float_value(account, "portfolio_value"),
            "long_market_value": _float_value(account, "long_market_value"),
        }

    def current_weights(self, symbols: pd.Index) -> tuple[pd.Series, float, list[dict], dict]:
        positions = self.get_positions()
        account = self.get_account()
        equity = float(account.get("equity") or account.get("portfolio_value") or 0.0)
        if equity <= 0:
            raise RuntimeError("Alpaca account equity is zero or unavailable")
        weights = pd.Series(0.0, index=symbols)
        for position in positions:
            symbol = position.get("symbol", "")
            if symbol in weights.index:
                weights.loc[symbol] = float(position.get("market_value") or 0.0) / equity
        return weights, equity, positions, account

    def write_receipts(self, receipts: list[dict], output: str) -> Path:
        return _write_dict_rows(receipts, output, ORDER_COLUMNS)

    def write_order_status(self, rows: list[dict], output: str) -> Path:
        return _write_dict_rows(rows, output, ORDER_COLUMNS)

    def write_positions(self, rows: list[dict], output: str) -> Path:
        return _write_dict_rows(rows, output, POSITION_COLUMNS)

    def write_account(self, row: dict, output: str) -> Path:
        return _write_dict_rows([row], output, ACCOUNT_COLUMNS)

    def write_position_snapshot(self, positions: list[dict], account: dict, positions_output: str, account_output: str) -> tuple[Path, Path]:
        return self.write_positions(positions, positions_output), self.write_account(account, account_output)

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
                from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
                from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
            except ImportError as exc:
                raise RuntimeError("Install alpaca-py first: python -m pip install alpaca-py") from exc
            self._sdk = {
                "TradingClient": TradingClient,
                "OrderSide": OrderSide,
                "QueryOrderStatus": QueryOrderStatus,
                "TimeInForce": TimeInForce,
                "GetOrdersRequest": GetOrdersRequest,
                "MarketOrderRequest": MarketOrderRequest,
            }
        return self._sdk


ORDER_COLUMNS = [
    "symbol",
    "side",
    "notional",
    "qty",
    "filled_qty",
    "filled_avg_price",
    "status",
    "id",
    "client_order_id",
    "order_type",
    "time_in_force",
    "submitted_at",
    "filled_at",
    "created_at",
    "updated_at",
]

POSITION_COLUMNS = [
    "symbol",
    "qty",
    "market_value",
    "avg_entry_price",
    "current_price",
    "unrealized_pl",
    "unrealized_plpc",
    "side",
]

ACCOUNT_COLUMNS = [
    "id",
    "status",
    "currency",
    "cash",
    "buying_power",
    "equity",
    "portfolio_value",
    "long_market_value",
]


def _write_dict_rows(rows: list[dict], output: str, columns: list[str]) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


def normalize_order_status(order: Any) -> dict:
    return {
        "symbol": _clean(getattr(order, "symbol", "")),
        "side": _clean(getattr(order, "side", "")),
        "notional": _clean(getattr(order, "notional", "")),
        "qty": _clean(getattr(order, "qty", "")),
        "filled_qty": _clean(getattr(order, "filled_qty", "")),
        "filled_avg_price": _clean(getattr(order, "filled_avg_price", "")),
        "status": _clean(getattr(order, "status", "")),
        "id": _clean(getattr(order, "id", "")),
        "client_order_id": _clean(getattr(order, "client_order_id", "")),
        "order_type": _clean(getattr(order, "order_type", getattr(order, "type", ""))),
        "time_in_force": _clean(getattr(order, "time_in_force", "")),
        "submitted_at": _clean(getattr(order, "submitted_at", "")),
        "filled_at": _clean(getattr(order, "filled_at", "")),
        "created_at": _clean(getattr(order, "created_at", "")),
        "updated_at": _clean(getattr(order, "updated_at", "")),
    }


def normalize_position(position: Any) -> dict:
    return {
        "symbol": _clean(getattr(position, "symbol", "")),
        "qty": _float_value(position, "qty"),
        "market_value": _float_value(position, "market_value"),
        "avg_entry_price": _float_value(position, "avg_entry_price"),
        "current_price": _float_value(position, "current_price"),
        "unrealized_pl": _float_value(position, "unrealized_pl"),
        "unrealized_plpc": _float_value(position, "unrealized_plpc"),
        "side": _clean(getattr(position, "side", "")),
    }


def _normalize_submission(submitted: Any, order: Order) -> dict:
    normalized = normalize_order_status(submitted)
    normalized["symbol"] = normalized["symbol"] or order.symbol
    normalized["side"] = normalized["side"] or order.side
    normalized["notional"] = normalized["notional"] or abs(order.estimated_notional)
    normalized["status"] = normalized["status"] or "submitted"
    return normalized


def _float_value(obj: Any, name: str) -> float:
    value = getattr(obj, name, 0.0)
    if value in {None, ""}:
        return 0.0
    return float(value)


def _clean(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value)


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

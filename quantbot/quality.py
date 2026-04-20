from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data import OHLCV


@dataclass(frozen=True)
class DataQualityIssue:
    symbol: str
    severity: str
    check: str
    message: str
    count: int


def check_market_data(data: dict[str, pd.DataFrame]) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    for symbol, frame in data.items():
        issues.extend(_check_symbol(symbol, frame))
    return issues


def quality_summary(issues: list[DataQualityIssue]) -> pd.DataFrame:
    columns = ["symbol", "severity", "check", "message", "count"]
    if not issues:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame([issue.__dict__ for issue in issues], columns=columns)


def write_quality_report(issues: list[DataQualityIssue], output: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    quality_summary(issues).to_csv(path, index=False)
    return path


def _check_symbol(symbol: str, frame: pd.DataFrame) -> list[DataQualityIssue]:
    issues: list[DataQualityIssue] = []
    missing_columns = [column for column in OHLCV if column not in frame.columns]
    if missing_columns:
        issues.append(
            DataQualityIssue(symbol, "error", "columns", f"Missing columns: {', '.join(missing_columns)}", len(missing_columns))
        )
        return issues
    if not frame.index.is_monotonic_increasing:
        issues.append(DataQualityIssue(symbol, "error", "index", "Dates are not sorted ascending", 1))
    duplicates = int(frame.index.duplicated().sum())
    if duplicates:
        issues.append(DataQualityIssue(symbol, "error", "duplicates", "Duplicate timestamps found", duplicates))
    nulls = int(frame.loc[:, OHLCV].isna().sum().sum())
    if nulls:
        issues.append(DataQualityIssue(symbol, "error", "nulls", "Missing OHLCV values found", nulls))
    non_positive_prices = int((frame[["open", "high", "low", "close"]] <= 0).sum().sum())
    if non_positive_prices:
        issues.append(DataQualityIssue(symbol, "error", "prices", "Non-positive OHLC prices found", non_positive_prices))
    non_positive_volume = int((frame["volume"] <= 0).sum())
    if non_positive_volume:
        issues.append(DataQualityIssue(symbol, "warning", "volume", "Non-positive volume rows found", non_positive_volume))
    bad_high_low = int(((frame["high"] < frame[["open", "close"]].max(axis=1)) | (frame["low"] > frame[["open", "close"]].min(axis=1))).sum())
    if bad_high_low:
        issues.append(DataQualityIssue(symbol, "error", "ohlc", "High/low does not contain open/close", bad_high_low))
    missing_days = _missing_business_days(frame)
    if missing_days:
        issues.append(DataQualityIssue(symbol, "warning", "calendar", "Missing business days in date range", missing_days))
    stale = _stale_prices(frame["close"])
    if stale:
        issues.append(DataQualityIssue(symbol, "warning", "stale", "Repeated close prices found", stale))
    outliers = _return_outliers(frame["close"])
    if outliers:
        issues.append(DataQualityIssue(symbol, "warning", "outliers", "Large daily return outliers found", outliers))
    return issues


def _missing_business_days(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    expected = pd.bdate_range(frame.index.min(), frame.index.max())
    return int(len(expected.difference(frame.index)))


def _stale_prices(close: pd.Series) -> int:
    repeated = close.pct_change().fillna(0.0).eq(0.0)
    return int(repeated.rolling(5).sum().ge(5).sum())


def _return_outliers(close: pd.Series) -> int:
    returns = close.pct_change().dropna()
    if returns.empty:
        return 0
    return int(returns.abs().gt(0.25).sum())

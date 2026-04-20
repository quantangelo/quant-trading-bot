from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig

OHLCV = ["open", "high", "low", "close", "volume"]


def load_market_data(config: DataConfig) -> dict[str, pd.DataFrame]:
    if config.source.lower() == "csv":
        return load_csv_data(config.path, config.symbols, config.start, config.end)
    if config.source.lower() in {"cache", "cached"}:
        return load_csv_data(config.cache_path, config.symbols, config.start, config.end)
    if config.source.lower() in {"yahoo", "yfinance"}:
        return load_yahoo_data(config.symbols, config.start, config.end, config.cache_path)
    raise ValueError(f"Unsupported data source: {config.source}")


def load_csv_data(path: str, symbols: list[str], start: str | None, end: str | None) -> dict[str, pd.DataFrame]:
    base = Path(path)
    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        file_path = base / f"{symbol}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Missing CSV for {symbol}: {file_path}")
        frame = pd.read_csv(file_path)
        frame.columns = [str(col).lower() for col in frame.columns]
        if "date" not in frame.columns:
            raise ValueError(f"{file_path} must include a date column")
        missing = set(OHLCV) - set(frame.columns)
        if missing:
            raise ValueError(f"{file_path} missing required columns: {sorted(missing)}")
        frame["date"] = pd.to_datetime(frame["date"], utc=True).dt.tz_convert(None)
        frame = frame.set_index("date").sort_index()
        frame = frame.loc[:, OHLCV].astype(float)
        data[symbol] = _slice(frame, start, end)
    return data


def load_yahoo_data(symbols: list[str], start: str | None, end: str | None, cache_path: str | None = None) -> dict[str, pd.DataFrame]:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("Install yfinance or use CSV data: python -m pip install yfinance") from exc

    data: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        frame = yf.download(symbol, start=start, end=end, auto_adjust=False, progress=False)
        if frame.empty:
            raise ValueError(f"No Yahoo data returned for {symbol}")
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
        frame = frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        frame.index = pd.to_datetime(frame.index)
        normalized = frame.loc[:, OHLCV].astype(float)
        data[symbol] = normalized
        if cache_path:
            write_symbol_csv(symbol, normalized, cache_path)
    return data


def download_yahoo_to_cache(symbols: list[str], start: str | None, end: str | None, cache_path: str) -> list[Path]:
    data = load_yahoo_data(symbols, start, end, cache_path)
    return [Path(cache_path) / f"{symbol}.csv" for symbol in data]


def missing_cached_symbols(symbols: list[str], cache_path: str) -> list[str]:
    base = Path(cache_path)
    return [symbol for symbol in symbols if not (base / f"{symbol}.csv").exists()]


def close_matrix(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    closes = {symbol: frame["close"] for symbol, frame in data.items()}
    return pd.DataFrame(closes).dropna(how="all").ffill().dropna()


def field_matrix(data: dict[str, pd.DataFrame], field: str) -> pd.DataFrame:
    values = {symbol: frame[field] for symbol, frame in data.items()}
    return pd.DataFrame(values).dropna(how="all").ffill().dropna()


def write_symbol_csv(symbol: str, frame: pd.DataFrame, out: str) -> Path:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    export = frame.loc[:, OHLCV].copy()
    export.insert(0, "date", export.index)
    path = output / f"{symbol}.csv"
    export.to_csv(path, index=False)
    return path


def make_demo_data(out: str, seed: int = 7) -> None:
    output = Path(out)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2014-01-02", "2025-12-31")
    specs = {
        "SPY": (0.085, 0.16, 430.0),
        "TLT": (0.030, 0.13, 95.0),
        "GLD": (0.055, 0.15, 190.0),
    }
    for symbol, (drift, vol, start_price) in specs.items():
        daily_mu = drift / 252
        daily_vol = vol / np.sqrt(252)
        shocks = rng.normal(daily_mu, daily_vol, len(dates))
        cycle = 0.0008 * np.sin(np.linspace(0, 10 * np.pi, len(dates)))
        close = start_price * np.exp(np.cumsum(shocks + cycle))
        open_ = close * (1 + rng.normal(0, 0.002, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.0005, 0.01, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.0005, 0.01, len(dates)))
        volume = rng.integers(1_000_000, 8_000_000, len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
            }
        )
        frame.to_csv(output / f"{symbol}.csv", index=False)


def _slice(frame: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if start:
        frame = frame.loc[pd.Timestamp(start) :]
    if end:
        frame = frame.loc[: pd.Timestamp(end)]
    return frame

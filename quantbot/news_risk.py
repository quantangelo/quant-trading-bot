from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import re
from typing import Iterable
from urllib.request import urlopen
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str = "local"
    url: str = ""


@dataclass(frozen=True)
class NewsRiskHit:
    theme: str
    score: float
    source: str
    title: str
    url: str
    positive_assets: str
    negative_assets: str
    defensive_assets: str


THEMES = {
    "middle_east_conflict": {
        "keywords": ["iran", "israel", "middle east", "strait of hormuz", "missile", "war", "sanction"],
        "positive": ["GLD", "DBC", "USO", "XLE"],
        "negative": ["SPY", "QQQ", "EFA", "EEM"],
        "defensive": ["IEF", "TLT", "GLD"],
        "weight": 1.0,
    },
    "oil_supply_shock": {
        "keywords": ["oil", "crude", "opec", "brent", "wti", "pipeline", "refinery", "supply disruption"],
        "positive": ["DBC", "USO", "XLE"],
        "negative": ["SPY", "QQQ"],
        "defensive": ["GLD", "IEF"],
        "weight": 0.9,
    },
    "inflation_rates": {
        "keywords": ["inflation", "cpi", "ppi", "fed", "interest rate", "yields", "hawkish"],
        "positive": ["DBC", "GLD"],
        "negative": ["TLT", "IEF", "QQQ"],
        "defensive": ["GLD"],
        "weight": 0.7,
    },
    "banking_credit_stress": {
        "keywords": ["bank failure", "credit crisis", "liquidity crunch", "default", "contagion"],
        "positive": ["TLT", "IEF", "GLD"],
        "negative": ["SPY", "QQQ", "EFA", "EEM", "VNQ"],
        "defensive": ["TLT", "IEF", "GLD"],
        "weight": 0.9,
    },
    "china_growth_risk": {
        "keywords": ["china", "property crisis", "stimulus", "yuan", "tariff"],
        "positive": ["GLD"],
        "negative": ["EEM", "DBC", "EFA"],
        "defensive": ["IEF", "GLD"],
        "weight": 0.6,
    },
}


DEFAULT_RSS_FEEDS = [
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
]


def load_local_news(path: str) -> list[NewsItem]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing news file: {path}")
    if file_path.suffix.lower() == ".json":
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        return [NewsItem(str(item.get("title", "")), str(item.get("source", "local")), str(item.get("url", ""))) for item in raw]
    items = []
    for line in file_path.read_text(encoding="utf-8").splitlines():
        title = line.strip()
        if title:
            items.append(NewsItem(title=title))
    return items


def fetch_rss_news(feeds: Iterable[str] | None = None, limit: int = 50, timeout: int = 10) -> list[NewsItem]:
    items: list[NewsItem] = []
    for feed in feeds or DEFAULT_RSS_FEEDS:
        try:
            with urlopen(feed, timeout=timeout) as response:
                root = ET.fromstring(response.read())
        except Exception:
            continue
        for item in root.findall(".//item"):
            title = _text(item, "title")
            link = _text(item, "link")
            if title:
                items.append(NewsItem(title=title, source=feed, url=link))
            if len(items) >= limit:
                return items
    return items


def score_news(items: list[NewsItem], symbols: list[str]) -> tuple[list[NewsRiskHit], dict[str, float | str | int]]:
    hits: list[NewsRiskHit] = []
    universe = set(symbols)
    for item in items:
        normalized = _normalize(item.title)
        for theme, spec in THEMES.items():
            matches = [keyword for keyword in spec["keywords"] if keyword in normalized]
            if not matches:
                continue
            score = min(1.0, float(spec["weight"]) * (0.35 + 0.15 * len(matches)))
            hits.append(
                NewsRiskHit(
                    theme=theme,
                    score=score,
                    source=item.source,
                    title=item.title,
                    url=item.url,
                    positive_assets=",".join(asset for asset in spec["positive"] if asset in universe),
                    negative_assets=",".join(asset for asset in spec["negative"] if asset in universe),
                    defensive_assets=",".join(asset for asset in spec["defensive"] if asset in universe),
                )
            )
    max_score = max((hit.score for hit in hits), default=0.0)
    summary = {
        "headline_count": len(items),
        "hit_count": len(hits),
        "max_score": max_score,
        "risk_level": risk_level(max_score),
        "action": recommended_action(max_score),
    }
    return hits, summary


def risk_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    if score > 0:
        return "low"
    return "none"


def recommended_action(score: float) -> str:
    if score >= 0.75:
        return "block_new_risk_orders"
    if score >= 0.45:
        return "manual_review"
    if score > 0:
        return "log_only"
    return "none"


def should_block_trading(summary: dict[str, float | str | int], threshold: float) -> bool:
    return float(summary.get("max_score", 0.0)) >= threshold


def write_news_risk_report(hits: list[NewsRiskHit], summary: dict[str, float | str | int], output: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["theme", "score", "source", "title", "url", "positive_assets", "negative_assets", "defensive_assets"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for hit in hits:
            writer.writerow(hit.__dict__)
    summary_path = path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def _text(item: ET.Element, tag: str) -> str:
    child = item.find(tag)
    return "" if child is None or child.text is None else child.text.strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())

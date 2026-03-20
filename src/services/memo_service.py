from __future__ import annotations

from datetime import datetime, timezone

from src.services.llm_service import AnalysisResult
from src.services.market_data_service import MarketSnapshot


def build_research_memo(
    analysis: AnalysisResult,
    mode: str,
    raw_input: str,
    market: MarketSnapshot | None = None,
) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    if market is not None:
        company_line = f"{market.company_name} ({market.ticker})"
        ticker_line = market.ticker
    else:
        company_line = "N/A (News-Driven Analysis)"
        ticker_line = "N/A"

    lines = [
        "# Internal Research Memo",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Company / Ticker:** {company_line}",
        f"**Ticker:** {ticker_line}",
        f"**Source Mode:** {mode}",
        "",
        "## 1) Executive Summary",
        analysis.summary,
        "",
        "## 2) Sentiment",
        analysis.sentiment.title(),
        "",
        "## 3) Key Investment Insights",
        f"- {analysis.key_insights[0]}",
        f"- {analysis.key_insights[1]}",
        f"- {analysis.key_insights[2]}",
        "",
        "## 4) Risks / Watch Items",
        f"- {analysis.risks[0]}",
        f"- {analysis.risks[1]}",
        f"- {analysis.risks[2]}",
        "",
        "## 5) Bottom Line",
        analysis.conclusion,
        "",
        "## Input Reference",
        raw_input.strip(),
        "",
        "---",
        "Internal use only.",
    ]
    return "\n".join(lines)


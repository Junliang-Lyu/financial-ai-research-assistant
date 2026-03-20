from __future__ import annotations

from dataclasses import dataclass

from src.services.llm_service import AnalysisResult, LLMService
from src.services.market_data_service import MarketSnapshot, fetch_market_snapshot


@dataclass
class TickerAnalysisResponse:
    market: MarketSnapshot
    analysis: AnalysisResult


def analyze_ticker(ticker: str, llm_service: LLMService) -> TickerAnalysisResponse:
    market = fetch_market_snapshot(ticker)

    context = (
        f"Ticker: {market.ticker}\n"
        f"Company: {market.company_name}\n"
        f"Sector: {market.sector}\n"
        f"Industry: {market.industry}\n"
        f"Current Price: {market.current_price} {market.currency or ''}\n"
        f"Market Cap: {market.market_cap}\n"
        "Recent prices (latest 7 trading days):\n"
        f"{market.recent_prices.to_string(index=False)}\n"
        "Task: produce investor-focused analysis."
    )

    analysis = llm_service.analyze(context)
    return TickerAnalysisResponse(market=market, analysis=analysis)


def analyze_news(news_text: str, llm_service: LLMService) -> AnalysisResult:
    cleaned = (news_text or "").strip()
    if not cleaned:
        raise ValueError("News text cannot be empty.")
    return llm_service.analyze(cleaned)


from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st

from src.config import get_settings
from src.services.analysis_service import analyze_news, analyze_ticker
from src.services.llm_service import LLMError, LLMService
from src.services.market_data_service import MarketDataError
from src.services.memo_service import build_research_memo


st.set_page_config(page_title="Financial AI Research Assistant", page_icon=":chart_with_upwards_trend:", layout="wide")


def _sentiment_badge(sentiment: str) -> str:
    sentiment = sentiment.lower()
    if sentiment == "positive":
        return "Positive"
    if sentiment == "negative":
        return "Negative"
    return "Neutral"


def _format_market_cap(value: int | None) -> str:
    if value is None or value <= 0:
        return "Unavailable"
    if value >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:.2f}T"
    if value >= 1_000_000_000:
        return f"${value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"${value / 1_000_000:.2f}M"
    return f"${value:,}"


@st.cache_resource
def _get_llm_service() -> LLMService:
    settings = get_settings()
    return LLMService(api_key=settings.openai_api_key, model=settings.openai_model)


def main() -> None:
    st.title("Financial AI Research Assistant")
    st.write(
        "Demo workflow for hedge-fund style research. Provide a ticker or paste news text to get a concise "
        "investor-oriented analysis."
    )

    mode = st.radio("Input Type", options=["Ticker", "News Text"], horizontal=True)
    if mode == "Ticker":
        user_input = st.text_input("Ticker Symbol", placeholder="AAPL")
    else:
        user_input = st.text_area("Financial News Text", placeholder="Paste earnings or macro news here...", height=180)

    if st.button("Run Analysis", type="primary"):
        if not user_input or not user_input.strip():
            st.error("Please provide valid input before running analysis.")
            return

        try:
            llm_service = _get_llm_service()
        except LLMError as exc:
            st.error(str(exc))
            st.info("Create a `.env` file from `.env.example` and set your API key.")
            return

        with st.spinner("Analyzing..."):
            try:
                if mode == "Ticker":
                    result = analyze_ticker(user_input, llm_service)

                    market = result.market
                    company_label = market.company_name if market.company_name and market.company_name != "N/A" else market.ticker

                    st.subheader(f"Market Snapshot: {company_label} ({market.ticker})")
                    col1, col2, col3 = st.columns(3)
                    if market.current_price is not None:
                        price_label = f"${market.current_price:.2f}"
                        if market.currency and market.currency != "USD":
                            price_label = f"{price_label} {market.currency}"
                    else:
                        price_label = "Unavailable"
                    col1.metric("Current Price", price_label)
                    col2.metric("Market Cap", _format_market_cap(market.market_cap))
                    col3.metric(
                        "Data Status",
                        "Available" if not market.recent_prices.empty else "Limited",
                    )

                    profile_bits = []
                    if market.sector and market.sector != "N/A":
                        profile_bits.append(f"Sector: {market.sector}")
                    if market.industry and market.industry != "N/A":
                        profile_bits.append(f"Industry: {market.industry}")
                    if profile_bits:
                        st.caption(" | ".join(profile_bits))

                    if market.recent_prices.empty:
                        st.info("Market data currently unavailable from provider")
                    else:
                        st.dataframe(market.recent_prices, use_container_width=True, hide_index=True)
                    analysis = result.analysis
                else:
                    analysis = analyze_news(user_input, llm_service)
            except (MarketDataError, ValueError) as exc:
                st.error(str(exc))
                return
            except LLMError as exc:
                st.error(str(exc))
                return
            except Exception:
                st.error("Unexpected error during analysis. Please try again.")
                return

        st.subheader("AI Analyst Output")
        st.markdown("### 1) Executive Summary")
        st.write(analysis.summary)

        st.markdown("### 2) Sentiment")
        st.write(_sentiment_badge(analysis.sentiment))

        st.markdown("### 3) Key Investment Insights")
        for item in analysis.key_insights:
            st.markdown(f"- {item}")

        st.markdown("### 4) Risks / Watch Items")
        for item in analysis.risks:
            st.markdown(f"- {item}")

        st.markdown("### 5) Bottom Line")
        st.write(analysis.conclusion)

        st.session_state["last_analysis"] = analysis
        st.session_state["last_mode"] = mode
        st.session_state["last_input"] = user_input
        st.session_state["last_market"] = result.market if mode == "Ticker" else None

    if "last_analysis" in st.session_state:
        memo_text = build_research_memo(
            analysis=st.session_state["last_analysis"],
            mode=st.session_state.get("last_mode", "N/A"),
            raw_input=st.session_state.get("last_input", ""),
            market=st.session_state.get("last_market"),
        )
        now_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        st.download_button(
            label="Generate Research Memo",
            data=memo_text,
            file_name=f"research_memo_{now_tag}.md",
            mime="text/markdown",
            use_container_width=False,
        )


if __name__ == "__main__":
    main()

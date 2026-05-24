# AI Financial Research Assistant

## 1) Project Overview

AI Financial Research Assistant is a small Streamlit demo that simulates a hedge-fund-style first-pass research workflow.

Users can analyze either:

- A stock ticker (for quick market context)
- Pasted financial news text (for event-driven analysis)

The app produces a structured analyst-style output:

1. Executive Summary
2. Sentiment
3. Key Investment Insights
4. Risks / Watch Items
5. Bottom Line

## 2) Why This Project

This project demonstrates practical AI workflow design for investment research: turning noisy financial inputs into concise, decision-oriented outputs.

It is built as a portfolio project for AI analyst, AI application, fintech, and research-oriented software roles, with emphasis on usability, structured output, and graceful failure handling.

## 3) Features

- Dual input modes:
  - `Ticker` mode pulls basic company and recent price data via `yfinance`
  - `News Text` mode analyzes pasted financial text directly
- Structured LLM output with investor-oriented sections
- Internal memo export via **Generate Research Memo** (`.md`)
- Graceful fallback behavior when market data is missing:
  - Analysis still runs
  - UI clearly indicates when provider market data is unavailable
- Defensive parsing and validation for LLM responses

## 4) Tech Stack

- Python
- Streamlit
- OpenAI API
- yfinance
- pandas
- python-dotenv

## 5) Project Structure

```text
.
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── __init__.py
    ├── config.py
    └── services/
        ├── __init__.py
        ├── analysis_service.py
        ├── llm_service.py
        ├── market_data_service.py
        └── memo_service.py
```

## 6) Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
pip install -r requirements.txt
```

4. Configure environment variables:

```powershell
Copy-Item .env.example .env
```

5. Set API credentials in `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

## 7) Run Locally

```powershell
streamlit run app.py
```

## 8) Example Use Cases

- Pre-market triage: Paste an earnings headline or macro update and get a quick investment brief
- Single-name check: Enter `AAPL`, `MSFT`, `NVDA`, or `TSLA` to combine recent pricing context with analyst-style output
- Internal handoff: Export output as a short research memo for team discussion

## 9) Portfolio Highlights

- Demonstrates an end-to-end AI application flow: user input, data retrieval, LLM analysis, structured parsing, and memo export.
- Shows fintech-oriented product thinking through risk/watch-item framing and investor-style output sections.
- Handles missing market data explicitly instead of failing silently, which is important for external API reliability.

## 10) Current Limitations

- This is a demo, not a production research platform
- Data quality depends on third-party providers (`yfinance`) and API availability
- LLM outputs are structured and constrained, but still probabilistic
- No persistent storage, authentication, audit trail, or portfolio-level analytics

## 11) Future Improvements

- Add source linking or citation snippets for analyzed news
- Add scenario comparison (base, bull, bear) and confidence scoring
- Add simple watchlist support and memo history
- Add lightweight test coverage for service-level reliability

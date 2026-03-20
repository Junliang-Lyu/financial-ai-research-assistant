from __future__ import annotations

import json
import re
from dataclasses import dataclass

from openai import OpenAI


class LLMError(Exception):
    pass


@dataclass
class AnalysisResult:
    summary: str
    sentiment: str
    key_insights: list[str]
    risks: list[str]
    conclusion: str


class LLMService:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise LLMError("OPENAI_API_KEY is missing. Add it to your .env file.")
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def analyze(self, content: str) -> AnalysisResult:
        prompt = (
            "You are an AI analyst writing for an investment team.\n"
            "Style requirements: concise, professional, investor-oriented, and specific.\n"
            "Avoid generic phrasing, educational tone, and filler.\n"
            "Prioritize business relevance, operating/financial momentum, risks, and likely market implications.\n"
            "Use clear, decision-useful statements.\n"
            "Return ONLY valid JSON with keys:\n"
            'summary (string), sentiment (positive|neutral|negative), key_insights (array of 3 strings), '
            'risks (array of 3 strings), conclusion (string).\n'
            "Length guidance:\n"
            "- summary: 2-3 sentences\n"
            "- key_insights: exactly 3 bullets, each <= 20 words\n"
            "- risks: exactly 3 bullets, each <= 20 words\n"
            "- conclusion: 1-2 sentences with investor implication\n"
            "Keep wording factual and business-oriented; avoid hype.\n"
            "Input:\n"
            f"{content}"
        )

        try:
            response = self._client.responses.create(
                model=self._model,
                input=prompt,
                temperature=0.2,
            )
            raw_text = response.output_text.strip()
        except Exception as exc:
            raise LLMError("LLM request failed. Check API key, model, and network access.") from exc

        data = self._parse_json(raw_text)
        return self._validate(data)

    def _parse_json(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise LLMError("Model output was not valid JSON.")
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as exc:
                raise LLMError("Unable to parse model output JSON.") from exc

    def _validate(self, payload: dict) -> AnalysisResult:
        summary = str(payload.get("summary", "")).strip()
        sentiment = str(payload.get("sentiment", "neutral")).strip().lower()
        key_insights = [str(x).strip() for x in payload.get("key_insights", []) if str(x).strip()]
        risks = [str(x).strip() for x in payload.get("risks", []) if str(x).strip()]
        conclusion = str(payload.get("conclusion", "")).strip()

        if sentiment not in {"positive", "neutral", "negative"}:
            sentiment = "neutral"

        if not summary:
            summary = "No summary was generated."
        if not conclusion:
            conclusion = "No investor conclusion was generated."

        key_insights = (key_insights + ["Not available."] * 3)[:3]
        risks = (risks + ["Not available."] * 3)[:3]

        return AnalysisResult(
            summary=summary,
            sentiment=sentiment,
            key_insights=key_insights,
            risks=risks,
            conclusion=conclusion,
        )

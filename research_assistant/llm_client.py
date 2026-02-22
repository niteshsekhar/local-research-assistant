from __future__ import annotations

import json
import re
from typing import Any

import requests

from .config import Settings
from .models import PaperInsight, ParsedPaper


class LocalLLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _chat(self, prompt: str) -> str:
        url = f"{self.settings.llm_api_base.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Return concise, accurate analysis in JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 1200,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(url, json=payload, headers=headers, timeout=90)
        response.raise_for_status()
        body = response.json()
        return body["choices"][0]["message"]["content"]

    def check_server(self) -> dict[str, Any]:
        base = self.settings.llm_api_base.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }
        result: dict[str, Any] = {
            "ok": False,
            "base_url": base,
            "model": self.settings.llm_model,
            "models_endpoint": f"{base}/models",
            "chat_endpoint": f"{base}/chat/completions",
        }
        try:
            models_resp = requests.get(f"{base}/models", headers=headers, timeout=15)
            result["models_status_code"] = models_resp.status_code
            if models_resp.ok:
                data = models_resp.json()
                available = [item.get("id", "") for item in data.get("data", [])]
                result["available_models"] = available
            else:
                result["models_error"] = models_resp.text[:300]
        except Exception as exc:
            result["models_error"] = str(exc)

        try:
            _ = self._chat("Return exactly: OK")
            result["chat_ok"] = True
        except Exception as exc:
            result["chat_ok"] = False
            result["chat_error"] = str(exc)

        target_model = self.settings.llm_model
        available_models = result.get("available_models", [])
        model_known = (not available_models) or (target_model in available_models)
        result["model_found"] = model_known
        result["ok"] = bool(result.get("chat_ok")) and model_known
        return result

    def analyze_paper(self, parsed: ParsedPaper) -> PaperInsight:
        eq_sample = "\n".join(parsed.equation_candidates[:15])
        text_chunks = self._build_text_chunks(parsed.full_text)
        parsed_candidates: list[dict[str, Any]] = []

        for index, chunk in enumerate(text_chunks, start=1):
            prompt = f"""
Analyze this paper excerpt and equation candidates.
This is chunk {index}/{len(text_chunks)}.

Return strict JSON with keys:
- summary (string)
- innovations (array of 3-6 important innovations)
- contributions (array of 3-6 bullet-style strings)
- method_type (one of: scaling law, optimization, RL, architecture, systems, data, theory, other)
- training_info (array of 3-8 items including hyperparameters, objectives, losses, schedule, datasets if present)
- architecture (string, describe model/system architecture if present, else 'Not specified')
- pros (array of 2-5 strengths)
- cons (array of 2-5 limitations)
- next_steps (array of 3-6 concrete next experiments/extensions)
- research_ideas (array of exactly 5 concrete research ideas)

Paper text:
{chunk}

Equation candidates:
{eq_sample}
""".strip()

            try:
                raw = self._chat(prompt)
                parsed_candidates.append(self._safe_json(raw))
            except Exception:
                continue

        parsed_json = self._merge_candidates(parsed_candidates)
        if not parsed_json:
            parsed_json = self._fallback_analysis(parsed)

        contributions = parsed_json.get("contributions") or []
        innovations = parsed_json.get("innovations") or []
        training_info = parsed_json.get("training_info") or []
        pros = parsed_json.get("pros") or []
        cons = parsed_json.get("cons") or []
        next_steps = parsed_json.get("next_steps") or []
        ideas = parsed_json.get("research_ideas") or []
        normalized_ideas = [str(x).strip() for x in ideas if str(x).strip()]
        while len(normalized_ideas) < 5:
            normalized_ideas.append(
                f"Investigate an extension of this method with ablation focus #{len(normalized_ideas) + 1}."
            )

        return PaperInsight(
            summary=str(parsed_json.get("summary", "")).strip(),
            innovations=[str(x).strip() for x in innovations if str(x).strip()][:6],
            contributions=[str(x) for x in contributions][:6],
            method_type=str(parsed_json.get("method_type", "other")),
            training_info=[str(x).strip() for x in training_info if str(x).strip()][:8],
            architecture=str(parsed_json.get("architecture", "Not specified")).strip() or "Not specified",
            pros=[str(x).strip() for x in pros if str(x).strip()][:5],
            cons=[str(x).strip() for x in cons if str(x).strip()][:5],
            next_steps=[str(x).strip() for x in next_steps if str(x).strip()][:6],
            research_ideas=normalized_ideas[:5],
        )

    @staticmethod
    def _build_text_chunks(text: str, max_chunk_chars: int = 2600) -> list[str]:
        clean = text.strip()
        if not clean:
            return [""]
        if len(clean) <= max_chunk_chars:
            return [clean]
        middle_start = max((len(clean) // 2) - (max_chunk_chars // 2), 0)
        chunks = [
            clean[:max_chunk_chars],
            clean[middle_start : middle_start + max_chunk_chars],
            clean[-max_chunk_chars:],
        ]
        deduped: list[str] = []
        seen = set()
        for item in chunks:
            marker = item[:120]
            if marker not in seen:
                seen.add(marker)
                deduped.append(item)
        return deduped

    @staticmethod
    def _merge_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not candidates:
            return {}

        def list_field(name: str, limit: int) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for item in candidates:
                for value in item.get(name, []) or []:
                    value_str = str(value).strip()
                    if not value_str or value_str in seen:
                        continue
                    seen.add(value_str)
                    out.append(value_str)
                    if len(out) >= limit:
                        return out
            return out

        summary = ""
        method_type = "other"
        architecture = "Not specified"
        for item in candidates:
            if not summary and str(item.get("summary", "")).strip():
                summary = str(item.get("summary", "")).strip()
            if method_type == "other" and str(item.get("method_type", "other")).strip():
                method_type = str(item.get("method_type", "other")).strip()
            arch = str(item.get("architecture", "")).strip()
            if arch and arch.lower() != "not specified":
                architecture = arch

        return {
            "summary": summary,
            "innovations": list_field("innovations", 6),
            "contributions": list_field("contributions", 6),
            "method_type": method_type or "other",
            "training_info": list_field("training_info", 8),
            "architecture": architecture,
            "pros": list_field("pros", 5),
            "cons": list_field("cons", 5),
            "next_steps": list_field("next_steps", 6),
            "research_ideas": list_field("research_ideas", 5),
        }

    @staticmethod
    def _safe_json(text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise json.JSONDecodeError("Empty response", "", 0)
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?", "", text).strip()
            text = re.sub(r"```$", "", text).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            obj = LocalLLMClient._extract_first_json_object(text)
            if obj is not None:
                return json.loads(obj)
            raise

    @staticmethod
    def _extract_first_json_object(text: str) -> str | None:
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for idx, char in enumerate(text[start:], start=start):
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : idx + 1]
        return None

    @staticmethod
    def _fallback_analysis(parsed: ParsedPaper) -> dict[str, Any]:
        lines = [line.strip() for line in parsed.full_text.splitlines() if line.strip()]
        summary = " ".join(lines[:3])[:600] if lines else parsed.file_name
        contributions = lines[3:8] if len(lines) > 3 else [f"Initial extraction from {parsed.file_name}."]
        ideas = [
            "Test the core method under different dataset scales.",
            "Compare with a stronger modern baseline using equal compute.",
            "Run robustness analysis for domain shift and noisy inputs.",
            "Perform component-level ablations to isolate gains.",
            "Explore a smaller and faster variant for deployment.",
        ]
        return {
            "summary": summary,
            "innovations": contributions[:4],
            "contributions": contributions[:6],
            "method_type": "other",
            "training_info": [
                "Loss function, optimizer, and exact hyperparameters were not reliably extracted.",
                "Re-index with a stronger local model to recover detailed training settings.",
            ],
            "architecture": "Not specified from fallback extraction.",
            "pros": [
                "Pipeline preserves ingestion even if model output is malformed.",
                "Extracted text still supports search and retrieval.",
            ],
            "cons": [
                "Fine-grained training setup may be incomplete in fallback mode.",
                "Architecture details may require stronger model reasoning.",
            ],
            "next_steps": [
                "Run extraction again after confirming local LLM stability.",
                "Add paper-specific regex extraction for optimizer/loss mentions.",
                "Compare claims against baseline papers in the same domain.",
            ],
            "research_ideas": ideas,
        }

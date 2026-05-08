from dataclasses import dataclass
from typing import Any, Dict, Optional


ROUTER_SYSTEM = """You classify HotpotQA question type.

Return "comparison" if the question compares two or more named entities, asks which is bigger,
earlier, first, more, fewer, or asks which of two entities satisfies an attribute.

Return "bridge" if the question requires finding an intermediate entity or following a relation
chain before answering.

Return ONLY valid JSON:
{"question_type":"bridge | comparison", "reason":"brief reason"}
"""


@dataclass
class RouteDecision:
    question_type: str
    reason: str = ""
    source: str = "router"


class QuestionTypeRouter:
    """Route examples to bridge or comparison agents."""

    def __init__(self, llm=None, prefer_dataset_type: bool = True):
        self.llm = llm
        self.prefer_dataset_type = prefer_dataset_type

    def route(self, example: Dict[str, Any]) -> RouteDecision:
        dataset_type = str(example.get("type", "")).strip().lower()
        if self.prefer_dataset_type and dataset_type in {"bridge", "comparison"}:
            return RouteDecision(
                question_type=dataset_type,
                reason="Used HotpotQA dataset type field.",
                source="dataset",
            )

        if not self.llm or not self.llm.enabled:
            return RouteDecision(
                question_type=self._heuristic_route(example.get("question", "")),
                reason="Used heuristic fallback because LLM router is not configured.",
                source="heuristic",
            )

        data = self.llm.chat_json(
            ROUTER_SYSTEM,
            f"Question: {example.get('question', '')}",
        )
        question_type = str(data.get("question_type", "bridge")).strip().lower()
        if question_type not in {"bridge", "comparison"}:
            question_type = "bridge"
        return RouteDecision(
            question_type=question_type,
            reason=str(data.get("reason", "")).strip(),
            source="llm",
        )

    def _heuristic_route(self, question: str) -> str:
        text = question.lower()
        comparison_markers = [
            " or ",
            " more ",
            " fewer ",
            " less ",
            " earlier",
            " later",
            " first",
            " larger",
            " smaller",
            " older",
            " younger",
        ]
        return "comparison" if any(marker in text for marker in comparison_markers) else "bridge"

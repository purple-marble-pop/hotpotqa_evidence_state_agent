from typing import Any, Dict

from .memory import ComparisonMemory
from .schema import ComparisonSchema


COMPARISON_PLANNER_SYSTEM = """You are a planner for HotpotQA comparison questions.

Build a Comparison Schema. Do not answer the question.

Schema fields:
- entity_a: first compared entity.
- entity_b: second compared entity.
- comparison_attribute: the shared attribute or property to compare.
- comparison_type: one of boolean_attribute, numeric_count, date_order, category, other.
- answer_rule: how to select the answer after values are extracted.
- state: always "schema_created".

Rules:
- Preserve full entity names from the question.
- For "who/which has more/fewer", use comparison_type "numeric_count".
- For "who was born first/earlier/later", use comparison_type "date_order".
- For "which entity satisfies property P", use comparison_type "boolean_attribute".
- Keep the comparison_attribute evidence-checkable.

Return ONLY valid JSON:
{
  "entity_a": "...",
  "entity_b": "...",
  "comparison_attribute": "...",
  "comparison_type": "boolean_attribute | numeric_count | date_order | category | other",
  "answer_rule": "...",
  "state": "schema_created"
}
"""


class ComparisonPlanner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, memory: ComparisonMemory) -> ComparisonSchema:
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        data: Dict[str, Any] = self.llm.chat_json(
            COMPARISON_PLANNER_SYSTEM,
            f"Current comparison memory:\n\n{memory.compact_context()}\nBuild the schema.",
        )
        return ComparisonSchema.from_planner_json(data)

from hotpotqa_agent.core.llm import LLMClient
from hotpotqa_agent.core.search import ContextSearchTool, format_evidence_list

from .schema import ComparisonSchema, ComparisonState, EntityComparisonValue


VALUE_EXTRACT_SYSTEM = """You extract the comparison value for one entity.

Given an entity, comparison attribute, comparison type, and evidence, extract the value needed for comparison.
Use only evidence. If the evidence states the entity satisfies a boolean attribute, value should be "true".
If evidence does not support the attribute, value should be "false" or "unknown".

Return ONLY valid JSON:
{
  "value": "...",
  "evidence_ref": "title[sent_id]",
  "evidence_text": "supporting sentence",
  "confidence": "low | medium | high"
}
"""


COMPARE_SYSTEM = """You compare two extracted entity values and choose the final answer.

Use the answer_rule and comparison_type.
For yes/no or boolean questions, return exactly "yes" or "no".
For common-value questions, return the shared value, not an entity name.
Return one of the entity names only when the question asks which entity satisfies a condition.

Return ONLY valid JSON:
{
  "answer": "...",
  "reason": "brief comparison reason"
}
"""


class ComparisonTools:
    def __init__(self, example, llm: LLMClient, top_k: int = 5):
        self.search_tool = ContextSearchTool(example)
        self.llm = llm
        self.top_k = top_k

    def _search_entity_attribute(self, entity: str, attribute: str):
        queries = [
            f"{entity} {attribute}",
            entity,
            attribute,
        ]
        evidence_by_ref = {}
        for query in queries:
            for item in self.search_tool.search(query, top_k=self.top_k):
                evidence_by_ref[item.ref] = item
        return sorted(evidence_by_ref.values(), key=lambda item: item.score, reverse=True)[
            : self.top_k * 2
        ]

    def collect_and_extract(self, schema: ComparisonSchema) -> ComparisonSchema:
        schema.entity_a_value = self.extract_value(schema, schema.entity_a)
        schema.entity_b_value = self.extract_value(schema, schema.entity_b)
        schema.state = ComparisonState.VALUES_EXTRACTED
        return schema

    def extract_value(self, schema: ComparisonSchema, entity: str) -> EntityComparisonValue:
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        evidence = self._search_entity_attribute(entity, schema.comparison_attribute)
        result = self.llm.chat_json(
            VALUE_EXTRACT_SYSTEM,
            (
                f"Entity: {entity}\n"
                f"Comparison attribute: {schema.comparison_attribute}\n"
                f"Comparison type: {schema.comparison_type}\n"
                f"Evidence:\n{format_evidence_list(evidence)}"
            ),
        )
        return EntityComparisonValue(
            entity=entity,
            value=str(result.get("value", "")).strip(),
            evidence_ref=str(result.get("evidence_ref", "")).strip(),
            evidence_text=str(result.get("evidence_text", "")).strip(),
            confidence=str(result.get("confidence", "unknown")).strip(),
        )

    def compare(self, schema: ComparisonSchema) -> ComparisonSchema:
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        result = self.llm.chat_json(
            COMPARE_SYSTEM,
            (
                f"Question comparison schema:\n{schema.to_dict()}\n\n"
                f"Entity A value: {schema.entity_a_value}\n"
                f"Entity B value: {schema.entity_b_value}\n"
            ),
        )
        schema.final_answer = self._normalize_boolean_answer(
            schema,
            str(result.get("answer", "")).strip(),
        )
        schema.reason = str(result.get("reason", "")).strip()
        schema.state = ComparisonState.COMPARED
        return schema

    def _normalize_boolean_answer(self, schema: ComparisonSchema, answer: str) -> str:
        if schema.comparison_type != "boolean_attribute":
            return answer
        rule = schema.answer_rule.lower()
        boolean_rule = any(
            marker in rule
            for marker in (
                "both",
                "same",
                "return true",
                "return false",
                "answer yes",
                "yes",
                "no",
            )
        )
        if not boolean_rule:
            return answer
        a_value = (schema.entity_a_value.value if schema.entity_a_value else "").lower()
        b_value = (schema.entity_b_value.value if schema.entity_b_value else "").lower()
        if a_value in {"true", "false"} and b_value in {"true", "false"}:
            return "yes" if a_value == "true" and b_value == "true" else "no"
        if answer.lower() in {"yes", "no"}:
            return answer.lower()
        return answer

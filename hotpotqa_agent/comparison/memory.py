from dataclasses import dataclass, field
from typing import Any, Dict, List

from hotpotqa_agent.core.state import Evidence

from .schema import ComparisonSchema


@dataclass
class ComparisonMemory:
    question: str
    evidence_chain: List[Evidence] = field(default_factory=list)
    schemas: List[ComparisonSchema] = field(default_factory=list)
    trace: List[Dict[str, Any]] = field(default_factory=list)

    def add_schema(self, schema: ComparisonSchema) -> None:
        self.schemas.append(schema)
        self.trace.append({"event": "schema_created", "schema": schema.to_dict()})

    def add_evidence(self, evidence_items: List[Evidence]) -> None:
        for item in evidence_items:
            if item not in self.evidence_chain:
                self.evidence_chain.append(item)

    def compact_context(self) -> str:
        recent = "\n".join(
            (
                f"- {schema.entity_a} vs {schema.entity_b}; "
                f"attribute={schema.comparison_attribute}; state={schema.state.value}"
            )
            for schema in self.schemas[-3:]
        ) or "No comparison schema yet."
        return f"Question: {self.question}\nRecent comparison schemas:\n{recent}\n"

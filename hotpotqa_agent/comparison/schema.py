from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ComparisonState(str, Enum):
    SCHEMA_CREATED = "schema_created"
    EVIDENCE_COLLECTED = "evidence_collected"
    VALUES_EXTRACTED = "values_extracted"
    COMPARED = "compared"
    FINISHED = "finished"


@dataclass
class EntityComparisonValue:
    entity: str
    value: str = ""
    evidence_ref: str = ""
    evidence_text: str = ""
    confidence: str = "unknown"


@dataclass
class ComparisonSchema:
    entity_a: str
    entity_b: str
    comparison_attribute: str
    comparison_type: str
    answer_rule: str
    state: ComparisonState = ComparisonState.SCHEMA_CREATED
    entity_a_value: Optional[EntityComparisonValue] = None
    entity_b_value: Optional[EntityComparisonValue] = None
    final_answer: str = ""
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        return data

    @classmethod
    def from_planner_json(cls, data: Dict[str, Any]) -> "ComparisonSchema":
        return cls(
            entity_a=str(data.get("entity_a", "")).strip(),
            entity_b=str(data.get("entity_b", "")).strip(),
            comparison_attribute=str(data.get("comparison_attribute", "")).strip(),
            comparison_type=str(data.get("comparison_type", "attribute")).strip(),
            answer_rule=str(data.get("answer_rule", "")).strip(),
            state=ComparisonState(data.get("state", ComparisonState.SCHEMA_CREATED.value)),
        )

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class BridgeState(str, Enum):

    SCHEMA_CREATED = "schema_created"
    CANDIDATE_FOUND = "candidate_found"
    CANDIDATE_NOT_FOUND = "candidate_not_found"
    VERIFIED = "verified"
    VERIFICATION_FAILED = "verification_failed"
    FINISHED = "finished"


class AttributeStatus(str, Enum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    FAILED = "failed"
    RELAXED = "relaxed"


@dataclass
class AttributeConstraint:

    description: str
    status: AttributeStatus = AttributeStatus.UNVERIFIED
    constraint_type: str = "hard"
    evidence_ref: str = ""
    evidence_text: str = ""
    retry_count: int = 0

    def mark_verified(self, evidence_ref: str = "", evidence_text: str = "") -> None:
        self.status = AttributeStatus.VERIFIED
        self.evidence_ref = evidence_ref
        self.evidence_text = evidence_text

    def mark_failed(self) -> None:
        self.status = AttributeStatus.FAILED
        self.retry_count += 1

    def relax(self) -> None:
        self.status = AttributeStatus.RELAXED
        self.constraint_type = "soft"


@dataclass
class CandidateEntity:
    """A searched entity that still needs attribute verification."""

    name: str
    evidence_ref: str = ""
    evidence_text: str = ""
    score: float = 0.0
    rejected_reason: str = ""


@dataclass
class BridgeEntitySchema:
    """Planner output for the current intermediate entity."""

    object: str
    attributes: List[AttributeConstraint] = field(default_factory=list)
    next_relation: str = ""
    state: BridgeState = BridgeState.SCHEMA_CREATED
    candidate_entities: List[CandidateEntity] = field(default_factory=list)
    confirmed_entity: Optional[CandidateEntity] = None
    hidden_bridge_notes: List[str] = field(default_factory=list)
    final_answer: str = ""

    def hard_attribute_texts(self) -> List[str]:
        return [
            item.description
            for item in self.attributes
            if item.constraint_type == "hard" and item.status != AttributeStatus.VERIFIED
        ]

    def relaxed_attribute_texts(self) -> List[str]:
        return [
            item.description
            for item in self.attributes
            if item.constraint_type == "soft" or item.status == AttributeStatus.RELAXED
        ]

    def all_hard_attributes_verified(self) -> bool:
        return all(
            item.status == AttributeStatus.VERIFIED
            for item in self.attributes
            if item.constraint_type == "hard"
        )

    def mark_failed_attributes_relaxed(self) -> None:
        for item in self.attributes:
            if item.status == AttributeStatus.FAILED:
                item.relax()

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["state"] = self.state.value
        for item in data["attributes"]:
            item["status"] = item["status"].value
            evidence_ref = item.pop("evidence_ref", "")
            evidence_text = item.pop("evidence_text", "")
            item["evidence"] = (
                {"ref": evidence_ref, "text": evidence_text}
                if evidence_ref or evidence_text
                else None
            )
        return data

    @classmethod
    def from_planner_json(cls, data: Dict[str, Any]) -> "BridgeEntitySchema":
        attrs = []
        for item in data.get("attributes", []):
            if isinstance(item, dict):
                attrs.append(
                    AttributeConstraint(
                        description=str(item.get("description", "")).strip(),
                        status=AttributeStatus(
                            item.get("status", AttributeStatus.UNVERIFIED.value)
                        ),
                        constraint_type=str(item.get("constraint_type", "hard")).strip()
                        or "hard",
                    )
                )
            else:
                attrs.append(AttributeConstraint(description=str(item).strip()))
        return cls(
            object=str(data.get("object", "")).strip(),
            attributes=attrs,
            next_relation=str(data.get("next_relation", "")).strip(),
            state=BridgeState(data.get("state", BridgeState.SCHEMA_CREATED.value)),
        )

    @classmethod
    def for_next_relation(cls, entity: CandidateEntity, relation: str) -> "BridgeEntitySchema":
        """Create the next schema for an already verified bridge entity."""
        relation = relation.strip()
        return cls(
            object=f"answer value for relation: {relation}",
            attributes=[
                AttributeConstraint(
                    description=f"{relation} of {entity.name}",
                )
            ],
            next_relation="",
            state=BridgeState.SCHEMA_CREATED,
        )

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List

from hotpotqa_agent.core.state import Evidence, KnownFact

from .schema import BridgeEntitySchema, CandidateEntity


@dataclass
class EvidenceMemory:
    """Structured memory for attribute-constrained bridge reasoning."""

    question: str
    known_facts: List[KnownFact] = field(default_factory=list)
    evidence_chain: List[Evidence] = field(default_factory=list)
    bridge_schemas: List[BridgeEntitySchema] = field(default_factory=list)
    confirmed_entities: List[CandidateEntity] = field(default_factory=list)
    rejected_candidates: List[CandidateEntity] = field(default_factory=list)
    reasoning_trace: List[Dict[str, Any]] = field(default_factory=list)

    def add_schema(self, schema: BridgeEntitySchema) -> None:
        self.bridge_schemas.append(schema)
        self.reasoning_trace.append({"event": "schema_created", "schema": schema.to_dict()})

    def add_confirmed_entity(self, schema: BridgeEntitySchema) -> None:
        if schema.confirmed_entity:
            if (
                self.confirmed_entities
                and self.confirmed_entities[-1].name == schema.confirmed_entity.name
            ):
                return
            self.confirmed_entities.append(schema.confirmed_entity)
        self.reasoning_trace.append({"event": "entity_verified", "schema": schema.to_dict()})

    def add_rejected_candidate(self, candidate: CandidateEntity, reason: str = "") -> None:
        candidate.rejected_reason = reason
        self.rejected_candidates.append(candidate)
        self.reasoning_trace.append(
            {"event": "candidate_rejected", "candidate": asdict(candidate)}
        )

    def add_evidence(self, evidence_items: List[Evidence]) -> None:
        for item in evidence_items:
            if item not in self.evidence_chain:
                self.evidence_chain.append(item)

    def compact_context(self) -> str:
        confirmed = ", ".join(item.name for item in self.confirmed_entities) or "None"
        schemas = "\n".join(
            (
                f"- object={schema.object}; next_relation={schema.next_relation}; "
                f"state={schema.state.value}; "
                f"confirmed={schema.confirmed_entity.name if schema.confirmed_entity else 'None'}"
            )
            for schema in self.bridge_schemas[-3:]
        ) or "No bridge schemas yet."
        return (
            f"Question: {self.question}\n"
            f"Confirmed bridge entities: {confirmed}\n"
            f"Recent bridge schemas:\n{schemas}\n"
        )

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

@dataclass
class Evidence:
    title: str
    sent_id: Optional[int]
    sentence: str
    score: float = 0.0
    @property
    def ref(self) -> str:
        return self.title if self.sent_id is None else f"{self.title}[{self.sent_id}]"

@dataclass
class KnownFact:
    fact: str
    evidence_ref: str
    solves: str = ""

@dataclass
class AgentStep:
    hop: int
    thought: str
    action_type: str
    action_input: Dict[str, Any]
    observation: str
    extracted_fact: str = ""
    missing_information: str = ""

@dataclass
class ReasoningState:
    question: str
    known_facts: List[KnownFact] = field(default_factory=list)
    evidence_chain: List[Evidence] = field(default_factory=list)
    missing_information: str = "initial question not solved"
    agent_trace: List[AgentStep] = field(default_factory=list)
    answer_ready: bool = False
    final_answer: Optional[str] = None
    confidence: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def compact_context(self) -> str:
        facts = "\n".join(f"- {fact.fact} (source: {fact.evidence_ref})" for fact in self.known_facts) or "No known facts yet."
        return f"Question: {self.question}\nKnown facts:\n{facts}\nMissing information: {self.missing_information}\n"

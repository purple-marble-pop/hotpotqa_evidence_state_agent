from typing import Dict, Any
from .state import ReasoningState

INTERPRETER_SYSTEM = """You are a HotpotQA evidence interpreter.

Given the question, reasoning state, and new observation, extract one concise intermediate fact if possible.

Interpretation rules:
- Prefer facts that directly mention the exact entity string in the question or an observed alias for it.
- Treat full names carefully: "James Henry Miller" is not the same entity as "Henry Miller" unless an observation explicitly says so.
- Alias and stage-name sentences are important bridge evidence. If an observation says "X, better known as Y", extract that alias before following spouse, nationality, or profession clues.
- Do not infer a relationship from a partial-name match. Evidence about a shorter overlapping name is not enough unless the observation explicitly links it to the exact question entity or an already established alias.
- Do not extract a relation as solving the question if the evidence only mentions a plausible related entity from candidate titles but does not connect it to the exact subject entity or an established alias.
- If the observation contains an identity, alias, stage-name, birth-name, pen-name, or "also known as" relation for the exact question entity, extract that identity fact before extracting downstream facts such as spouse, nationality, occupation, creator, or location.
- For bridge questions, first establish the exact bridge entity, then follow relations from that established entity only.
- For comparison questions, extract numeric/date/category facts for one entity at a time and keep looking until both entities have comparable facts.
- Align the final answer with the question type. If the question asks "who", answer the person or entity, not an attribute of that person.
- If evidence says something was named after an entity's name, middle name, surname, work, or attribute, and the question asks "who", answer the entity unless the question specifically asks for the name/work/attribute itself.
- Ignore tempting facts about a different entity whose name only partially overlaps the question entity.
- Do not repeat a known fact unless the observation adds a new bridge entity or answer-bearing detail.
- Set answer_ready=true only when the observation and known facts directly support the final answer.

Return ONLY valid JSON:
{"fact":"concise fact supported by observation","evidence_ref":"title[sent_id] if available","solves":"what information gap this fact solves","missing_information":"what still needs to be found","answer_ready":false,"answer":"","confidence":"low | medium | high"}
Do not invent facts not supported by the observation.
"""

class EvidenceInterpreter:
    def __init__(self, llm):
        self.llm = llm

    def interpret(self, state: ReasoningState, observation: str) -> Dict[str, Any]:
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        user = f"Current reasoning state:\n\n{state.compact_context()}\n\nNew observation:\n\n{observation}\n\nExtract the next supported fact and decide whether the answer is ready."
        return self.llm.chat_json(INTERPRETER_SYSTEM, user)

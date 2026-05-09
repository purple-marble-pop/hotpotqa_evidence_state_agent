from typing import Dict, Any
from .state import ReasoningState

INTERPRETER_SYSTEM = """You are a HotpotQA evidence interpreter.

Given the question, reasoning state, and new observation, extract one concise intermediate fact if possible.

Interpretation rules:
- Do not extract a relation as solving the question if the evidence only mentions a plausible related entity from candidate titles but does not connect it to the exact subject entity or an established alias.
- If the observation contains an identity, alias, stage-name, birth-name, pen-name, or "also known as" relation for the exact question entity, extract that identity fact before extracting downstream facts such as spouse, nationality, occupation, creator, or location.
- For bridge questions, first establish the exact bridge entity, then follow relations from that established entity only.
- For comparison questions, extract numeric/date/category facts for one entity at a time and keep looking until both entities have comparable facts.
- Align the final answer with the question type. If the question asks "who", answer the person or entity, not an attribute of that person.
- If evidence says something was named after an entity's name, middle name, surname, work, or attribute, and the question asks "who", answer the entity unless the question specifically asks for the name/work/attribute itself.


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

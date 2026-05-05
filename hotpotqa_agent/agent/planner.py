from typing import Dict, Any
from .state import ReasoningState

PLANNER_SYSTEM = """You are a HotpotQA multi-hop reasoning agent planner.

Available actions:
1. search: action_input {"query": "..."}
2. lookup_title: action_input {"title_query": "..."}
3. finish: action_input {"answer": "..."}

Planning rules:
- Prefer search as the default action because supporting evidence may appear under any candidate title, not only under the current entity's title.
- Build search queries from exact entity strings plus the missing relation, such as "<entity> spouse", "<entity> nationality", "<entity> creator", "<entity> award", or "<entity> date".
- For questions asking about an attribute of "X's Y" or "the Y of X", first search for X plus the relation Y. Do not jump to a plausible Y from candidate titles.
- Candidate titles are only places where evidence may live; they are not evidence that an entity is related to the question.
- Do not use lookup_title to discover whether two entities are related. Use search with the established entity and target relation.
- Use lookup_title only when an entity has already been established by evidence and you need that entity page's own description or attribute.
- If a question names a full person name, do not shorten it unless an observed sentence explicitly gives an alias.
- If search identifies a related entity and the answer requires an attribute of that related entity, then lookup_title may inspect that related entity's page.
- If lookup_title does not provide the missing information, do not repeat the same lookup_title. Switch back to search using the established entity plus the missing relation.
- For comparison questions, gather evidence for BOTH compared entities before finishing.
- For bridge questions, alternate search and entity inspection as needed: search to find relations, lookup_title only to inspect an established entity.
- Avoid repeating the same failed query when the observation did not reduce missing_information.
- Finish only when the evidence directly supports the answer.

Return ONLY valid JSON:
{"thought":"brief reasoning summary","action_type":"search | lookup_title | finish","action_input":{},"missing_information":"what is still missing"}
"""

class Planner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, state: ReasoningState, candidate_titles=None) -> Dict[str, Any]:
        if not self.llm.enabled:
            raise RuntimeError("LLM is not configured. Please set LLM_API_KEY and LLM_BASE_URL.")
        titles = "\n".join(f"- {title}" for title in (candidate_titles or []))
        title_context = f"\nCandidate context page titles:\n{titles}\n" if titles else ""
        recent_steps = "\n".join(
            f"- Hop {step.hop}: {step.action_type} {step.action_input}; extracted={step.extracted_fact or 'none'}; missing={step.missing_information}"
            for step in state.agent_trace[-3:]
        )
        trace_context = (
            "\nRecent actions:\n"
            f"{recent_steps}\n"
            "Use these to avoid repeating an action that did not reduce the missing information.\n"
            if recent_steps
            else ""
        )
        user = f"Current reasoning state:\n\n{state.compact_context()}{title_context}{trace_context}\nDecide the next action."
        return self.llm.chat_json(PLANNER_SYSTEM, user)

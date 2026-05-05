from typing import Any

from .state import AgentStep, KnownFact, ReasoningState
from .tools import ContextSearchTool, format_evidence_list
from .planner import Planner
from .interpreter import EvidenceInterpreter
from .llm import LLMClient


def normalize_action_input(action_type: str, action_input: Any) -> dict:
    if isinstance(action_input, dict):
        return action_input
    if action_input is None:
        return {}
    text = str(action_input)
    if action_type == "lookup_title":
        return {"title_query": text}
    if action_type == "finish":
        return {"answer": text}
    return {"query": text}


def normalize_action_type(action_type: Any) -> str:
    text = str(action_type or "search").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"lookup", "lookup_title", "title_lookup"}:
        return "lookup_title"
    if text == "finish":
        return "finish"
    return "search"


class HotpotEvidenceStateAgent:
    """Evidence-state driven multi-hop Agent for one HotpotQA sample."""

    def __init__(self, llm: LLMClient | None = None, max_hops: int = 4, top_k: int = 5):
        self.llm = llm or LLMClient()
        self.max_hops = max_hops
        self.top_k = top_k
        self.planner = Planner(self.llm)
        self.interpreter = EvidenceInterpreter(self.llm)

    def run(self, example) -> ReasoningState:
        state = ReasoningState(question=example["question"])
        tool = ContextSearchTool(example)

        for hop in range(1, self.max_hops + 1):
            plan = self.planner.plan(state, candidate_titles=tool.list_titles())
            action_type = normalize_action_type(plan.get("action_type", "search"))
            action_input = normalize_action_input(action_type, plan.get("action_input", {}))
            thought = plan.get("thought", "")
            state.missing_information = plan.get("missing_information", state.missing_information)

            if action_type == "finish":
                state.answer_ready = True
                state.final_answer = action_input.get("answer", "")
                state.confidence = "medium"
                state.agent_trace.append(
                    AgentStep(
                        hop=hop,
                        thought=thought,
                        action_type=action_type,
                        action_input=action_input,
                        observation="Agent decided to finish.",
                        missing_information=state.missing_information,
                    )
                )
                break

            if action_type == "lookup_title":
                evidence = tool.lookup_title(action_input.get("title_query", ""), top_k=self.top_k)
            else:
                evidence = tool.search(action_input.get("query", state.question), top_k=self.top_k)

            observation = format_evidence_list(evidence)
            interpretation = self.interpreter.interpret(state, observation)

            fact_text = interpretation.get("fact", "")
            evidence_ref = interpretation.get("evidence_ref", "")
            solves = interpretation.get("solves", "")
            if fact_text:
                state.known_facts.append(
                    KnownFact(fact=fact_text, evidence_ref=evidence_ref, solves=solves)
                )

            for item in evidence:
                if item not in state.evidence_chain:
                    state.evidence_chain.append(item)

            state.missing_information = interpretation.get(
                "missing_information", state.missing_information
            )
            state.answer_ready = bool(interpretation.get("answer_ready", False))
            state.confidence = interpretation.get("confidence", state.confidence)
            state.agent_trace.append(
                AgentStep(
                    hop=hop,
                    thought=thought,
                    action_type=action_type,
                    action_input=action_input,
                    observation=observation,
                    extracted_fact=fact_text,
                    missing_information=state.missing_information,
                )
            )

            if state.answer_ready:
                state.final_answer = interpretation.get("answer", "")
                break

        return state


def print_state(state: ReasoningState):
    print("\n" + "=" * 100)
    print("FINAL RESULT")
    print("=" * 100)
    print(f"Question: {state.question}")
    print(f"Answer: {state.final_answer}")
    print(f"Confidence: {state.confidence}")
    print(f"Missing information: {state.missing_information}")
    print("\nKnown facts:")
    for fact in state.known_facts:
        print(f"- {fact.fact}")
        if fact.evidence_ref:
            print(f"  source: {fact.evidence_ref}")
        if fact.solves:
            print(f"  solves: {fact.solves}")
    print("\nAgent trace:")
    for step in state.agent_trace:
        print(f"\nHop {step.hop}")
        print(f"Thought: {step.thought}")
        print(f"Action: {step.action_type} {step.action_input}")
        print(f"Observation:\n{step.observation}")
        if step.extracted_fact:
            print(f"Extracted fact: {step.extracted_fact}")
        print(f"Missing information: {step.missing_information}")

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hotpotqa_agent.core.llm import LLMClient
from hotpotqa_agent.bridge.controller import StrategyController
from hotpotqa_agent.bridge.memory import EvidenceMemory
from hotpotqa_agent.bridge.planner import BridgeSchemaPlanner
from hotpotqa_agent.bridge.schema import BridgeEntitySchema, BridgeState
from hotpotqa_agent.bridge.tools import AttributeBridgeTools
from hotpotqa_agent.data.load_hotpotqa import load_hotpotqa


def tool_for_state(schema):
    if schema.state == BridgeState.SCHEMA_CREATED:
        return "search"
    if schema.state == BridgeState.CANDIDATE_NOT_FOUND:
        if len(schema.attributes) > 1:
            return "none (multi-attribute candidate not found)"
        return "hidden_search"
    if schema.state == BridgeState.CANDIDATE_FOUND:
        return "verify"
    if schema.state == BridgeState.VERIFICATION_FAILED:
        return "relax_failed_attributes + search/hidden_search"
    if schema.state == BridgeState.VERIFIED and schema.next_relation:
        return "memory_update + build_next_schema"
    if schema.state == BridgeState.VERIFIED:
        return "finish"
    return "none"


def emit(lines, trace_lines):
    for line in lines:
        print(line)
    trace_lines.extend(lines)


def round_summary_lines(schema, round_id, tool_name=None):
    lines = ["", f"## Bridge Round {round_id}", ""]
    if tool_name:
        lines.append(f"- tool: `{tool_name}`")
    lines.extend(
        [
            f"- state: `{schema.state.value}`",
            f"- object: `{schema.object}`",
            f"- next_relation: `{schema.next_relation or '<final answer>'}`",
            "",
            "attributes:",
        ]
    )

    for item in schema.attributes:
        evidence = item.evidence_ref or item.evidence_text
        suffix = f" | evidence: {item.evidence_ref}" if evidence else ""
        lines.append(
            f"- [{item.status.value}/{item.constraint_type}] "
            f"{item.description}{suffix}"
        )

    lines.append("")
    if schema.candidate_entities:
        candidate = schema.candidate_entities[0]
        lines.extend(
            [
                "candidate bridge entity:",
                f"- {candidate.name} ({candidate.evidence_ref}, score={candidate.score:.2f})",
            ]
        )
    else:
        lines.append("candidate bridge entity: none")

    if schema.confirmed_entity:
        lines.extend(["", f"confirmed bridge entity: {schema.confirmed_entity.name}"])

    return lines


def write_trace(output_path, trace_lines):
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write("\n".join(trace_lines).rstrip() + "\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--name", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-index", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument(
        "--output",
        default="outputs/bridge_trace.md",
        help="Markdown file path for saving every printed bridge round.",
    )
    args = parser.parse_args()

    sample_split = f"{args.split}[{args.sample_index}:{args.sample_index + 1}]"
    dataset = load_hotpotqa(
        split=sample_split,
        name=args.name,
        hf_cache_dir=args.hf_cache_dir,
        offline=args.offline,
    )
    example = dataset[0]
    trace_lines = [
        "---",
        "",
        "# Bridge Agent Trace",
        "",
        f"- run_time: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- command_sample_index: `{args.sample_index}`",
        "",
        "## Loaded Sample",
        "",
        f"- id: `{example['id']}`",
        f"- type: `{example['type']}`",
        f"- level: `{example['level']}`",
        f"- question: {example['question']}",
        f"- gold answer: `{example['answer']}`",
    ]

    emit(trace_lines[:], [])

    llm = LLMClient(model=args.llm_model, max_tokens=args.llm_max_tokens)
    planner = BridgeSchemaPlanner(llm)
    tools = AttributeBridgeTools(example, llm=llm, top_k=args.top_k)
    controller = StrategyController(tools)
    memory = EvidenceMemory(question=example["question"])

    schema = planner.plan(memory)
    memory.add_schema(schema)
    emit(round_summary_lines(schema, 0), trace_lines)

    for round_id in range(1, args.max_rounds + 1):
        tool_name = tool_for_state(schema)
        schema = controller.step(schema, memory)
        if memory.bridge_schemas:
            memory.bridge_schemas[-1] = schema
        emit(round_summary_lines(schema, round_id, tool_name=tool_name), trace_lines)

        if schema.state == BridgeState.FINISHED:
            answer = (
                memory.confirmed_entities[-1].name
                if memory.confirmed_entities
                else schema.final_answer
                or (schema.confirmed_entity.name if schema.confirmed_entity else "")
            )
            final_lines = [
                "",
                "## Final",
                "",
                f"- final answer: `{answer}`",
                f"- confirmed entities: {[item.name for item in memory.confirmed_entities]}",
            ]
            emit(final_lines, trace_lines)
            write_trace(args.output, trace_lines)
            print(f"\nSaved bridge trace to {args.output}")
            return

        if schema.state == BridgeState.VERIFIED and schema.next_relation:
            schema = BridgeEntitySchema.for_next_relation(
                schema.confirmed_entity,
                schema.next_relation,
            )
            memory.add_schema(schema)
            emit(
                round_summary_lines(
                    schema,
                    f"{round_id}.next",
                    tool_name="build_next_schema",
                ),
                trace_lines,
            )

    final_lines = [
        "",
        "## Stopped Before Finish",
        "",
        f"- confirmed entities: {[item.name for item in memory.confirmed_entities]}",
    ]
    emit(final_lines, trace_lines)
    write_trace(args.output, trace_lines)
    print(f"\nSaved bridge trace to {args.output}")


if __name__ == "__main__":
    main()

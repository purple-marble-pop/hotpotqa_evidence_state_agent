import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hotpotqa_agent.core.llm import LLMClient
from hotpotqa_agent.comparison.controller import ComparisonController
from hotpotqa_agent.comparison.memory import ComparisonMemory
from hotpotqa_agent.comparison.planner import ComparisonPlanner
from hotpotqa_agent.comparison.schema import ComparisonState
from hotpotqa_agent.comparison.tools import ComparisonTools
from hotpotqa_agent.data.load_hotpotqa import load_hotpotqa


def tool_for_state(schema):
    if schema.state == ComparisonState.SCHEMA_CREATED:
        return "collect_evidence + extract_values"
    if schema.state == ComparisonState.VALUES_EXTRACTED:
        return "compare"
    if schema.state == ComparisonState.COMPARED:
        return "finish"
    return "none"


def emit(lines, trace_lines):
    for line in lines:
        print(line)
    trace_lines.extend(lines)


def value_lines(label, value):
    if not value:
        return [f"- {label}: none"]
    return [
        f"- {label}: `{value.entity}`",
        f"  - value: `{value.value}`",
        f"  - evidence: `{value.evidence_ref}`",
        f"  - confidence: `{value.confidence}`",
    ]


def schema_lines(schema, round_id, tool_name=None):
    lines = ["", f"## Comparison Round {round_id}", ""]
    if tool_name:
        lines.append(f"- tool: `{tool_name}`")
    lines.extend(
        [
            f"- state: `{schema.state.value}`",
            f"- entity_a: `{schema.entity_a}`",
            f"- entity_b: `{schema.entity_b}`",
            f"- comparison_attribute: `{schema.comparison_attribute}`",
            f"- comparison_type: `{schema.comparison_type}`",
            f"- answer_rule: `{schema.answer_rule}`",
            "",
            "values:",
        ]
    )
    lines.extend(value_lines("entity_a", schema.entity_a_value))
    lines.extend(value_lines("entity_b", schema.entity_b_value))
    if schema.final_answer:
        lines.extend(["", f"- final_answer: `{schema.final_answer}`"])
    if schema.reason:
        lines.append(f"- reason: {schema.reason}")
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
    parser.add_argument("--sample-index", type=int, default=36)
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--output", default="outputs/comparison_trace.md")
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
        "# Comparison Agent Trace",
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
    planner = ComparisonPlanner(llm)
    tools = ComparisonTools(example, llm=llm, top_k=args.top_k)
    controller = ComparisonController(tools)
    memory = ComparisonMemory(question=example["question"])

    schema = planner.plan(memory)
    memory.add_schema(schema)
    emit(schema_lines(schema, 0), trace_lines)

    for round_id in range(1, args.max_rounds + 1):
        tool_name = tool_for_state(schema)
        schema = controller.step(schema, memory)
        if memory.schemas:
            memory.schemas[-1] = schema
        emit(schema_lines(schema, round_id, tool_name), trace_lines)

        if schema.state == ComparisonState.FINISHED:
            final_lines = [
                "",
                "## Final",
                "",
                f"- final answer: `{schema.final_answer}`",
                f"- reason: {schema.reason}",
            ]
            emit(final_lines, trace_lines)
            write_trace(args.output, trace_lines)
            print(f"\nSaved comparison trace to {args.output}")
            return

    final_lines = [
        "",
        "## Stopped Before Finish",
        "",
        f"- current answer: `{schema.final_answer}`",
    ]
    emit(final_lines, trace_lines)
    write_trace(args.output, trace_lines)
    print(f"\nSaved comparison trace to {args.output}")


if __name__ == "__main__":
    main()

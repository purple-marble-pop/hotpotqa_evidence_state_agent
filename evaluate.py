import argparse
import json
import math
import re
import string
from pathlib import Path
from collections import Counter
from statistics import mean
from typing import Any, Dict, Iterable, Optional, Sequence

from hotpotqa_agent.bridge.controller import StrategyController
from hotpotqa_agent.bridge.memory import EvidenceMemory
from hotpotqa_agent.bridge.planner import BridgeSchemaPlanner
from hotpotqa_agent.bridge.schema import BridgeEntitySchema, BridgeState
from hotpotqa_agent.bridge.tools import AttributeBridgeTools
from hotpotqa_agent.comparison.controller import ComparisonController
from hotpotqa_agent.comparison.memory import ComparisonMemory
from hotpotqa_agent.comparison.planner import ComparisonPlanner
from hotpotqa_agent.comparison.schema import ComparisonState
from hotpotqa_agent.comparison.tools import ComparisonTools
from hotpotqa_agent.core.llm import LLMClient
from hotpotqa_agent.data.load_hotpotqa import get_supporting_facts, load_hotpotqa
from hotpotqa_agent.router import QuestionTypeRouter


def normalize_answer(text: str) -> str:
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1_pr(prediction: str, gold: str) -> tuple[float, float, float]:
    normalized_prediction = normalize_answer(prediction)
    normalized_gold = normalize_answer(gold)
    zero = (0.0, 0.0, 0.0)

    if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
        return zero
    if normalized_gold in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
        return zero

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_gold.split()
    if not pred_tokens and not gold_tokens:
        return (1.0, 1.0, 1.0)
    if not pred_tokens or not gold_tokens:
        return zero

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return zero
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def support_prf(
    prediction: Iterable[tuple[str, int]],
    gold: Iterable[tuple[str, int]],
) -> tuple[float, float, float, float]:
    predicted = set(prediction)
    gold_set = set(gold)
    if not predicted and not gold_set:
        return 1.0, 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0, 0.0
    tp = len(predicted & gold_set)
    precision = tp / len(predicted)
    recall = tp / len(gold_set) if gold_set else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    em = 1.0 if predicted == gold_set else 0.0
    return em, f1, precision, recall


def joint_metrics(
    answer_em_value: float,
    answer_precision: float,
    answer_recall: float,
    sp_em: float,
    sp_precision: float,
    sp_recall: float,
) -> tuple[float, float, float, float]:
    joint_em = answer_em_value * sp_em
    joint_precision = answer_precision * sp_precision
    joint_recall = answer_recall * sp_recall
    joint_f1 = (
        0.0
        if joint_precision + joint_recall == 0
        else 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
    )
    return joint_em, joint_f1, joint_precision, joint_recall


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0]
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current.append(previous[j - 1] + 1)
            else:
                current.append(max(previous[j], current[-1]))
        previous = current
    return previous[-1]


def rouge_l(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    lcs = lcs_length(pred_tokens, gold_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(gold_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def bleu(prediction: str, gold: str, max_n: int = 4) -> float:
    """Sentence BLEU with add-one smoothing for short answers."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(ngrams(pred_tokens, n))
        gold_ngrams = Counter(ngrams(gold_tokens, n))
        overlap = sum((pred_ngrams & gold_ngrams).values())
        total = sum(pred_ngrams.values())
        precisions.append((overlap + 1) / (total + 1))

    log_precision = sum(math.log(value) for value in precisions) / max_n
    brevity_penalty = (
        1.0
        if len(pred_tokens) > len(gold_tokens)
        else math.exp(1 - len(gold_tokens) / len(pred_tokens))
    )
    return brevity_penalty * math.exp(log_precision)


def meteor(prediction: str, gold: str) -> float:
    """Lightweight METEOR-style score without external dependencies."""

    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    gold_positions: dict[str, list[int]] = {}
    for index, token in enumerate(gold_tokens):
        gold_positions.setdefault(token, []).append(index)

    matches = []
    used_positions = set()
    for pred_index, token in enumerate(pred_tokens):
        for gold_index in gold_positions.get(token, []):
            if gold_index not in used_positions:
                used_positions.add(gold_index)
                matches.append((pred_index, gold_index))
                break

    match_count = len(matches)
    if match_count == 0:
        return 0.0

    precision = match_count / len(pred_tokens)
    recall = match_count / len(gold_tokens)
    harmonic = (10 * precision * recall) / (recall + 9 * precision) if precision and recall else 0.0

    matches.sort()
    chunks = 1
    for (_, prev_gold), (_, current_gold) in zip(matches, matches[1:]):
        if current_gold != prev_gold + 1:
            chunks += 1
    penalty = 0.5 * (chunks / match_count) ** 3
    return harmonic * (1 - penalty)


REF_RE = re.compile(r"(.+?)\[(\d+)\]")


def refs_from_text(text: str) -> set[tuple[str, int]]:
    refs = set()
    for title, sent_id in REF_RE.findall(str(text or "")):
        refs.add((title.strip(), int(sent_id)))
    return refs


def refs_from_bridge_schema(schema) -> set[tuple[str, int]]:
    refs = set()
    for attr in getattr(schema, "attributes", []):
        refs |= refs_from_text(attr.evidence_ref)
    for candidate in getattr(schema, "candidate_entities", []):
        refs |= refs_from_text(candidate.evidence_ref)
    if getattr(schema, "confirmed_entity", None):
        refs |= refs_from_text(schema.confirmed_entity.evidence_ref)
    return refs


def refs_from_comparison_schema(schema) -> set[tuple[str, int]]:
    refs = set()
    for value in (schema.entity_a_value, schema.entity_b_value):
        if value:
            refs |= refs_from_text(value.evidence_ref)
    return refs


def run_bridge_eval(example, llm, max_rounds: int, top_k: int) -> dict:
    planner = BridgeSchemaPlanner(llm)
    tools = AttributeBridgeTools(example, llm=llm, top_k=top_k)
    controller = StrategyController(tools)
    memory = EvidenceMemory(question=example["question"])

    schema = planner.plan(memory)
    memory.add_schema(schema)
    schemas = [schema]
    process = {
        "rounds": 0,
        "candidate_search_count": 0,
        "candidate_found_count": 0,
        "verification_attempt_count": 0,
        "verification_success_count": 0,
    }

    final_answer = ""
    for _ in range(1, max_rounds + 1):
        previous_state = schema.state
        if previous_state in {BridgeState.SCHEMA_CREATED, BridgeState.CANDIDATE_NOT_FOUND}:
            process["candidate_search_count"] += 1
        if previous_state == BridgeState.CANDIDATE_FOUND:
            process["verification_attempt_count"] += 1

        schema = controller.step(schema, memory)
        process["rounds"] += 1
        if memory.bridge_schemas:
            memory.bridge_schemas[-1] = schema
        schemas.append(schema)

        if (
            previous_state in {BridgeState.SCHEMA_CREATED, BridgeState.CANDIDATE_NOT_FOUND}
            and schema.candidate_entities
        ):
            process["candidate_found_count"] += 1
        if previous_state == BridgeState.CANDIDATE_FOUND and schema.state == BridgeState.VERIFIED:
            process["verification_success_count"] += 1

        if schema.state == BridgeState.FINISHED:
            final_answer = (
                memory.confirmed_entities[-1].name
                if memory.confirmed_entities
                else schema.final_answer
                or (schema.confirmed_entity.name if schema.confirmed_entity else "")
            )
            break

        if schema.state == BridgeState.VERIFIED and schema.next_relation:
            schema = BridgeEntitySchema.for_next_relation(schema.confirmed_entity, schema.next_relation)
            memory.add_schema(schema)
            schemas.append(schema)

    predicted_support = set()
    for item in schemas:
        predicted_support |= refs_from_bridge_schema(item)

    return {
        "prediction": final_answer,
        "predicted_support": predicted_support,
        "rounds": process["rounds"],
        "candidate_found_rate": safe_rate(
            process["candidate_found_count"], process["candidate_search_count"]
        ),
        "verification_success_rate": safe_rate(
            process["verification_success_count"], process["verification_attempt_count"]
        ),
        "confirmed_chain_length": len(memory.confirmed_entities),
        "confirmed_entities": [item.name for item in memory.confirmed_entities],
    }


def run_comparison_eval(example, llm, max_rounds: int, top_k: int) -> dict:
    planner = ComparisonPlanner(llm)
    tools = ComparisonTools(example, llm=llm, top_k=top_k)
    controller = ComparisonController(tools)
    memory = ComparisonMemory(question=example["question"])

    schema = planner.plan(memory)
    memory.add_schema(schema)
    schemas = [schema]
    rounds = 0

    for _ in range(1, max_rounds + 1):
        schema = controller.step(schema, memory)
        rounds += 1
        if memory.schemas:
            memory.schemas[-1] = schema
        schemas.append(schema)
        if schema.state == ComparisonState.FINISHED:
            break

    predicted_support = set()
    for item in schemas:
        predicted_support |= refs_from_comparison_schema(item)

    value_successes = sum(
        1
        for value in (schema.entity_a_value, schema.entity_b_value)
        if value and value.value and value.value.lower() != "unknown"
    )

    return {
        "prediction": schema.final_answer,
        "predicted_support": predicted_support,
        "rounds": rounds,
        "candidate_found_rate": None,
        "verification_success_rate": None,
        "confirmed_chain_length": value_successes,
        "confirmed_entities": [
            value.entity
            for value in (schema.entity_a_value, schema.entity_b_value)
            if value and value.value and value.value.lower() != "unknown"
        ],
    }


def safe_rate(numerator: int, denominator: int) -> Optional[float]:
    return None if denominator == 0 else numerator / denominator


def is_api_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "connection error",
            "connectionerror",
            "timeout",
            "rate limit",
            "api",
            "ssl",
            "temporarily unavailable",
        )
    )


def score_example(example, prediction: str, predicted_support: Iterable[tuple[str, int]]) -> dict:
    gold_answer = example["answer"]
    em = answer_em(prediction, gold_answer)
    f1, precision, recall = answer_f1_pr(prediction, gold_answer)
    sp_em, sp_f1, sp_precision, sp_recall = support_prf(
        predicted_support,
        get_supporting_facts(example),
    )
    joint_em, joint_f1, joint_precision, joint_recall = joint_metrics(
        em,
        precision,
        recall,
        sp_em,
        sp_precision,
        sp_recall,
    )
    return {
        "answer_em": em,
        "answer_f1": f1,
        "sp_f1": sp_f1,
        "joint_f1": joint_f1,
        "bleu": bleu(prediction, gold_answer),
        "rouge_l": rouge_l(prediction, gold_answer),
        "meteor": meteor(prediction, gold_answer),
    }


def summarize(results: list[dict]) -> dict:
    valid_results = [item for item in results if not item.get("api_error")]
    summary = {
        "count": len(results),
        "evaluated_count": len(valid_results),
        "api_error_count": len(results) - len(valid_results),
    }
    metric_map = {
        "avg_answer_em": "answer_em",
        "avg_answer_f1": "answer_f1",
        "avg_sp_f1": "sp_f1",
        "avg_joint_f1": "joint_f1",
        "avg_bleu": "bleu",
        "avg_rouge_l": "rouge_l",
        "avg_meteor": "meteor",
        "avg_reasoning_rounds": "rounds",
        "avg_candidate_found_rate": "candidate_found_rate",
        "avg_verification_success_rate": "verification_success_rate",
        "avg_confirmed_chain_length": "confirmed_chain_length",
    }
    for summary_name, item_name in metric_map.items():
        values = [item[item_name] for item in valid_results if item.get(item_name) is not None]
        summary[summary_name] = mean(values) if values else None
    return summary


def printable_support(refs: Iterable[tuple[str, int]]) -> list[str]:
    return sorted(f"{title}[{sent_id}]" for title, sent_id in refs)


def evaluate_example(example, index: int, llm, router, max_rounds: int, top_k: int) -> dict:
    route = router.route(example)
    if route.question_type == "comparison":
        run = run_comparison_eval(example, llm, max_rounds=max_rounds, top_k=top_k)
    else:
        run = run_bridge_eval(example, llm, max_rounds=max_rounds, top_k=top_k)

    scores = score_example(example, run["prediction"], run["predicted_support"])
    return {
        "index": index,
        "id": example["id"],
        "gold_type": example.get("type", ""),
        "route_type": route.question_type,
        "route_source": route.source,
        "question": example["question"],
        "prediction": run["prediction"],
        "gold_answer": example["answer"],
        **scores,
        "rounds": run["rounds"],
        "candidate_found_rate": run["candidate_found_rate"],
        "verification_success_rate": run["verification_success_rate"],
        "confirmed_chain_length": run["confirmed_chain_length"],
        "confirmed_entities": run["confirmed_entities"],
        "predicted_support": printable_support(run["predicted_support"]),
        "gold_support": printable_support(get_supporting_facts(example)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--name", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--use-llm-router", action="store_true")
    parser.add_argument("--output", default="outputs/eval_hybrid.jsonl")
    args = parser.parse_args()

    sample_split = f"{args.split}[{args.sample_index}:{args.sample_index + args.sample_count}]"
    dataset = load_hotpotqa(
        split=sample_split,
        name=args.name,
        hf_cache_dir=args.hf_cache_dir,
        offline=args.offline,
    )
    llm = LLMClient(model=args.llm_model, max_tokens=args.llm_max_tokens)
    router = QuestionTypeRouter(llm=llm, prefer_dataset_type=not args.use_llm_router)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with output.open("w", encoding="utf-8") as f:
        for offset, example in enumerate(dataset):
            index = args.sample_index + offset
            try:
                result = evaluate_example(
                    example,
                    index=index,
                    llm=llm,
                    router=router,
                    max_rounds=args.max_rounds,
                    top_k=args.top_k,
                )
            except Exception as exc:
                api_error = is_api_error(exc)
                result = {
                    "index": index,
                    "id": example.get("id", ""),
                    "gold_type": example.get("type", ""),
                    "question": example.get("question", ""),
                    "gold_answer": example.get("answer", ""),
                    "prediction": "",
                    "error": str(exc),
                    "api_error": api_error,
                    "answer_em": 0.0,
                    "answer_f1": 0.0,
                    "sp_f1": 0.0,
                    "joint_f1": 0.0,
                    "bleu": 0.0,
                    "rouge_l": 0.0,
                    "meteor": 0.0,
                    "rounds": 0,
                    "candidate_found_rate": None,
                    "verification_success_rate": None,
                    "confirmed_chain_length": 0,
                }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{index}] type={result.get('route_type', result.get('gold_type'))} "
                f"EM={result['answer_em']:.0f} F1={result['answer_f1']:.2f} "
                f"BLEU={result['bleu']:.2f} R-L={result['rouge_l']:.2f} "
                f"METEOR={result['meteor']:.2f} "
                f"Pred={result.get('prediction', '')!r} Gold={result.get('gold_answer', '')!r}"
            )

    summary = summarize(results)
    print("\nSummary")
    print("=" * 80)
    for key, value in summary.items():
        if value is None:
            print(f"{key}: n/a")
        elif key in {"count", "evaluated_count", "api_error_count"}:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"\nSaved per-example results to {output}")


if __name__ == "__main__":
    main()

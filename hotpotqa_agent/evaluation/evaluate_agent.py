import argparse
import json
import math
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable, Sequence

from hotpotqa_agent.agent.hotpot_agent import HotpotEvidenceStateAgent
from hotpotqa_agent.agent.llm import LLMClient
from hotpotqa_agent.data.load_hotpotqa import get_supporting_facts, load_hotpotqa


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles, and extra whitespace."""
    text = str(text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_em(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def answer_f1_pr(prediction: str, gold: str) -> tuple[float, float, float]:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def answer_f1(prediction: str, gold: str) -> float:
    return answer_f1_pr(prediction, gold)[0]


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


def evidence_refs(items: Iterable) -> set[tuple[str, int]]:
    refs = set()
    for item in items:
        if item.sent_id is not None:
            refs.add((item.title, int(item.sent_id)))
    return refs


def prf(predicted: set, gold: set) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0

    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted)
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def joint_f1(answer_precision: float, answer_recall: float, support_precision: float, support_recall: float) -> float:
    joint_precision = answer_precision * support_precision
    joint_recall = answer_recall * support_recall
    return (
        0.0
        if joint_precision + joint_recall == 0
        else 2 * joint_precision * joint_recall / (joint_precision + joint_recall)
    )


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


def evaluate_example(agent: HotpotEvidenceStateAgent, example, index: int) -> dict:
    state = agent.run(example)
    prediction = state.final_answer or ""
    gold_answer = example["answer"]
    predicted_support = evidence_refs(state.evidence_chain)
    gold_support = set(get_supporting_facts(example))
    support_precision, support_recall, support_f1 = prf(predicted_support, gold_support)
    answer_f1_value, answer_precision, answer_recall = answer_f1_pr(prediction, gold_answer)

    return {
        "index": index,
        "id": example["id"],
        "type": example.get("type", ""),
        "level": example.get("level", ""),
        "question": example["question"],
        "prediction": prediction,
        "gold_answer": gold_answer,
        "answer_em": answer_em(prediction, gold_answer),
        "answer_f1": answer_f1_value,
        "answer_rate": float(bool(prediction)),
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "joint_f1": joint_f1(answer_precision, answer_recall, support_precision, support_recall),
        "bleu": bleu(prediction, gold_answer),
        "rouge_l": rouge_l(prediction, gold_answer),
        "meteor": meteor(prediction, gold_answer),
        "predicted_support": sorted([f"{title}[{sent_id}]" for title, sent_id in predicted_support]),
        "gold_support": sorted([f"{title}[{sent_id}]" for title, sent_id in gold_support]),
        "hops": len(state.agent_trace),
        "confidence": state.confidence,
        "missing_information": state.missing_information,
    }


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    valid_results = [item for item in results if not item.get("api_error")]
    metric_names = [
        "answer_em",
        "answer_f1",
        "answer_rate",
        "support_precision",
        "support_recall",
        "support_f1",
        "joint_f1",
        "bleu",
        "rouge_l",
        "meteor",
        "hops",
    ]
    summary = {
        "count": len(results),
        "evaluated_count": len(valid_results),
        "api_error_count": len(results) - len(valid_results),
    }
    for name in metric_names:
        values = [item[name] for item in valid_results if item.get(name) is not None]
        summary[name] = mean(values) if values else None
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--name", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--output", default="outputs/eval_agent.jsonl")
    args = parser.parse_args()

    sample_split = f"{args.split}[{args.sample_index}:{args.sample_index + args.sample_count}]"
    dataset = load_hotpotqa(
        split=sample_split,
        name=args.name,
        hf_cache_dir=args.hf_cache_dir,
        offline=args.offline,
    )
    llm = LLMClient(model=args.llm_model, max_tokens=args.llm_max_tokens)
    agent = HotpotEvidenceStateAgent(llm=llm, max_hops=args.max_hops, top_k=args.top_k)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with output.open("w", encoding="utf-8") as f:
        for offset, example in enumerate(dataset):
            index = args.sample_index + offset
            try:
                result = evaluate_example(agent, example, index)
            except Exception as exc:
                api_error = is_api_error(exc)
                result = {
                    "index": index,
                    "id": example.get("id", ""),
                    "question": example.get("question", ""),
                    "gold_answer": example.get("answer", ""),
                    "error": str(exc),
                    "api_error": api_error,
                    "answer_em": 0.0,
                    "answer_f1": 0.0,
                    "answer_rate": 0.0,
                    "support_precision": 0.0,
                    "support_recall": 0.0,
                    "support_f1": 0.0,
                    "joint_f1": 0.0,
                    "bleu": 0.0,
                    "rouge_l": 0.0,
                    "meteor": 0.0,
                    "hops": 0,
                }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{index}] EM={result['answer_em']:.0f} "
                f"F1={result['answer_f1']:.2f} "
                f"SupportF1={result['support_f1']:.2f} "
                f"JointF1={result['joint_f1']:.2f} "
                f"BLEU={result['bleu']:.2f} "
                f"R-L={result['rouge_l']:.2f} "
                f"METEOR={result['meteor']:.2f} "
                f"Pred={result.get('prediction', '')!r} "
                f"Gold={result.get('gold_answer', '')!r}"
            )

    summary = summarize(results)
    print("\nSummary")
    print("=" * 80)
    for key, value in summary.items():
        if key in {"count", "evaluated_count", "api_error_count"}:
            print(f"{key}: {value}")
        elif value is None:
            print(f"{key}: n/a")
        else:
            print(f"{key}: {value:.4f}")
    print(f"\nSaved per-example results to {output}")


if __name__ == "__main__":
    main()

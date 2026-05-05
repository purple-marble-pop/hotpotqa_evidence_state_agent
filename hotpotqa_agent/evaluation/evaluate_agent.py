import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Iterable

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


def answer_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


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


def evaluate_example(agent: HotpotEvidenceStateAgent, example, index: int) -> dict:
    state = agent.run(example)
    prediction = state.final_answer or ""
    gold_answer = example["answer"]
    predicted_support = evidence_refs(state.evidence_chain)
    gold_support = set(get_supporting_facts(example))
    support_precision, support_recall, support_f1 = prf(predicted_support, gold_support)

    return {
        "index": index,
        "id": example["id"],
        "type": example.get("type", ""),
        "level": example.get("level", ""),
        "question": example["question"],
        "prediction": prediction,
        "gold_answer": gold_answer,
        "answer_em": answer_em(prediction, gold_answer),
        "answer_f1": answer_f1(prediction, gold_answer),
        "answer_rate": float(bool(prediction)),
        "support_precision": support_precision,
        "support_recall": support_recall,
        "support_f1": support_f1,
        "predicted_support": sorted([f"{title}[{sent_id}]" for title, sent_id in predicted_support]),
        "gold_support": sorted([f"{title}[{sent_id}]" for title, sent_id in gold_support]),
        "hops": len(state.agent_trace),
        "confidence": state.confidence,
        "missing_information": state.missing_information,
    }


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    metric_names = [
        "answer_em",
        "answer_f1",
        "answer_rate",
        "support_precision",
        "support_recall",
        "support_f1",
        "hops",
    ]
    summary = {"count": len(results)}
    for name in metric_names:
        summary[name] = mean(item[name] for item in results)
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
                result = {
                    "index": index,
                    "id": example.get("id", ""),
                    "question": example.get("question", ""),
                    "gold_answer": example.get("answer", ""),
                    "error": str(exc),
                    "answer_em": 0.0,
                    "answer_f1": 0.0,
                    "answer_rate": 0.0,
                    "support_precision": 0.0,
                    "support_recall": 0.0,
                    "support_f1": 0.0,
                    "hops": 0,
                }
            results.append(result)
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            print(
                f"[{index}] EM={result['answer_em']:.0f} "
                f"F1={result['answer_f1']:.2f} "
                f"SupportF1={result['support_f1']:.2f} "
                f"Pred={result.get('prediction', '')!r} "
                f"Gold={result.get('gold_answer', '')!r}"
            )

    summary = summarize(results)
    print("\nSummary")
    print("=" * 80)
    for key, value in summary.items():
        if key == "count":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"\nSaved per-example results to {output}")


if __name__ == "__main__":
    main()

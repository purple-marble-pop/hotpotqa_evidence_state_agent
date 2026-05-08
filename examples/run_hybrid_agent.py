import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hotpotqa_agent.core.llm import LLMClient
from hotpotqa_agent.data.load_hotpotqa import load_hotpotqa
from hotpotqa_agent.router import QuestionTypeRouter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--name", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--llm-model", default=None)
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument(
        "--use-llm-router",
        action="store_true",
        help="Ignore HotpotQA type field and classify the question with the LLM router.",
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
    llm = LLMClient(model=args.llm_model, max_tokens=args.llm_max_tokens)
    router = QuestionTypeRouter(llm=llm, prefer_dataset_type=not args.use_llm_router)
    decision = router.route(example)

    print("Route decision")
    print("=" * 100)
    print(f"question type: {decision.question_type}")
    print(f"source: {decision.source}")
    print(f"reason: {decision.reason}")

    script = (
        "examples/run_comparison_agent.py"
        if decision.question_type == "comparison"
        else "examples/run_bridge_agent.py"
    )
    command = [
        sys.executable,
        script,
        "--split",
        args.split,
        "--name",
        args.name,
        "--sample-index",
        str(args.sample_index),
        "--max-rounds",
        str(args.max_rounds),
        "--top-k",
        str(args.top_k),
    ]
    if args.hf_cache_dir:
        command.extend(["--hf-cache-dir", args.hf_cache_dir])
    if args.offline:
        command.append("--offline")
    if args.llm_model:
        command.extend(["--llm-model", args.llm_model])
    if args.llm_max_tokens:
        command.extend(["--llm-max-tokens", str(args.llm_max_tokens)])

    print("\nDispatch")
    print("=" * 100)
    print(" ".join(command))
    raise SystemExit(subprocess.call(command, cwd=PROJECT_ROOT))


if __name__ == "__main__":
    main()

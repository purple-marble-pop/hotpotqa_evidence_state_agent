import argparse, json, sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from hotpotqa_agent.data.load_hotpotqa import load_hotpotqa
from hotpotqa_agent.agent.llm import LLMClient
from hotpotqa_agent.agent.hotpot_agent import HotpotEvidenceStateAgent, print_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--name", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-index", type=int, default=3)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--llm-model", default=None, help="Override LLM_MODEL from .env for this run.")
    parser.add_argument("--llm-max-tokens", type=int, default=None, help="Override LLM_MAX_TOKENS from .env for this run.")
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()
    sample_split = f"{args.split}[{args.sample_index}:{args.sample_index + 1}]"
    dataset = load_hotpotqa(split=sample_split, name=args.name, hf_cache_dir=args.hf_cache_dir, offline=args.offline)
    example = dataset[0]

    print("Loaded sample")
    print("=" * 100)
    print(f"split: {args.split}")
    print(f"sample index: {args.sample_index}")
    print(f"id: {example['id']}")
    print(f"type: {example['type']}")
    print(f"level: {example['level']}")
    print(f"question: {example['question']}")
    print(f"gold answer: {example['answer']}")
    llm = LLMClient(model=args.llm_model, max_tokens=args.llm_max_tokens)
    print(f"llm model: {llm.model}")
    print(f"llm max tokens: {llm.max_tokens}")
    agent = HotpotEvidenceStateAgent(
        llm=llm,
        max_hops=args.max_hops,
        top_k=args.top_k,
    )
    state = agent.run(example)
    print_state(state)
    if args.save_json:
        output = Path(args.save_json); output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved JSON to {output}")
if __name__ == "__main__":
    main()

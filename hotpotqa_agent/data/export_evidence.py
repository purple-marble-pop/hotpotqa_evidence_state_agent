import argparse
from itertools import islice
from pathlib import Path
from .load_hotpotqa import load_hotpotqa, get_supporting_facts, get_sentence


def format_example(example, index):
    lines = [f"## {index}. {example['question']}", "", f"- id: `{example['id']}`", f"- type: `{example['type']}`", f"- level: `{example['level']}`", f"- answer: `{example['answer']}`", "", "Evidence:", ""]
    for title, sent_id in get_supporting_facts(example):
        sentence = get_sentence(example, title, sent_id)
        lines.append(f"- `{title}[{sent_id}]`: {sentence}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--name", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--output", default="outputs/hotpot_evidence.md")
    args = parser.parse_args()
    dataset = load_hotpotqa(split=args.split, name=args.name, hf_cache_dir=args.hf_cache_dir, offline=args.offline)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write("# HotpotQA Evidence View\n\n")
        f.write("Each item shows question, answer, metadata, and gold evidence sentences.\n\n")
        for offset, example in enumerate(islice(dataset, args.sample_index, args.sample_index + args.sample_count)):
            f.write(format_example(example, args.sample_index + offset))
    print(f"Saved to {output}")

if __name__ == "__main__":
    main()

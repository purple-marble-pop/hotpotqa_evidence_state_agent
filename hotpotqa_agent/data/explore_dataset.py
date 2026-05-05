import argparse
from itertools import islice
from .load_hotpotqa import load_hotpotqa, get_context_pages, get_supporting_facts


def print_sample(example, index):
    support_set = set(get_supporting_facts(example))
    print("\n" + "=" * 100)
    print(f"Sample #{index}")
    print("=" * 100)
    print(f"id: {example['id']}")
    print(f"type: {example['type']}")
    print(f"level: {example['level']}")
    print(f"question: {example['question']}")
    print(f"answer: {example['answer']}")
    print("\nSupporting facts:")
    for title, sent_id in get_supporting_facts(example):
        print(f"  - {title}[{sent_id}]")
    print("\nContext:")
    for page_index, (title, sentences) in enumerate(get_context_pages(example)):
        print(f"\n  Candidate page {page_index}: {title}")
        for sent_id, sentence in enumerate(sentences):
            marker = " <-- supporting fact" if (title, sent_id) in support_set else ""
            print(f"    [{sent_id}] {sentence}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation"])
    parser.add_argument("--name", default="distractor", choices=["distractor", "fullwiki"])
    parser.add_argument("--hf-cache-dir", default=None)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=1)
    args = parser.parse_args()
    dataset = load_hotpotqa(split=args.split, name=args.name, hf_cache_dir=args.hf_cache_dir, offline=args.offline)
    samples = islice(dataset, args.sample_index, args.sample_index + args.sample_count)
    for offset, example in enumerate(samples):
        print_sample(example, args.sample_index + offset)

if __name__ == "__main__":
    main()

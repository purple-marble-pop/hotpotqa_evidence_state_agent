import os
import re
from pathlib import Path

from hotpotqa_agent.core.llm import ensure_valid_ssl_cert_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_CACHE_DIR = PROJECT_ROOT.parent / "hf_cache"


def load_hotpotqa(split="validation", name="distractor", hf_cache_dir=None, offline=False):
    """Load HotpotQA from Hugging Face."""
    ensure_valid_ssl_cert_env()
    if hf_cache_dir is None:
        env_cache_dir = os.getenv("HOTPOTQA_HF_CACHE_DIR")
        if env_cache_dir:
            hf_cache_dir = env_cache_dir
            offline = True
        elif DEFAULT_HF_CACHE_DIR.exists():
            hf_cache_dir = str(DEFAULT_HF_CACHE_DIR)
            offline = True
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    from datasets import load_dataset
    try:
        return load_dataset("hotpotqa/hotpot_qa", name=name, split=split, cache_dir=hf_cache_dir)
    except Exception:
        if not offline or not hf_cache_dir:
            raise
        return load_hotpotqa_from_arrow_cache(split=split, name=name, hf_cache_dir=hf_cache_dir)


def load_hotpotqa_from_arrow_cache(split="validation", name="distractor", hf_cache_dir=None):
    """Load a dragged/copied Hugging Face HotpotQA Arrow cache without Hub access."""
    from datasets import Dataset, concatenate_datasets

    split_name, start, end = parse_split_slice(split)
    cache_root = Path(hf_cache_dir).expanduser()
    pattern = f"**/{name}/**/hotpot_qa-{split_name}*.arrow"
    files = sorted(cache_root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No HotpotQA Arrow files found under {cache_root} with pattern {pattern}. "
            "Check whether --hf-cache-dir points to the directory above hotpotqa___hotpot_qa."
        )

    datasets = [Dataset.from_file(str(path)) for path in files]
    dataset = datasets[0] if len(datasets) == 1 else concatenate_datasets(datasets)
    if start is not None or end is not None:
        dataset = dataset.select(range(start or 0, min(end if end is not None else len(dataset), len(dataset))))
    return dataset


def parse_split_slice(split):
    match = re.fullmatch(r"([A-Za-z_]+)(?:\[(\d*)?:(\d*)?\])?", split)
    if not match:
        return split, None, None
    split_name, start, end = match.groups()
    return split_name, int(start) if start else None, int(end) if end else None


def get_context_pages(example):
    return list(zip(example["context"]["title"], example["context"]["sentences"]))


def get_supporting_facts(example):
    return list(zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"]))


def get_sentence(example, title, sent_id):
    for current_title, sentences in get_context_pages(example):
        if current_title == title:
            return sentences[sent_id] if 0 <= sent_id < len(sentences) else None
    return None

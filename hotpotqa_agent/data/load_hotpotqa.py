import os
from datasets import load_dataset

from hotpotqa_agent.agent.llm import ensure_valid_ssl_cert_env


def load_hotpotqa(split="validation", name="distractor", hf_cache_dir=None, offline=False):
    """Load HotpotQA from Hugging Face."""
    ensure_valid_ssl_cert_env()
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    return load_dataset("hotpotqa/hotpot_qa", name=name, split=split, cache_dir=hf_cache_dir)


def get_context_pages(example):
    return list(zip(example["context"]["title"], example["context"]["sentences"]))


def get_supporting_facts(example):
    return list(zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"]))


def get_sentence(example, title, sent_id):
    for current_title, sentences in get_context_pages(example):
        if current_title == title:
            return sentences[sent_id] if 0 <= sent_id < len(sentences) else None
    return None

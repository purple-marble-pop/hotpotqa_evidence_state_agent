import re
from typing import List, Tuple
from .state import Evidence

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "of", "on", "or", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "whom", "whose",
}

def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

def tokens(text: str) -> List[str]:
    return [token for token in normalize(text).split() if token not in STOPWORDS]

def token_set(text: str):
    return set(tokens(text))

def ngrams(items: List[str], n: int):
    return {" ".join(items[i:i+n]) for i in range(max(0, len(items) - n + 1))}

def phrase_score(query: str, text: str) -> float:
    q_tokens = tokens(query)
    normalized_text = normalize(text)
    score = 0.0

    for n, weight in ((4, 8.0), (3, 5.0), (2, 2.5)):
        for phrase in ngrams(q_tokens, n):
            if phrase in normalized_text:
                score += weight

    normalized_query = " ".join(q_tokens)
    if normalized_query and normalized_query in normalized_text:
        score += 10.0

    return score

class ContextSearchTool:
    """Search over one HotpotQA sample's candidate context pages."""
    def __init__(self, example):
        self.example = example
        self.pages = list(zip(example["context"]["title"], example["context"]["sentences"]))

    def list_titles(self) -> List[str]:
        return [title for title, _ in self.pages]

    def search(self, query: str, top_k: int = 5) -> List[Evidence]:
        q_tokens = token_set(query)
        scored: List[Evidence] = []
        for title, sentences in self.pages:
            title_tokens = token_set(title)
            for sent_id, sentence in enumerate(sentences):
                text_tokens = title_tokens | token_set(sentence)
                score = float(len(q_tokens & text_tokens))
                score += phrase_score(query, title) * 1.5
                score += phrase_score(query, sentence)
                if score > 0:
                    scored.append(Evidence(title=title, sent_id=sent_id, sentence=sentence, score=score))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]

    def lookup_title(self, title_query: str, top_k: int = 3) -> List[Evidence]:
        q_tokens = token_set(title_query)
        candidates: List[Tuple[float, str, list]] = []
        for title, sentences in self.pages:
            score = float(len(q_tokens & token_set(title)))
            score += phrase_score(title_query, title) * 1.5
            if score > 0:
                candidates.append((score, title, sentences))
        candidates.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, title, sentences in candidates[:top_k]:
            text = " ".join(f"[{i}] {s}" for i, s in enumerate(sentences))
            results.append(Evidence(title=title, sent_id=None, sentence=text, score=score))
        return results


def format_evidence_list(evidence: List[Evidence]) -> str:
    return "No evidence found." if not evidence else "\n".join(f"- {item.ref}: {item.sentence}" for item in evidence)

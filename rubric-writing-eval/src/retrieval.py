from __future__ import annotations

import math
import re
from collections import Counter


def _terms(text: str) -> list[str]:
    english = re.findall(r"[a-z]+", text.lower())
    chinese = re.findall(r"[\u4e00-\u9fff]", text)
    return english + chinese


class RubricRetriever:
    def __init__(self, passages: dict[str, str]):
        self.keys = list(passages)
        self.passages = passages
        self.docs = [_terms(passages[k]) for k in self.keys]
        self.document_frequency = Counter()
        for doc in self.docs:
            self.document_frequency.update(set(doc))

    def _vector(self, tokens: list[str]) -> dict[str, float]:
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        vector = {}
        for term, count in counts.items():
            idf = math.log((1 + len(self.docs)) / (1 + self.document_frequency.get(term, 0))) + 1
            vector[term] = (count / total) * idf
        return vector

    @staticmethod
    def _cosine(left: dict[str, float], right: dict[str, float]) -> float:
        shared = set(left) & set(right)
        numerator = sum(left[t] * right[t] for t in shared)
        lnorm = math.sqrt(sum(v * v for v in left.values()))
        rnorm = math.sqrt(sum(v * v for v in right.values()))
        return numerator / (lnorm * rnorm) if lnorm and rnorm else 0.0

    def retrieve(self, query: str, top_k: int = 2) -> list[tuple[str, float, str]]:
        qvec = self._vector(_terms(query))
        scored = []
        for key, doc in zip(self.keys, self.docs):
            scored.append((key, self._cosine(qvec, self._vector(doc)), self.passages[key]))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:top_k]

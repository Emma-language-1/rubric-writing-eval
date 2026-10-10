from __future__ import annotations

import re
from collections import Counter


EN_CONNECTIVES = {
    "however", "therefore", "because", "although", "first", "second",
    "for example", "in conclusion", "rather than", "while",
}
ZH_CONNECTIVES = {"因此", "但是", "然而", "因为", "例如", "首先", "其次", "所以", "既", "也", "而"}


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?。！？]+", text) if s.strip()]


def _tokens(text: str, language: str) -> list[str]:
    if language == "zh":
        return re.findall(r"[\u4e00-\u9fff]", text)
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower())


def extract_features(text: str, language: str) -> dict[str, float]:
    tokens = _tokens(text, language)
    sentences = _sentences(text)
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    token_count = max(len(tokens), 1)
    unique_count = len(set(tokens))
    connectives = ZH_CONNECTIVES if language == "zh" else EN_CONNECTIVES
    lower = text.lower()
    connective_hits = sum(lower.count(item) for item in connectives)
    repeated = sum(count - 1 for count in Counter(tokens).values() if count > 1)
    return {
        "token_count": float(len(tokens)),
        "sentence_count": float(len(sentences)),
        "paragraph_count": float(len(paragraphs)),
        "type_token_ratio": unique_count / token_count,
        "mean_sentence_length": len(tokens) / max(len(sentences), 1),
        "connective_density": connective_hits / max(len(sentences), 1),
        "repetition_ratio": repeated / token_count,
        "example_signal": float(any(x in lower for x in ("for example", "例如"))),
        "position_signal": float(any(x in lower for x in ("should", "must", "应该", "不宜", "需要"))),
    }

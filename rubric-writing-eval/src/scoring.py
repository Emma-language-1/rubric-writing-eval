from __future__ import annotations

from .features import extract_features


TRAITS = ("task_response", "organisation", "language_use", "evidence")


def _clip(value: float) -> int:
    return int(max(1, min(9, round(value))))


def score_essay(text: str, language: str) -> dict[str, int]:
    """Transparent baseline used to exercise the evaluation pipeline.

    The feature weights are illustrative and are not calibrated for deployment.
    """
    f = extract_features(text, language)
    length_target = 120 if language == "en" else 180
    length_component = min(f["token_count"] / length_target, 1.0)
    task = 2.5 + 3.0 * length_component + 1.2 * f["position_signal"] + 0.8 * f["example_signal"]
    organisation = 3.0 + min(f["sentence_count"], 6) * 0.45 + min(f["connective_density"], 2) * 0.8
    language_use = 3.0 + 3.5 * f["type_token_ratio"] + min(f["mean_sentence_length"] / 20, 1.0)
    evidence = 2.5 + 2.0 * f["example_signal"] + 1.2 * min(f["connective_density"], 1.5) + 1.5 * length_component
    return {
        "task_response": _clip(task),
        "organisation": _clip(organisation),
        "language_use": _clip(language_use),
        "evidence": _clip(evidence),
    }

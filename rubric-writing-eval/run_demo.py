from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data_prep import build_preference_records, build_sft_records, load_jsonl, write_jsonl
from src.diagnostics import dbscan, pca_2d
from src.metrics import agreement_summary
from src.retrieval import RubricRetriever
from src.scoring import TRAITS, score_essay


ROOT = Path(__file__).parent


def main() -> None:
    essays = load_jsonl(ROOT / "data" / "sample_essays.jsonl")
    preferences = load_jsonl(ROOT / "data" / "feedback_preferences.jsonl")
    rubric = json.loads((ROOT / "data" / "rubric.json").read_text(encoding="utf-8"))
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    predictions = {item["id"]: score_essay(item["text"], item["language"]) for item in essays}
    metrics = {}
    error_rows = []
    for trait in TRAITS:
        human = [item["human_scores"][trait] for item in essays]
        model = [predictions[item["id"]][trait] for item in essays]
        metrics[trait] = agreement_summary(human, model)
    for item in essays:
        error_rows.append([
            predictions[item["id"]][trait] - item["human_scores"][trait]
            for trait in TRAITS
        ])

    points = pca_2d(np.asarray(error_rows, dtype=float))
    clusters = dbscan(points, eps=1.75, min_samples=2)
    diagnostics = [{
        "essay_id": essay["id"],
        "pca_x": round(float(point[0]), 4),
        "pca_y": round(float(point[1]), 4),
        "cluster": int(cluster),
    } for essay, point, cluster in zip(essays, points, clusters)]

    passages = {
        f"{trait}_{level}": text
        for trait, content in rubric["traits"].items()
        for level, text in content.items()
        if level in {"low", "mid", "high"}
    }
    retrieved = RubricRetriever(passages).retrieve("clear position with relevant examples", top_k=3)

    summary = {
        "data_statement": "Synthetic demonstration data only",
        "essay_count": len(essays),
        "metrics": metrics,
        "retrieval_example": [
            {"rubric_passage": key, "similarity": round(score, 4), "text": text}
            for key, score, text in retrieved
        ],
        "diagnostic_clusters": diagnostics,
    }
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_jsonl(build_sft_records(essays, rubric), output_dir / "sft_records.jsonl")
    write_jsonl(build_preference_records(preferences), output_dir / "preference_records.jsonl")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

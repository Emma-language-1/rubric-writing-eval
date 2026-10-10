from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(records: list[dict], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_sft_records(essays: list[dict], rubric: dict) -> list[dict]:
    records = []
    for essay in essays:
        instruction = (
            "Score the essay using the supplied multi-trait rubric. Return concise, "
            "auditable JSON scores and do not invent evidence."
        )
        records.append({
            "instruction": instruction,
            "input": {"prompt": essay["prompt"], "essay": essay["text"], "rubric": rubric},
            "output": {"scores": essay["human_scores"], "label_source": "synthetic_human_rating"},
        })
    return records


def build_preference_records(preferences: list[dict]) -> list[dict]:
    return [{
        "prompt": f"Provide rubric-grounded feedback for trait: {item['rubric_trait']}",
        "chosen": item["chosen"],
        "rejected": item["rejected"],
        "metadata": {"essay_id": item["essay_id"], "source": item["source"]},
    } for item in preferences]

from __future__ import annotations

import json
import os
import urllib.request


def score_with_local_endpoint(system_prompt: str, user_content: str) -> dict:
    base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("LOCAL_LLM_MODEL", "local-model")
    payload = {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
    return json.loads(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    print("Import score_with_local_endpoint after starting an OpenAI-compatible local server.")

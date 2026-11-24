"""Optional LLM summaries via the Anthropic Messages API (stdlib only).

Set ANTHROPIC_API_KEY to enable. Without a key (or on any error) the app falls
back to the extractive summary, so the search never depends on the network.
The prompt restricts the model to the retrieved passages to limit invented facts.
"""
from __future__ import annotations

import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("HERITAGE_LLM_MODEL", "claude-haiku-5-5")


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def summarize(query: str, results: list[dict], timeout: int = 30) -> str | None:
    if not llm_available() or not results:
        return None
    passages = "\n\n".join(
        f"[{r['rank']}] {r['title']} ({r['region']}, {r['period']}): {r['text']}"
        for r in results
    )
    prompt = (
        f"A visitor searched a digital heritage collection for: \"{query}\".\n\n"
        f"Retrieved passages:\n{passages}\n\n"
        "Write a 3-4 sentence summary of what these passages say about the search. "
        "Use ONLY facts in the passages, cite passage numbers like [1], and say so "
        "if the passages do not really answer the search."
    )
    body = json.dumps({
        "model": DEFAULT_MODEL,
        "max_tokens": 300,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(API_URL, data=body, headers={
        "content-type": "application/json",
        "x-api-key": os.environ["ANTHROPIC_API_KEY"],
        "anthropic-version": "2023-06-01",
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        return "".join(b.get("text", "") for b in data.get("content", [])).strip() or None
    except Exception:
        return None

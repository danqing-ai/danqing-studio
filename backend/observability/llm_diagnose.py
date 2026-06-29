"""LLM-assisted task diagnosis (product UI + agents)."""

from __future__ import annotations

import json
from typing import Any

from backend.core.contracts import ChatCompletionRequest, ChatMessage
from backend.engine.llm.prompts.system import TASK_DIAGNOSE_SYSTEM


def _compact_bundle(bundle: dict[str, Any]) -> dict[str, Any]:
    failure = bundle.get("failure") or {}
    graph = bundle.get("graph") or {}
    logs = bundle.get("logs") or []
    tail = logs[-12:] if isinstance(logs, list) else []
    return {
        "task_id": bundle.get("task_id"),
        "kind": bundle.get("kind"),
        "status": bundle.get("status"),
        "model": bundle.get("model"),
        "failure": failure,
        "graph": {
            "graph_id": graph.get("graph_id"),
            "active_node": graph.get("active_node"),
            "progress": graph.get("progress"),
            "nodes": graph.get("nodes"),
        },
        "recent_logs": tail,
    }


def llm_diagnose_task(bundle: dict[str, Any], llm_service: Any, *, locale: str = "zh") -> str:
    """Run local LLM over a diagnostic bundle; raises if LLM unavailable."""
    if llm_service is None:
        raise RuntimeError("LLM service is not configured")
    compact = _compact_bundle(bundle)
    user_loc = (
        "## Output language\nRespond in Simplified Chinese (简体中文)."
        if locale.startswith("zh")
        else "## Output language\nRespond in English."
    )
    request = ChatCompletionRequest(
        model=getattr(llm_service, "_model_id", "") or "",
        messages=[
            ChatMessage(role="system", content=TASK_DIAGNOSE_SYSTEM),
            ChatMessage(
                role="user",
                content=f"{user_loc}\n\n## Diagnostic bundle\n```json\n{json.dumps(compact, ensure_ascii=False)}\n```",
            ),
        ],
        temperature=0.2,
        top_p=0.9,
        max_tokens=900,
        stream=False,
    )
    if not request.model:
        raise RuntimeError("No default LLM model configured for task diagnosis")
    result = llm_service.chat_completion(request)
    return str(result.choices[0].message.content or "").strip()

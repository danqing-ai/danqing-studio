"""Resolve Hugging Face repo identifiers from registry placeholders.

Reserved for mapping stable sentinels in ``models_registry.json`` to actual
``repo_id`` strings used by ``huggingface_hub`` at download time.
"""

from __future__ import annotations


def resolve_huggingface_repo_id(repo_id: str | None) -> str | None:
    return repo_id

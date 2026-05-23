"""Registry ``bundle_repos`` — ordered list of repos that form one installable model version."""
from __future__ import annotations

from typing import Any


def _normalize_entry(item: dict[str, Any]) -> dict[str, Any]:
    repo = item.get("repo_id")
    lp = item.get("local_path")
    if not repo or not lp:
        raise ValueError(
            "Each bundle_repos entry requires repo_id and local_path: "
            f"repo_id={repo!r} local_path={lp!r}"
        )
    out: dict[str, Any] = {
        "repo_id": str(repo).strip(),
        "local_path": str(lp).strip(),
        "name": item.get("name") or str(repo).split("/")[-1],
    }
    if item.get("size"):
        out["size"] = item["size"]
    if item.get("source"):
        out["source"] = str(item["source"]).strip().lower()
    if item.get("allow_patterns"):
        out["allow_patterns"] = item["allow_patterns"]
    return out


def bundle_repos_from_version(ver_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return normalized ``bundle_repos`` for a version entry (empty if absent)."""
    if not ver_config:
        return []
    raw = ver_config.get("bundle_repos")
    if not isinstance(raw, list) or not raw:
        return []
    return [_normalize_entry(item) for item in raw if isinstance(item, dict)]


def require_bundle_repos(ver_config: dict[str, Any], *, model_id: str = "") -> list[dict[str, Any]]:
    """Like ``bundle_repos_from_version`` but fail loud when the version has no bundle_repos."""
    entries = bundle_repos_from_version(ver_config)
    if not entries:
        label = model_id or "model"
        raise ValueError(
            f"{label}: version config must define non-empty bundle_repos "
            "(repo_id + local_path per repo, in download order)"
        )
    return entries


def primary_and_follow_ups(
    ver_config: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    entries = bundle_repos_from_version(ver_config)
    if not entries:
        return None, []
    return entries[0], entries[1:]


def version_primary_local_path(ver_config: dict[str, Any]) -> str:
    """Install root: first bundle_repos path, else legacy single-repo ``local_path``."""
    entries = bundle_repos_from_version(ver_config)
    if entries:
        return entries[0]["local_path"]
    lp = ver_config.get("local_path")
    if isinstance(lp, str) and lp.strip():
        return lp.strip()
    raise ValueError("version config requires bundle_repos[0].local_path or local_path")


def bundle_local_paths(ver_config: dict[str, Any] | None) -> list[str]:
    """Distinct local_path values for uninstall / disk accounting."""
    seen: set[str] = set()
    out: list[str] = []
    for e in bundle_repos_from_version(ver_config):
        lp = e["local_path"]
        if lp not in seen:
            seen.add(lp)
            out.append(lp)
    return out

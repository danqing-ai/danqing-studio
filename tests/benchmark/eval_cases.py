"""Registry-driven image eval cases with a shared prompt pack."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from .registry_utils import (
    EDIT_ACTIONS,
    bundle_ready,
    iter_image_eval_models,
    load_registry,
    param_default,
    resolve_eval_version_key,
)

Profile = Literal["smoke", "full"]

PROMPTS_PATH = Path(__file__).resolve().parent / "prompts.json"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
EDIT_SOURCE = FIXTURES_DIR / "edit_source.png"
EDIT_MASK = FIXTURES_DIR / "edit_mask.png"
UPSCALE_SOURCE = FIXTURES_DIR / "upscale_source.png"
GOLDEN_PATH = Path(__file__).resolve().parent / "golden" / "eval_scores.json"

EVAL_SIZE = 384
EVAL_SEED = 42
EVAL_IMAGE_STRENGTH = 0.65
EVAL_UPSCALE_SCALE = 2
EVAL_EXTEND_PIXELS = 256
EVAL_EXTEND_DIRECTIONS: tuple[str, ...] = ("right",)
SMOKE_EDIT_PROMPT_ID = "E2"
SMOKE_EDIT_ACTION_ORDER: tuple[str, ...] = ("rewrite", "retouch", "extend")

MODEL_TIMEOUT_SEC: dict[str, int] = {
    "qwen-image-edit": 1200,
    "firered-image-edit-1.1": 1200,
    "cogview4-6b": 900,
    "fibo": 900,
    "fibo-lite": 900,
    "fibo-edit": 900,
    "fibo-edit-rmbg": 900,
    "hidream-o1-image-dev": 3600,
    "hidream-o1-image-full": 5400,
}

MODEL_STEPS_OVERRIDE: dict[str, int] = {
    "cogview4-6b": 4,
    "fibo": 8,
    "fibo-lite": 8,
    "fibo-edit": 8,
    "fibo-edit-rmbg": 8,
}


@dataclass
class EvalCase:
    id: str
    model_id: str
    family: str
    action: str
    prompt_id: str
    prompt_text: str
    judge_prompt: str
    seed: int = EVAL_SEED
    width: int = EVAL_SIZE
    height: int = EVAL_SIZE
    steps: int = 4
    guidance: float = 3.5
    image_strength: float = EVAL_IMAGE_STRENGTH
    upscale_scale: int = EVAL_UPSCALE_SCALE
    timeout_sec: int = 600
    omit_image_strength: bool = False
    judge_floor: float | None = None
    version_key: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def model_field(self) -> str:
        vk = str(self.version_key or "").strip()
        if vk:
            return f"{self.model_id}:{vk}"
        return self.model_id

    @property
    def encoded_prompt(self) -> str:
        return encode_prompt(family=self.family, action=self.action, text=self.prompt_text)

    @property
    def l1_expected_width(self) -> int | None:
        if self.action == "create":
            return self.width
        if self.action == "extend":
            w = EVAL_SIZE
            if "left" in EVAL_EXTEND_DIRECTIONS:
                w += EVAL_EXTEND_PIXELS
            if "right" in EVAL_EXTEND_DIRECTIONS:
                w += EVAL_EXTEND_PIXELS
            return w
        if self.action in EDIT_ACTIONS:
            return EVAL_SIZE
        if self.action == "upscale":
            return EVAL_SIZE * int(self.upscale_scale or EVAL_UPSCALE_SCALE)
        return None

    @property
    def l1_expected_height(self) -> int | None:
        if self.action == "create":
            return self.height
        if self.action == "extend":
            h = EVAL_SIZE
            if "top" in EVAL_EXTEND_DIRECTIONS:
                h += EVAL_EXTEND_PIXELS
            if "bottom" in EVAL_EXTEND_DIRECTIONS:
                h += EVAL_EXTEND_PIXELS
            return h
        if self.action in EDIT_ACTIONS:
            return EVAL_SIZE
        if self.action == "upscale":
            return EVAL_SIZE * int(self.upscale_scale or EVAL_UPSCALE_SCALE)
        return None


def load_prompt_pack() -> dict[str, Any]:
    return json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))


def fixture_scene(pack: dict[str, Any] | None = None, *, key: str) -> str:
    pack = pack or load_prompt_pack()
    fixtures = pack.get("fixtures") or {}
    return str(fixtures.get(key) or "").strip()


def encode_prompt(*, family: str, action: str, text: str) -> str:
    fam = (family or "").strip().lower()
    act = (action or "").strip().lower()
    body = (text or "").strip()
    if fam == "fibo" and act == "create":
        return json.dumps({"description": body}, ensure_ascii=False)
    return body


def edit_judge_prompt(*, scene: str, instruction: str) -> str:
    """PickScore prompt for rewrite/retouch edits.

    The fixture image already encodes ``scene``; judge on the edit instruction only
    so PickScore reflects whether the edit was applied (scene+instruction scores ~0.17).
    """
    _ = scene
    return (instruction or "").strip()


def _prompt_subset(pack: dict[str, Any], profile: Profile) -> tuple[list[dict], list[dict]]:
    create = list(pack.get("create") or [])
    edit = list(pack.get("edit") or [])
    if profile == "smoke":
        create = create[:1]
        smoke_edit = next((item for item in edit if str(item.get("id") or "") == SMOKE_EDIT_PROMPT_ID), None)
        edit = [smoke_edit] if smoke_edit else edit[:1]
    return create, edit


def _case_id(model_id: str, prompt_id: str, action: str) -> str:
    return f"{model_id}:{prompt_id}:{action}"


def _timeout_for(model_id: str) -> int:
    return int(MODEL_TIMEOUT_SEC.get(model_id, 600))


def _optional_judge_floor(item: dict[str, Any]) -> float | None:
    raw = item.get("judge_floor")
    if raw is None:
        return None
    return float(raw)


def _omit_image_strength(model_id: str, family: str) -> bool:
    if model_id in {
        "qwen-image-edit",
        "firered-image-edit-1.1",
        "fibo-edit",
        "fibo-edit-rmbg",
    }:
        return True
    if family == "fibo" and model_id.startswith("fibo-edit"):
        return True
    return False


def expand_eval_cases(*, profile: Profile = "full", reg: dict[str, Any] | None = None) -> list[EvalCase]:
    reg = reg or load_registry()
    pack = load_prompt_pack()
    create_prompts, edit_prompts = _prompt_subset(pack, profile)
    edit_scene = fixture_scene(pack, key="edit_scene")
    upscale_meta = pack.get("upscale") or {}
    upscale_judge = str(upscale_meta.get("judge_prompt") or "").strip()
    if not upscale_judge:
        upscale_judge = fixture_scene(pack, key="upscale_scene")
    upscale_judge_floor = _optional_judge_floor(upscale_meta)
    upscale_id = str(upscale_meta.get("id") or "U1")

    cases: list[EvalCase] = []
    for model_id, spec in iter_image_eval_models(reg=reg):
        family = str(spec.get("family") or "")
        actions = spec.get("actions") or {}
        steps = int(MODEL_STEPS_OVERRIDE.get(model_id, param_default(spec, "steps", 4)))
        guidance = float(param_default(spec, "guidance", 3.5))
        timeout = _timeout_for(model_id)
        omit_strength = _omit_image_strength(model_id, family)
        version_key = resolve_eval_version_key(model_id, reg=reg)

        if "create" in actions:
            for item in create_prompts:
                pid = str(item.get("id") or "")
                text = str(item.get("text") or "").strip()
                if not pid or not text:
                    continue
                cases.append(
                    EvalCase(
                        id=_case_id(model_id, pid, "create"),
                        model_id=model_id,
                        family=family,
                        action="create",
                        prompt_id=pid,
                        prompt_text=text,
                        judge_prompt=text,
                        steps=steps,
                        guidance=guidance,
                        timeout_sec=timeout,
                        version_key=version_key,
                    )
                )

        edit_actions = sorted(a for a in actions if a in EDIT_ACTIONS)
        if profile == "smoke":
            if "rewrite" not in actions:
                edit_actions = []
            elif edit_actions:
                edit_actions = ["rewrite"]

        for action in edit_actions:
            for item in edit_prompts:
                pid = str(item.get("id") or "")
                text = str(item.get("text") or "").strip()
                if not pid or not text:
                    continue
                cases.append(
                    EvalCase(
                        id=_case_id(model_id, pid, action),
                        model_id=model_id,
                        family=family,
                        action=action,
                        prompt_id=pid,
                        prompt_text=text,
                        judge_prompt=edit_judge_prompt(scene=edit_scene, instruction=text),
                        judge_floor=_optional_judge_floor(item),
                        steps=steps,
                        guidance=guidance,
                        timeout_sec=timeout,
                        omit_image_strength=omit_strength,
                        version_key=version_key,
                    )
                )

        if "upscale" in actions and profile == "full":
            cases.append(
                EvalCase(
                    id=_case_id(model_id, upscale_id, "upscale"),
                    model_id=model_id,
                    family=family,
                    action="upscale",
                    prompt_id=upscale_id,
                    prompt_text="",
                    judge_prompt=upscale_judge,
                    judge_floor=upscale_judge_floor,
                    steps=1,
                    guidance=0.0,
                    timeout_sec=timeout,
                    version_key=version_key,
                )
            )

    return cases


def iter_runnable_eval_cases(*, profile: Profile = "full") -> list[EvalCase]:
    return [c for c in expand_eval_cases(profile=profile) if bundle_ready(c.model_id)[0]]


def list_skipped_eval_cases(*, profile: Profile = "full") -> list[tuple[str, str]]:
    skipped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for case in expand_eval_cases(profile=profile):
        if case.model_id in seen:
            continue
        ready, reason = bundle_ready(case.model_id)
        if ready:
            seen.add(case.model_id)
            continue
        seen.add(case.model_id)
        skipped.append((case.model_id, reason or "missing default bundle"))
    return skipped


def get_eval_case(case_id: str, *, profile: Profile = "full") -> EvalCase | None:
    for case in expand_eval_cases(profile=profile):
        if case.id == case_id:
            return case
    return None


def list_eval_case_ids(*, profile: Profile = "full") -> list[str]:
    return [c.id for c in expand_eval_cases(profile=profile)]


def load_golden_scores() -> dict[str, Any]:
    if not GOLDEN_PATH.is_file():
        return {"schema_version": 1, "judge_model": "yuvalkirstain/PickScore_v1", "cases": {}}
    data = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    if "cases" not in data:
        data["cases"] = {}
    return data


def golden_reward(case_id: str) -> float | None:
    data = load_golden_scores()
    node = (data.get("cases") or {}).get(case_id)
    if not isinstance(node, dict):
        return None
    reward = node.get("reward")
    if reward is None:
        return None
    return float(reward)


def save_golden_scores(data: dict[str, Any]) -> None:
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_edit_fixture(arr) -> None:
    import numpy as np

    h, w = EVAL_SIZE, EVAL_SIZE
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    sky = np.stack(
        [
            180 + 40 * (1 - y),
            150 + 30 * (1 - y),
            120 + 20 * (1 - y),
        ],
        axis=-1,
    )
    arr[:] = sky.astype(np.uint8)
    table_y0 = int(h * 0.62)
    arr[table_y0:, :, 0] = np.clip(arr[table_y0:, :, 0] * 0.45 + 70, 0, 255)
    arr[table_y0:, :, 1] = np.clip(arr[table_y0:, :, 1] * 0.45 + 45, 0, 255)
    arr[table_y0:, :, 2] = np.clip(arr[table_y0:, :, 2] * 0.45 + 25, 0, 255)
    win_x0, win_x1 = int(w * 0.08), int(w * 0.35)
    win_y0, win_y1 = int(h * 0.12), int(h * 0.45)
    arr[win_y0:win_y1, win_x0:win_x1, 0] = 255
    arr[win_y0:win_y1, win_x0:win_x1, 1] = 240
    arr[win_y0:win_y1, win_x0:win_x1, 2] = 180
    cy, cx, radius = int(h * 0.72), int(w * 0.55), int(min(h, w) * 0.07)
    yy, xx = np.ogrid[:h, :w]
    cup = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    arr[cup, 0] = 210
    arr[cup, 1] = 45
    arr[cup, 2] = 45


def _render_upscale_fixture(arr) -> None:
    import numpy as np

    h, w = EVAL_SIZE, EVAL_SIZE
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    arr[:, :, 0] = np.clip(220 - 40 * y + 25 * np.sin(x * 28), 0, 255).astype(np.uint8)
    arr[:, :, 1] = np.clip(210 - 30 * y + 20 * np.cos(y * 24), 0, 255).astype(np.uint8)
    arr[:, :, 2] = np.clip(180 - 20 * y, 0, 255).astype(np.uint8)
    bowl_cy, bowl_cx = int(h * 0.58), int(w * 0.5)
    yy, xx = np.ogrid[:h, :w]
    bowl = (xx - bowl_cx) ** 2 + (yy - bowl_cy) ** 2 <= int(min(h, w) * 0.22) ** 2
    arr[bowl] = (235, 225, 210)
    fruit = [
        (int(h * 0.52), int(w * 0.42), (220, 40, 40)),
        (int(h * 0.48), int(w * 0.52), (240, 180, 40)),
        (int(h * 0.55), int(w * 0.58), (80, 170, 70)),
        (int(h * 0.5), int(w * 0.63), (170, 90, 200)),
    ]
    r = int(min(h, w) * 0.045)
    for fy, fx, color in fruit:
        mask = (xx - fx) ** 2 + (yy - fy) ** 2 <= r**2
        arr[mask] = color


def _fixture_size_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        from PIL import Image

        with Image.open(path) as img:
            return img.size == (EVAL_SIZE, EVAL_SIZE)
    except OSError:
        return False


def _write_fixture(path: Path, render_fn) -> Path:
    from PIL import Image
    import numpy as np

    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    arr = np.zeros((EVAL_SIZE, EVAL_SIZE, 3), dtype=np.uint8)
    render_fn(arr)
    Image.fromarray(arr).save(path)
    return path


def ensure_edit_source() -> Path:
    if not _fixture_size_ok(EDIT_SOURCE):
        _write_fixture(EDIT_SOURCE, _render_edit_fixture)
    return EDIT_SOURCE


def ensure_edit_mask() -> Path:
    """Full-white mask matching ``edit_source`` (retouch repaint region)."""
    from PIL import Image

    ensure_edit_source()
    if not _fixture_size_ok(EDIT_MASK):
        FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (EVAL_SIZE, EVAL_SIZE), color=(255, 255, 255)).save(EDIT_MASK)
    return EDIT_MASK


def ensure_upscale_source() -> Path:
    if not _fixture_size_ok(UPSCALE_SOURCE):
        _write_fixture(UPSCALE_SOURCE, _render_upscale_fixture)
    return UPSCALE_SOURCE

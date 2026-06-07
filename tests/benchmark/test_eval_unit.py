"""Unit tests for image eval benchmark (no model weights / no PickScore load)."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from tests.benchmark.eval_cases import (  # noqa: E402
    EVAL_SIZE,
    SMOKE_EDIT_PROMPT_ID,
    edit_judge_prompt,
    encode_prompt,
    ensure_edit_source,
    ensure_upscale_source,
    expand_eval_cases,
    fixture_scene,
    load_prompt_pack,
)
from tests.benchmark.integrity import check_output_image_integrity  # noqa: E402
from tests.benchmark.judge import required_score  # noqa: E402


class EvalCaseExpansionTests(unittest.TestCase):
    def test_prompt_pack_has_core_ids(self) -> None:
        pack = load_prompt_pack()
        create_ids = {p["id"] for p in pack["create"]}
        edit_ids = {p["id"] for p in pack["edit"]}
        self.assertIn("P1", create_ids)
        self.assertIn("E1", edit_ids)
        self.assertTrue(fixture_scene(pack, key="edit_scene"))

    def test_smoke_smaller_than_full(self) -> None:
        smoke = expand_eval_cases(profile="smoke")
        full = expand_eval_cases(profile="full")
        self.assertLess(len(smoke), len(full))

    def test_smoke_edit_uses_e2(self) -> None:
        smoke = expand_eval_cases(profile="smoke")
        edit_cases = [c for c in smoke if c.action == "rewrite"]
        if edit_cases:
            self.assertEqual(edit_cases[0].prompt_id, SMOKE_EDIT_PROMPT_ID)

    def test_edit_judge_prompt_includes_scene(self) -> None:
        pack = load_prompt_pack()
        scene = fixture_scene(pack, key="edit_scene")
        out = edit_judge_prompt(scene=scene, instruction="make colors vivid")
        self.assertIn(scene, out)
        self.assertIn("vivid", out)

    def test_fibo_create_json_encode(self) -> None:
        text = "a red apple"
        out = encode_prompt(family="fibo", action="create", text=text)
        data = json.loads(out)
        self.assertEqual(data["description"], text)


class IntegrityTests(unittest.TestCase):
    def test_valid_png_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ok.png"
            arr = np.random.default_rng(0).integers(0, 256, (EVAL_SIZE, EVAL_SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)
            res = check_output_image_integrity(path, expected_width=EVAL_SIZE, expected_height=EVAL_SIZE)
            self.assertTrue(res.ok, res.reason)

    def test_blank_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blank.png"
            Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8)).save(path)
            res = check_output_image_integrity(path)
            self.assertFalse(res.ok)

    def test_size_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.png"
            arr = np.random.default_rng(0).integers(0, 256, (256, 256, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)
            res = check_output_image_integrity(path, expected_width=EVAL_SIZE, expected_height=EVAL_SIZE)
            self.assertFalse(res.ok)
            self.assertIn("width_mismatch", res.reason)


class FixtureTests(unittest.TestCase):
    def test_fixtures_are_eval_size(self) -> None:
        edit = ensure_edit_source()
        upscale = ensure_upscale_source()
        with Image.open(edit) as img:
            self.assertEqual(img.size, (EVAL_SIZE, EVAL_SIZE))
        with Image.open(upscale) as img:
            self.assertEqual(img.size, (EVAL_SIZE, EVAL_SIZE))


class JudgeThresholdTests(unittest.TestCase):
    def test_required_score_uses_golden(self) -> None:
        self.assertAlmostEqual(required_score(golden=1.0), 0.85)
        self.assertAlmostEqual(required_score(golden=None), 0.20)


class RunnerMockJudgeTests(unittest.TestCase):
    def test_run_one_passes_with_mock_judge(self) -> None:
        from tests.benchmark.eval_cases import EvalCase
        from tests.benchmark.judge import JudgeResult
        from tests.benchmark.runner import EvalRunner

        case = EvalCase(
            id="test:P1:create",
            model_id="test",
            family="flux2",
            action="create",
            prompt_id="P1",
            prompt_text="a red apple",
            judge_prompt="a red apple",
        )
        with tempfile.TemporaryDirectory() as tmp:
            runner = EvalRunner(output_dir=tmp)
            out = runner._output_path(case)
            arr = np.random.default_rng(1).integers(0, 256, (EVAL_SIZE, EVAL_SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(out)
            with patch.object(runner, "_run_generate", return_value=True):
                with patch(
                    "tests.benchmark.runner.judge_image",
                    return_value=JudgeResult(ok=True, score=0.5, min_required=0.20),
                ):
                    res = runner.run_one(case)
            self.assertTrue(res.ok)


if __name__ == "__main__":
    unittest.main()

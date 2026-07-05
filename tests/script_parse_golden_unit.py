"""Golden ScriptArtifact fixture validation (no LLM)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.engine.llm.script_parse.schemas import ScriptArtifact

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "script_parse"


class ScriptParseGoldenTests(unittest.TestCase):
    def test_rainy_night_decompose_fixture(self) -> None:
        raw = json.loads((FIXTURES / "rainy_night_decompose.json").read_text(encoding="utf-8"))
        art = ScriptArtifact.model_validate(raw)
        self.assertEqual(art.version, "2.0")
        self.assertEqual(len(art.beats), 2)
        protagonists = [c for c in art.characters if c.role == "protagonist"]
        self.assertEqual(len(protagonists), 1)
        self.assertEqual(protagonists[0].name, "林晓")


if __name__ == "__main__":
    unittest.main(verbosity=2)

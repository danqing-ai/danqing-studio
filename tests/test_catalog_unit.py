"""Catalog v3 loader, migration, and API DTO tests."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from backend.catalog.api_dto import build_catalog_response
from backend.catalog.expand_v2 import expand_registry_document
from backend.catalog.loader import expand_catalog_document, flatten_v3_model
from backend.catalog.migrate_v2 import migrate_v2_to_v3
from backend.catalog.validate import validate_v3_document
from backend.core.model_registry import ModelRegistry

ROOT = Path(__file__).resolve().parents[1]
FACTORY_REGISTRY = ROOT / "default_config" / "models_registry.json"
V2_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "registry_v2_minimal.json"


class CatalogMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not V2_FIXTURE.is_file():
            raise unittest.SkipTest("registry v2 fixture missing")
        cls.v2_data = json.loads(V2_FIXTURE.read_text(encoding="utf-8"))

    def test_migrate_minimal_registry(self) -> None:
        migrated, _report = migrate_v2_to_v3(self.v2_data)
        self.assertEqual(migrated["schema_version"], 3)
        self.assertIn("ui_profiles", migrated)
        self.assertIn("families", migrated)
        self.assertIn("flux2", migrated["families"])
        errors = validate_v3_document(migrated)
        self.assertEqual(errors, [], msg="\n".join(errors))

    def test_flux2_klein_9b_roundtrip_parameters(self) -> None:
        migrated, _ = migrate_v2_to_v3(self.v2_data)
        v2_expanded = expand_registry_document(self.v2_data)["models"]["flux2-klein-9b"]
        v3_flat = expand_catalog_document(migrated)["models"]["flux2-klein-9b"]

        self.assertEqual(v3_flat["family"], "flux2")
        self.assertEqual(v3_flat["name"], v2_expanded["name"])
        self.assertEqual(
            v3_flat["parameters"].get("vae_scale"),
            v2_expanded["parameters"].get("vae_scale"),
        )
        self.assertEqual(
            v3_flat["parameters"]["steps"]["default"],
            v2_expanded["parameters"]["steps"]["default"],
        )

    def test_model_registry_loads_v3_migrated(self) -> None:
        migrated, _ = migrate_v2_to_v3(self.v2_data)
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(migrated, f)
            tmp_path = Path(f.name)
        try:
            reg = ModelRegistry.load(tmp_path)
            entry = reg.require("flux2-klein-9b")
            self.assertEqual(entry.family, "flux2")
            self.assertIn("generate", entry.actions)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_build_catalog_response_index(self) -> None:
        migrated, _ = migrate_v2_to_v3(self.v2_data)
        index = {
            "flux2-klein-9b": {
                "media": "image",
                "family": "flux2",
                "engine": "danqing-image",
                "actions": ["create"],
            }
        }
        resp = build_catalog_response(migrated, index=index)
        self.assertEqual(resp["schema_version"], 3)
        self.assertIn("families", resp)
        self.assertIn("flux2-klein-9b", resp["models"])
        self.assertEqual(resp["_index"], index)

    def test_flatten_v3_requires_family(self) -> None:
        doc = {
            "ui_profiles": {},
            "families": {},
            "models": {
                "x": {
                    "catalog": {"name": {"en": "x"}},
                    "runtime": {},
                    "actions": {},
                }
            },
        }
        with self.assertRaises(ValueError):
            flatten_v3_model("x", doc["models"]["x"], doc)

    @unittest.skipUnless(FACTORY_REGISTRY.is_file(), "factory registry missing")
    def test_factory_registry_is_v3_and_valid(self) -> None:
        data = json.loads(FACTORY_REGISTRY.read_text(encoding="utf-8"))
        self.assertGreaterEqual(int(data.get("schema_version", 0)), 3)
        errors = validate_v3_document(data)
        self.assertEqual(errors, [], msg="\n".join(errors))


if __name__ == "__main__":
    unittest.main()

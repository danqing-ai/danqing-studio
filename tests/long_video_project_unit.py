"""Unit tests for long-video project persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from backend.persistence.long_video_project_store import LongVideoProjectStore


class LongVideoProjectStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        db_path = Path(self._tmpdir.name) / "test.db"
        self.store = LongVideoProjectStore(db_path)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_create_list_get_update_delete(self):
        created = self.store.create_project(title="Demo", state={"version": 1, "shots": []})
        self.assertTrue(created["id"].startswith("lvp_"))
        self.assertEqual(created["title"], "Demo")

        items = self.store.list_projects()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["shot_count"], 0)

        fetched = self.store.get_project(created["id"])
        assert fetched is not None
        self.assertEqual(fetched["title"], "Demo")

        updated = self.store.update_project(
            created["id"],
            title="Renamed",
            state={
                "version": 1,
                "shots": [{"id": "s1", "order": 0, "visual_prompt": "a", "motion_prompt": ""}],
            },
        )
        assert updated is not None
        self.assertEqual(updated["title"], "Renamed")
        self.assertEqual(len(updated["state"]["shots"]), 1)

        listed = self.store.list_projects()
        self.assertEqual(listed[0]["shot_count"], 1)

        self.assertTrue(self.store.delete_project(created["id"]))
        self.assertIsNone(self.store.get_project(created["id"]))


if __name__ == "__main__":
    unittest.main()

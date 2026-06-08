"""Unit tests for engine v3 observability (RunTrace, diagnostic bundle)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.observability.diagnostic import build_diagnostic_bundle
from backend.observability.error_codes import ErrorCode, classify_failed_span
from backend.observability.graph_runtime import graph_id_for_task_kind, load_graph_manifest, snapshot
from backend.observability.trace import RunTrace


class RunTraceTests(unittest.TestCase):
    def test_ingest_legacy_graph_steps(self) -> None:
        trace = RunTrace("tsk_" + "a" * 24, graph_id="image_create")
        trace.ingest_log_line("info", "[validate_bundle] ok family=flux2")
        trace.ingest_log_line("info", "[encode_prompt] start")
        trace.ingest_log_line("info", "[denoise] start")
        names = [s.name for s in trace.spans]
        self.assertIn("validate_bundle", names)
        self.assertIn("encode", names)
        self.assertIn("denoise", names)

    def test_failure_on_error_log(self) -> None:
        trace = RunTrace("tsk_" + "b" * 24)
        trace.ingest_log_line("error", "[load_transformer] missing key foo")
        self.assertIsNotNone(trace.failure)
        self.assertEqual(trace.failure.code, ErrorCode.WEIGHT_KEY_MISMATCH)

    def test_update_callback_on_span(self) -> None:
        trace = RunTrace("tsk_" + "e" * 24)
        seen: list[str] = []
        trace.set_update_callback(lambda: seen.append("ok"))
        with trace.span_ctx("encode", kind="phase"):
            pass
        self.assertGreaterEqual(len(seen), 2)

    def test_save_and_load_roundtrip(self) -> None:
        trace = RunTrace("tsk_" + "c" * 24)
        trace.ingest_log_line("info", "[validate_bundle] ok")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.json"
            trace.save(path)
            data = RunTrace.load(path)
            self.assertIsNotNone(data)
            self.assertEqual(len(data.get("spans") or []), 1)


class DiagnosticBundleTests(unittest.TestCase):
    def test_build_from_logs_when_no_trace_file(self) -> None:
        logs = [
            {"level": "info", "message": "[validate_bundle] ok", "time": "t0"},
            {"level": "info", "message": "[encode_prompt] done", "time": "t1"},
            {"level": "error", "message": "CUDA not available", "time": "t2"},
        ]
        bundle = build_diagnostic_bundle(
            task_id="tsk_" + "d" * 24,
            task_row={
                "status": "failed",
                "kind": "image.generation",
                "error": "CUDA not available",
                "params": {"model": "flux2-dev", "steps": 4},
            },
            logs=logs,
            work_dir=None,
        )
        self.assertEqual(bundle["status"], "failed")
        self.assertIsNotNone(bundle["failure"])
        self.assertIn("graph", bundle)
        self.assertTrue(bundle["context"]["classified_without_trace_file"])

    def test_classify_failed_span_bundle(self) -> None:
        self.assertEqual(
            classify_failed_span("validate_bundle", "bundle incomplete"),
            ErrorCode.BUNDLE_NOT_READY,
        )


class GraphManifestTests(unittest.TestCase):
    def test_image_create_manifest_loads(self) -> None:
        manifest = load_graph_manifest("image_create")
        self.assertEqual(manifest["id"], "image_create")
        self.assertGreaterEqual(len(manifest.get("nodes") or []), 5)

    def test_snapshot_pending_when_empty_trace(self) -> None:
        trace = RunTrace("tsk_" + "e" * 24)
        snap = snapshot(trace, graph_id_for_task_kind("image.generation"))
        self.assertEqual(snap.graph_id, "image_create")
        self.assertTrue(all(n.status == "pending" for n in snap.nodes))

    def test_graph_id_per_task_kind(self) -> None:
        self.assertEqual(graph_id_for_task_kind("image.edit"), "image_edit")
        self.assertEqual(graph_id_for_task_kind("image.upscale"), "image_upscale")
        self.assertEqual(graph_id_for_task_kind("video.generation"), "video_create")
        self.assertEqual(graph_id_for_task_kind("audio.generation"), "audio_create")

    def test_v3_graph_manifests_load(self) -> None:
        for graph_id in ("image_edit", "image_upscale", "video_create", "audio_create"):
            manifest = load_graph_manifest(graph_id)
            self.assertEqual(manifest["id"], graph_id)
            self.assertGreaterEqual(len(manifest.get("nodes") or []), 4)


if __name__ == "__main__":
    unittest.main()

"""Acceptance test: Batch preds.json schema smoke test (Section L)."""

import json

import pytest


class TestBatchPredsSchema:
    """preds.json must have required fields: instance_id, model_name_or_path, model_patch."""

    REQUIRED_FIELDS = {"instance_id", "model_name_or_path", "model_patch"}

    def test_preds_schema_required_fields(self, tmp_path):
        """A valid preds.json entry has all required fields."""
        preds = {
            "task_001": {
                "instance_id": "task_001",
                "model_name_or_path": "claude-sonnet-4-6",
                "model_patch": "diff --git a/file b/file\n...",
            },
            "task_002": {
                "instance_id": "task_002",
                "model_name_or_path": "claude-sonnet-4-6",
                "model_patch": "",
            },
        }

        # Validate schema
        for key, entry in preds.items():
            assert key == entry["instance_id"], f"Key '{key}' != instance_id '{entry['instance_id']}'"
            missing = self.REQUIRED_FIELDS - set(entry.keys())
            assert not missing, f"Entry '{key}' missing fields: {missing}"

    def test_preds_json_roundtrip(self, tmp_path):
        """preds.json can be written and read back."""
        preds = {
            "task_001": {
                "instance_id": "task_001",
                "model_name_or_path": "claude-sonnet-4-6",
                "model_patch": "patch content",
            },
        }

        output_path = tmp_path / "preds.json"
        with open(output_path, "w") as f:
            json.dump(preds, f)

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded == preds
        assert loaded["task_001"]["instance_id"] == "task_001"

    def test_preds_with_error_field(self, tmp_path):
        """Entries can have optional error field."""
        preds = {
            "task_001": {
                "instance_id": "task_001",
                "model_name_or_path": "claude-sonnet-4-6",
                "model_patch": "",
                "error": "Model API timeout",
            },
        }

        output_path = tmp_path / "preds.json"
        with open(output_path, "w") as f:
            json.dump(preds, f)

        with open(output_path) as f:
            loaded = json.load(f)

        assert loaded["task_001"]["error"] == "Model API timeout"

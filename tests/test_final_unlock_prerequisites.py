import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from medstyleaudit.protocol import (
    FINAL_HOSPITAL1_SETTINGS,
    FINAL_TRAINING_SEEDS,
    verify_final_test_prerequisites,
    write_protocol_lock,
)


def _stage_names():
    names = [f"train_resnet50_{seed:04d}" for seed in FINAL_TRAINING_SEEDS]
    for seed in FINAL_TRAINING_SEEDS:
        names.extend(f"audit_hospital1_{setting}_{seed:04d}" for setting in ("primary", "random_paired", "roi_only"))
        names.extend(f"robustness_hospital1_{setting}_{seed:04d}" for setting in ("buffer_r8", "hard_boundary"))
    return [*names, "aggregate_hospital1"]


def _complete_prerequisites(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    protocol = tmp_path / "FINAL_PROTOCOL.yaml"
    protocol.write_text("status: frozen\n", encoding="utf-8")
    digest = write_protocol_lock(protocol, root / "protocol" / "protocol_sha256.txt")
    monkeypatch.setattr("medstyleaudit.protocol.git_state", lambda repository=".": ("abc", False))
    stage_root = root / ".stage_state"
    stage_root.mkdir(parents=True)
    persisted_output = root / "hospital1-stage-output.txt"
    persisted_output.write_text("complete", encoding="utf-8")
    payload = {"status": "completed", "git_commit": "abc", "protocol_hash": digest, "stage_signature": ["scientific-stage"], "outputs": [str(persisted_output)]}
    for name in _stage_names():
        (stage_root / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    aggregate = root / "aggregate" / "hospital1"
    aggregate.mkdir(parents=True)
    pd.DataFrame([
        {"population": "hospital1", "setting": setting, "expected_runs": 3, "completed_runs": 3, "status": "complete"}
        for setting in FINAL_HOSPITAL1_SETTINGS
    ]).to_csv(aggregate / "run_completeness.csv", index=False)
    return root, protocol, digest


@pytest.mark.parametrize("seed", FINAL_TRAINING_SEEDS)
def test_prerequisites_reject_each_missing_training_seed(tmp_path, monkeypatch, seed):
    root, protocol, _ = _complete_prerequisites(tmp_path, monkeypatch)
    (root / ".stage_state" / f"train_resnet50_{seed:04d}.json").unlink()
    with pytest.raises(PermissionError, match="stage marker is missing"):
        verify_final_test_prerequisites(root, protocol)


def test_prerequisites_reject_incomplete_required_audit(tmp_path, monkeypatch):
    root, protocol, _ = _complete_prerequisites(tmp_path, monkeypatch)
    marker = root / ".stage_state" / "audit_hospital1_random_paired_0042.json"
    data = json.loads(marker.read_text(encoding="utf-8")); data["status"] = "failed"
    marker.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PermissionError, match="stage is incomplete"):
        verify_final_test_prerequisites(root, protocol)


def test_prerequisites_reject_incomplete_aggregation(tmp_path, monkeypatch):
    root, protocol, _ = _complete_prerequisites(tmp_path, monkeypatch)
    path = root / "aggregate" / "hospital1" / "run_completeness.csv"
    table = pd.read_csv(path); table.loc[table["setting"] == "hard_boundary", ["completed_runs", "status"]] = [2, "incomplete"]
    table.to_csv(path, index=False)
    with pytest.raises(PermissionError, match="aggregation is incomplete"):
        verify_final_test_prerequisites(root, protocol)


@pytest.mark.parametrize("field,value", [("git_commit", "old"), ("protocol_hash", "old")])
def test_prerequisites_reject_stale_stage_provenance(tmp_path, monkeypatch, field, value):
    root, protocol, _ = _complete_prerequisites(tmp_path, monkeypatch)
    marker = root / ".stage_state" / "train_resnet50_0011.json"
    data = json.loads(marker.read_text(encoding="utf-8")); data[field] = value
    marker.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(PermissionError, match="stale Git/protocol provenance"):
        verify_final_test_prerequisites(root, protocol)


def test_unlock_succeeds_only_with_complete_current_prerequisites(tmp_path, monkeypatch):
    root, protocol, digest = _complete_prerequisites(tmp_path, monkeypatch)
    (root / "protocol" / "final_preflight.json").write_text(
        json.dumps({"status": "PASS", "protocol_hash": digest, "git_commit": "abc"}), encoding="utf-8"
    )
    script = Path(__file__).resolve().parents[1] / "scripts" / "15_unlock_final_test.py"
    spec = importlib.util.spec_from_file_location("unlock_final_test", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, "git_state", lambda repository=".": ("abc", False))
    destination = module.unlock_final_test(root, protocol)
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["status"] == "unlocked"
    assert payload["git_commit"] == "abc"
    assert payload["protocol_sha256"] == digest

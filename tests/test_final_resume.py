import argparse
import importlib.util
import json
from pathlib import Path


def test_final_runner_resumes_only_when_marker_and_outputs_are_valid(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str((tmp_path / "output").resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    script = Path(__file__).resolve().parents[1] / "scripts/run_final_suite.py"
    spec = importlib.util.spec_from_file_location("final_suite", script); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    suite = module.FinalSuite(argparse.Namespace(force=False, device="cpu", allow_final_test=False))
    output = tmp_path / "done.txt"; output.write_text("ok", encoding="utf-8")
    (suite.stage_root / "stage.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    assert suite.completed("stage", [output])
    output.unlink()
    assert not suite.completed("stage", [output])


def test_final_runner_rejects_stage_marker_from_another_commit_or_command(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str((tmp_path / "output").resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    script = Path(__file__).resolve().parents[1] / "scripts/run_final_suite.py"
    spec = importlib.util.spec_from_file_location("final_suite_provenance", script); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    suite = module.FinalSuite(argparse.Namespace(force=False, device="cpu", allow_final_test=False))
    output = tmp_path / "done.txt"; output.write_text("ok", encoding="utf-8")
    command = ["python", "stage.py"]
    commit, _ = module.git_state(module.REPO)
    protocol_hash = (module.REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
    marker = suite.stage_root / "stage.json"
    marker.write_text(json.dumps({"status": "completed", "command": command, "git_commit": commit, "protocol_hash": protocol_hash}), encoding="utf-8")
    assert suite.completed("stage", [output], command=command)
    assert not suite.completed("stage", [output], command=["python", "other.py"])
    marker.write_text(json.dumps({"status": "completed", "command": command, "git_commit": "old", "protocol_hash": protocol_hash}), encoding="utf-8")
    assert not suite.completed("stage", [output], command=command)

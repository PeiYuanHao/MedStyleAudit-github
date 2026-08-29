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


def test_completed_training_uses_scientific_signature_not_recovery_command(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str((tmp_path / "output").resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    script = Path(__file__).resolve().parents[1] / "scripts/run_final_suite.py"
    spec = importlib.util.spec_from_file_location("final_suite_provenance", script); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    suite = module.FinalSuite(argparse.Namespace(force=False, device="cpu", allow_final_test=False))
    output = tmp_path / "done.txt"; output.write_text("ok", encoding="utf-8")
    first_command = ["python", "scripts/05_train_erm.py", "--config", "model.yaml", "--seed", "11", "--device", "cuda:0"]
    recovery_command = first_command + ["--overwrite", "--resume", "last.ckpt"]
    signature = suite.training_stage_signature(first_command)
    commit, _ = module.git_state(module.REPO)
    monkeypatch.setattr(module, "git_state", lambda repository=module.REPO: (commit, False))
    protocol_hash = (module.REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
    marker = suite.stage_root / "train_resnet50_0011.json"
    marker.write_text(json.dumps({"status": "completed", "execution_command": first_command, "stage_signature": signature, "git_commit": commit, "protocol_hash": protocol_hash}), encoding="utf-8")
    assert suite.completed("train_resnet50_0011", [output], command=recovery_command, stage_signature=signature)

    marker.write_text(json.dumps({"status": "completed", "stage_signature": signature, "git_commit": "old", "protocol_hash": protocol_hash}), encoding="utf-8")
    assert not suite.completed("train_resnet50_0011", [output], command=recovery_command, stage_signature=signature)
    marker.write_text(json.dumps({"status": "completed", "stage_signature": signature, "git_commit": commit, "protocol_hash": "old"}), encoding="utf-8")
    assert not suite.completed("train_resnet50_0011", [output], command=recovery_command, stage_signature=signature)
    marker.write_text(json.dumps({"status": "completed", "stage_signature": signature, "git_commit": commit, "protocol_hash": protocol_hash}), encoding="utf-8")
    changed = ["scripts/05_train_erm.py", "--config", "other.yaml", "--seed", "11"]
    assert not suite.completed("train_resnet50_0011", [output], command=recovery_command, stage_signature=changed)


def test_legacy_completed_training_command_is_normalized(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str((tmp_path / "output").resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    script = Path(__file__).resolve().parents[1] / "scripts/run_final_suite.py"
    spec = importlib.util.spec_from_file_location("final_suite_legacy", script); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    suite = module.FinalSuite(argparse.Namespace(force=False, device="cpu", allow_final_test=False))
    output = tmp_path / "done.txt"; output.write_text("ok", encoding="utf-8")
    command = ["python", "scripts/05_train_erm.py", "--config", "model.yaml", "--seed", "11", "--device", "cuda"]
    commit, _ = module.git_state(module.REPO)
    monkeypatch.setattr(module, "git_state", lambda repository=module.REPO: (commit, False))
    protocol_hash = (module.REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
    marker = suite.stage_root / "train_resnet50_0011.json"
    marker.write_text(json.dumps({"status": "completed", "command": command, "git_commit": commit, "protocol_hash": protocol_hash}), encoding="utf-8")
    assert suite.completed("train_resnet50_0011", [output], command=command + ["--overwrite", "--resume", "last.ckpt"], stage_signature=suite.training_stage_signature(command))


def test_allow_final_test_requires_existing_explicit_unlock(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str((tmp_path / "output").resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    script = Path(__file__).resolve().parents[1] / "scripts/run_final_suite.py"
    spec = importlib.util.spec_from_file_location("final_suite_unlock", script); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    suite = module.FinalSuite(argparse.Namespace(force=False, device="cpu", allow_final_test=True))
    monkeypatch.setattr(module, "authorize_final_test", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("locked")))
    lock = suite.output / "protocol" / "final_test_lock.json"
    import pytest
    with pytest.raises(PermissionError, match="Run"):
        suite.require_explicit_final_test_unlock()
    assert not lock.exists()

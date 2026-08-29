import argparse
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
import yaml


def _load_final_suite_module(name):
    script = Path(__file__).resolve().parents[1] / "scripts/run_final_suite.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _partial_checkpoint(module, tmp_path, *, seed=42, info_updates=None, config_updates=None):
    seed_dir = tmp_path / f"seed_{seed:04d}"
    seed_dir.mkdir(parents=True)
    checkpoint = seed_dir / "last.ckpt"
    checkpoint.write_bytes(b"checkpoint-not-deserialized")
    current_config = module.load_config(module.REPO / "configs/models/resnet50_hf.yaml")
    recorded_config = deepcopy(current_config)
    recorded_config["training"]["resume"] = "recovery-only-last.ckpt"
    if config_updates:
        for section, values in config_updates.items():
            recorded_config[section].update(values)
    info = {
        "experiment": "train_erm_resnet50",
        "git_commit": "current-commit",
        "protocol_hash": "current-protocol",
        "protocol_sha256": "current-protocol",
        "seed": seed,
        "dataset_version": current_config["data"]["version"],
    }
    info.update(info_updates or {})
    (seed_dir / "run_info.json").write_text(json.dumps(info), encoding="utf-8")
    (seed_dir / "config_resolved.yaml").write_text(yaml.safe_dump(recorded_config), encoding="utf-8")
    return seed_dir, current_config, checkpoint

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


def test_valid_partial_checkpoint_resume_allows_recovery_only_config_difference(tmp_path):
    module = _load_final_suite_module("final_suite_partial_valid")
    seed_dir, current_config, checkpoint = _partial_checkpoint(module, tmp_path)
    assert module.validate_partial_training_resume(
        seed_dir, 42, current_config, "current-commit", "current-protocol"
    ) == checkpoint


def test_train_adds_resume_only_after_valid_partial_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str((tmp_path / "output").resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    module = _load_final_suite_module("final_suite_partial_train_integration")
    suite = module.FinalSuite(argparse.Namespace(force=False, device="cpu", allow_final_test=False))
    commit, _ = module.git_state(module.REPO)
    protocol_hash = (module.REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
    seed_dir, _, checkpoint = _partial_checkpoint(
        module,
        suite.output / "checkpoints" / module.BACKBONE,
        seed=42,
        info_updates={
            "git_commit": commit,
            "protocol_hash": protocol_hash,
            "protocol_sha256": protocol_hash,
        },
    )
    monkeypatch.setattr(module, "git_state", lambda repository=module.REPO: (commit, False))
    captured = {}
    monkeypatch.setattr(
        suite,
        "run",
        lambda name, command, outputs, summary, stage_signature: captured.update(
            name=name, command=command, stage_signature=stage_signature
        ),
    )
    suite.train(42, "cpu")
    assert seed_dir == checkpoint.parent
    assert captured["command"][-2:] == ["--resume", str(checkpoint)]


def test_partial_checkpoint_rejects_stale_git_commit(tmp_path):
    module = _load_final_suite_module("final_suite_partial_git")
    seed_dir, current_config, _ = _partial_checkpoint(
        module, tmp_path, info_updates={"git_commit": "old-commit"}
    )
    with pytest.raises(RuntimeError, match="git_commit mismatch"):
        module.validate_partial_training_resume(
            seed_dir, 42, current_config, "current-commit", "current-protocol"
        )


def test_partial_checkpoint_rejects_stale_protocol_hash(tmp_path):
    module = _load_final_suite_module("final_suite_partial_protocol")
    seed_dir, current_config, _ = _partial_checkpoint(
        module,
        tmp_path,
        info_updates={"protocol_hash": "old-protocol", "protocol_sha256": "old-protocol"},
    )
    with pytest.raises(RuntimeError, match="protocol hash mismatch"):
        module.validate_partial_training_resume(
            seed_dir, 42, current_config, "current-commit", "current-protocol"
        )


def test_partial_checkpoint_rejects_wrong_seed(tmp_path):
    module = _load_final_suite_module("final_suite_partial_seed")
    seed_dir, current_config, _ = _partial_checkpoint(module, tmp_path, info_updates={"seed": 11})
    with pytest.raises(RuntimeError, match="seed mismatch"):
        module.validate_partial_training_resume(
            seed_dir, 42, current_config, "current-commit", "current-protocol"
        )


def test_partial_checkpoint_rejects_incompatible_scientific_config(tmp_path):
    module = _load_final_suite_module("final_suite_partial_config")
    seed_dir, current_config, _ = _partial_checkpoint(
        module, tmp_path, config_updates={"training": {"learning_rate": 0.123}}
    )
    with pytest.raises(RuntimeError, match=r"scientific config mismatch: training\.learning_rate"):
        module.validate_partial_training_resume(
            seed_dir, 42, current_config, "current-commit", "current-protocol"
        )


def test_partial_checkpoint_rejects_missing_run_info(tmp_path):
    module = _load_final_suite_module("final_suite_partial_no_info")
    seed_dir, current_config, _ = _partial_checkpoint(module, tmp_path)
    (seed_dir / "run_info.json").unlink()
    with pytest.raises(RuntimeError, match="run_info.json is missing"):
        module.validate_partial_training_resume(
            seed_dir, 42, current_config, "current-commit", "current-protocol"
        )


def test_partial_checkpoint_rejects_missing_resolved_config(tmp_path):
    module = _load_final_suite_module("final_suite_partial_no_config")
    seed_dir, current_config, _ = _partial_checkpoint(module, tmp_path)
    (seed_dir / "config_resolved.yaml").unlink()
    with pytest.raises(RuntimeError, match="config_resolved.yaml is missing"):
        module.validate_partial_training_resume(
            seed_dir, 42, current_config, "current-commit", "current-protocol"
        )


def test_completed_training_skips_before_partial_checkpoint_validation(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str((tmp_path / "output").resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    module = _load_final_suite_module("final_suite_completed_before_partial")
    suite = module.FinalSuite(argparse.Namespace(force=False, device="cpu", allow_final_test=False))
    seed = 11
    directory = suite.output / "checkpoints" / module.BACKBONE / f"seed_{seed:04d}"
    outputs = [
        suite.checkpoint(seed),
        directory / "metrics.csv",
        suite.id_logits(seed),
        suite.output / "predictions" / module.BACKBONE / f"seed_{seed:04d}" / "ood_val.parquet",
    ]
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("complete", encoding="utf-8")
    (directory / "last.ckpt").write_bytes(b"stale-partial-irrelevant-to-completed-stage")
    commit, _ = module.git_state(module.REPO)
    monkeypatch.setattr(module, "git_state", lambda repository=module.REPO: (commit, False))
    protocol_hash = (module.REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
    command = [module.PYTHON, "scripts/05_train_erm.py", "--config", suite.model_config, "--seed", str(seed), "--device", "cpu"]
    marker = suite.stage_root / f"train_{module.BACKBONE}_{seed:04d}.json"
    marker.write_text(
        json.dumps(
            {
                "status": "completed",
                "stage_signature": suite.training_stage_signature(command),
                "git_commit": commit,
                "protocol_hash": protocol_hash,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        module,
        "validate_partial_training_resume",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("partial validator called")),
    )
    suite.train(seed, "cpu")

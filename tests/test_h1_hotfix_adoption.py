import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml


def _load_script(name="h1_hotfix_adoption"):
    script = Path(__file__).resolve().parents[1] / "scripts/16_adopt_h1_hotfix_upstream.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_final_suite(name="final_suite_adoption"):
    script = Path(__file__).resolve().parents[1] / "scripts/run_final_suite.py"
    spec = importlib.util.spec_from_file_location(name, script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def adoption_case(tmp_path, monkeypatch):
    module = _load_script()
    source_repo = Path(__file__).resolve().parents[1]
    repository = tmp_path / "repository"
    output_root = tmp_path / "output"
    for relative in (
        "configs/final/FINAL_PROTOCOL.yaml",
        "configs/final/protocol_sha256.txt",
        "configs/models/resnet50.yaml",
        "configs/models/resnet50_hf.yaml",
    ):
        destination = repository / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_repo / relative, destination)
    protocol_dir = output_root / "protocol"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "protocol_sha256.txt").write_text(
        module.PROTOCOL_HASH + "\n", encoding="utf-8"
    )
    monkeypatch.setenv("MEDSTYLE_DATA_ROOT", str((tmp_path / "data").resolve()))
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str(output_root.resolve()))
    monkeypatch.setenv("MEDSTYLE_HF_REPO", "PeiyuanHao/MedStyleAudit-Experiments")
    current_config = module.load_config(repository / "configs/models/resnet50_hf.yaml")
    stage_root = output_root / ".stage_state"
    stage_root.mkdir(parents=True)

    def write_output(relative):
        path = output_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"artifact:{relative}".encode())
        return str(path.resolve())

    markers = {}
    for stage in module.ADOPTED_STAGES:
        outputs = [write_output(relative) for relative in sorted(module._expected_outputs(stage))]
        if stage in module.TRAINING_SEEDS:
            seed = module.TRAINING_SEEDS[stage]
            command = [
                sys.executable,
                "scripts/05_train_erm.py",
                "--config",
                "configs/models/resnet50_hf.yaml",
                "--seed",
                str(seed),
                "--device",
                "cuda:0",
            ]
            signature = [
                "scripts/05_train_erm.py",
                "--config",
                "configs/models/resnet50_hf.yaml",
                "--seed",
                str(seed),
            ]
            seed_dir = output_root / f"checkpoints/resnet50/seed_{seed:04d}"
            run_info = {
                "status": "completed",
                "experiment": "train_erm_resnet50",
                "git_commit": module.SOURCE_COMMIT,
                "protocol_hash": module.PROTOCOL_HASH,
                "protocol_sha256": module.PROTOCOL_HASH,
                "seed": seed,
                "dataset_version": current_config["data"]["version"],
            }
            (seed_dir / "run_info.json").write_text(json.dumps(run_info), encoding="utf-8")
            (seed_dir / "config_resolved.yaml").write_text(
                yaml.safe_dump(current_config), encoding="utf-8"
            )
        else:
            command = ["python", f"scripts/{stage}.py"]
            signature = command
        marker = {
            "stage": stage,
            "status": "completed",
            "command": command,
            "execution_command": command,
            "stage_signature": signature,
            "outputs": outputs,
            "git_commit": module.SOURCE_COMMIT,
            "protocol_hash": module.PROTOCOL_HASH,
            "completed_at": "2026-08-29T01:02:03+00:00",
        }
        path = stage_root / f"{stage}.json"
        path.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
        markers[stage] = marker

    failed_audit = stage_root / "audit_hospital1_primary_0011.json"
    failed_audit.write_text(
        json.dumps(
            {
                "stage": "audit_hospital1_primary_0011",
                "status": "failed",
                "git_commit": module.SOURCE_COMMIT,
                "protocol_hash": module.PROTOCOL_HASH,
            }
        ),
        encoding="utf-8",
    )
    target_commit = "7ffdcd2902bbf3143cd600991db803a601e804a0"
    diff_paths = sorted(module.ALLOWED_DIFF_PATHS)
    monkeypatch.setattr(module, "inspect_repository", lambda repo: (target_commit, diff_paths))
    return module, repository, output_root, markers, failed_audit, target_commit, diff_paths


def test_approved_audit_only_hotfix_validates_without_writing(adoption_case):
    module, repository, output_root, _, _, target_commit, _ = adoption_case
    ledger = module.run_adoption(repository, output_root)
    assert ledger["status"] == "validated"
    assert ledger["target_git_commit"] == target_commit
    assert ledger["adopted_stages"] == list(module.ADOPTED_STAGES)
    assert not (output_root / module.LEDGER_RELATIVE_PATH).exists()
    for stage in module.ADOPTED_STAGES:
        marker = json.loads(
            (output_root / ".stage_state" / f"{stage}.json").read_text(encoding="utf-8")
        )
        assert marker["git_commit"] == module.SOURCE_COMMIT


@pytest.mark.parametrize(
    "path",
    [
        "src/medstyleaudit/models/trainer.py",
        "configs/models/resnet50_hf.yaml",
        "src/medstyleaudit/matching/balanced.py",
        "configs/final/FINAL_PROTOCOL.yaml",
    ],
)
def test_unexpected_scientific_change_is_rejected(path):
    module = _load_script(f"h1_hotfix_diff_{path.replace('/', '_')}")
    with pytest.raises(module.AdoptionError, match="Unapproved source-to-target"):
        module.validate_diff_paths([path])


def test_required_approved_hotfix_change_must_be_present():
    module = _load_script("h1_hotfix_required_diff")
    with pytest.raises(module.AdoptionError, match="Required approved audit hotfix"):
        module.validate_diff_paths(["src/medstyleaudit/audit/inference.py"])


def test_wrong_protocol_hash_is_rejected(adoption_case):
    module, repository, output_root, *_ = adoption_case
    (output_root / "protocol/protocol_sha256.txt").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(module.AdoptionError, match="Frozen protocol hash"):
        module.run_adoption(repository, output_root)


def test_missing_recorded_output_is_rejected(adoption_case):
    module, repository, output_root, markers, *_ = adoption_case
    Path(markers["matching"]["outputs"][0]).unlink()
    with pytest.raises(module.AdoptionError, match="missing or not a regular file"):
        module.run_adoption(repository, output_root)


def test_incomplete_source_stage_is_rejected(adoption_case):
    module, repository, output_root, *_ = adoption_case
    marker_path = output_root / ".stage_state/descriptors.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["status"] = "failed"
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(module.AdoptionError, match="Source stage is not completed"):
        module.run_adoption(repository, output_root)


def test_training_provenance_mismatch_is_rejected(adoption_case):
    module, repository, output_root, *_ = adoption_case
    run_info_path = output_root / "checkpoints/resnet50/seed_0011/run_info.json"
    run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    run_info["git_commit"] = "wrong"
    run_info_path.write_text(json.dumps(run_info), encoding="utf-8")
    with pytest.raises(module.AdoptionError, match="run_info Git provenance mismatch"):
        module.run_adoption(repository, output_root)


def test_apply_preserves_origin_is_idempotent_and_excludes_failed_audit(adoption_case):
    module, repository, output_root, originals, failed_audit, target_commit, _ = adoption_case
    failed_audit_before = failed_audit.read_bytes()
    ledger = module.run_adoption(repository, output_root, apply=True)
    ledger_path = output_root / module.LEDGER_RELATIVE_PATH
    ledger_before = ledger_path.read_bytes()
    for stage in module.ADOPTED_STAGES:
        marker_path = output_root / ".stage_state" / f"{stage}.json"
        adopted = json.loads(marker_path.read_text(encoding="utf-8"))
        archived = json.loads(
            (output_root / module.ARCHIVE_RELATIVE_PATH / f"{stage}.json").read_text(
                encoding="utf-8"
            )
        )
        assert adopted["git_commit"] == target_commit
        assert adopted["artifact_origin_git_commit"] == module.SOURCE_COMMIT
        assert adopted["adopted_by_hotfix"] is True
        assert adopted["original_completed_at"] == originals[stage]["completed_at"]
        assert archived == originals[stage]
    assert failed_audit.read_bytes() == failed_audit_before
    assert "audit_hospital1_primary_0011" not in ledger["adopted_stages"]
    repeated = module.run_adoption(repository, output_root, apply=True)
    assert repeated == ledger
    assert ledger_path.read_bytes() == ledger_before


def test_final_suite_accepts_explicitly_adopted_training_marker(adoption_case, monkeypatch):
    module, repository, output_root, _, _, target_commit, _ = adoption_case
    module.run_adoption(repository, output_root, apply=True)
    final_suite = _load_final_suite()
    monkeypatch.setattr(
        final_suite, "git_state", lambda repository=final_suite.REPO: (target_commit, False)
    )
    suite = final_suite.FinalSuite(
        argparse.Namespace(force=False, device="cpu", devices=None, allow_final_test=False)
    )
    seed = 11
    directory = output_root / f"checkpoints/resnet50/seed_{seed:04d}"
    outputs = [
        suite.checkpoint(seed),
        directory / "metrics.csv",
        suite.id_logits(seed),
        output_root / f"predictions/resnet50/seed_{seed:04d}/ood_val.parquet",
    ]
    command = [
        final_suite.PYTHON,
        "scripts/05_train_erm.py",
        "--config",
        suite.model_config,
        "--seed",
        str(seed),
        "--device",
        "cpu",
    ]
    assert suite.completed(
        "train_resnet50_0011",
        outputs,
        command=command,
        stage_signature=suite.training_stage_signature(command),
    )

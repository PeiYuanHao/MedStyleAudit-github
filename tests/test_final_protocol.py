import json
from copy import deepcopy

import pytest

from medstyleaudit.protocol import assert_frozen_execution_config, authorize_final_test, protocol_sha256, verify_protocol_lock, write_protocol_lock
from medstyleaudit.utils.config import load_config


def test_protocol_hash_is_stable_for_mapping_order(tmp_path):
    assert protocol_sha256({"b": 2, "a": 1}) == protocol_sha256({"a": 1, "b": 2})
    protocol = tmp_path / "FINAL_PROTOCOL.yaml"; protocol.write_text("a: 1\nb: 2\n", encoding="utf-8")
    lock = tmp_path / "protocol_sha256.txt"; write_protocol_lock(protocol, lock)
    assert verify_protocol_lock(protocol, lock) == protocol_sha256(protocol)
    protocol.write_text("a: 2\nb: 2\n", encoding="utf-8")
    with pytest.raises(PermissionError, match="hash mismatch"):
        verify_protocol_lock(protocol, lock)


def test_final_test_authorization_requires_pass_and_matching_clean_commit(tmp_path, monkeypatch):
    protocol = tmp_path / "FINAL_PROTOCOL.yaml"; protocol.write_text("status: frozen\n", encoding="utf-8")
    root = tmp_path / "artifacts"; (root / "protocol").mkdir(parents=True)
    digest = write_protocol_lock(protocol, root / "protocol/protocol_sha256.txt")
    (root / "protocol/final_preflight.json").write_text(json.dumps({"status": "PASS", "protocol_hash": digest, "git_commit": "abc"}), encoding="utf-8")
    monkeypatch.setattr("medstyleaudit.protocol.git_state", lambda repository=".": ("abc", False))
    with pytest.raises(PermissionError, match="explicit unlock"):
        authorize_final_test(root, protocol)
    (root / "protocol/final_test_lock.json").write_text(json.dumps({"status": "unlocked", "protocol_sha256": digest, "git_commit": "abc"}), encoding="utf-8")
    assert authorize_final_test(root, protocol)["protocol_hash"] == digest
    monkeypatch.setattr("medstyleaudit.protocol.git_state", lambda repository=".": ("abc", True))
    with pytest.raises(PermissionError, match="Git must be clean"):
        authorize_final_test(root, protocol)


def test_frozen_execution_config_checks_all_result_defining_fields():
    matching = load_config("configs/matching/primary.yaml")["matching"]
    model = load_config("configs/models/resnet50_hf.yaml")
    audit = load_config("configs/audit/primary.yaml")["audit"]
    assert_frozen_execution_config(matching=matching, model=model, audit=audit, seed=11)

    changed_matching = deepcopy(matching); changed_matching["donor_slide_reuse_cap"] += 1
    with pytest.raises(PermissionError, match="donor_slide_reuse_cap"):
        assert_frozen_execution_config(matching=changed_matching)
    changed_audit = deepcopy(audit); changed_audit["bootstrap_draws"] += 1
    with pytest.raises(PermissionError, match="bootstrap_draws"):
        assert_frozen_execution_config(audit=changed_audit)
    changed_model = deepcopy(model); changed_model["training"]["selection_metric"] = "val_accuracy"
    with pytest.raises(PermissionError, match="selection_metric"):
        assert_frozen_execution_config(model=changed_model, seed=11)

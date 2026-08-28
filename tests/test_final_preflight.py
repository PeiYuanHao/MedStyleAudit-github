import json

import pandas as pd

from medstyleaudit.final_preflight import run_final_preflight
from medstyleaudit.protocol import write_protocol_lock


def test_final_preflight_passes_only_complete_threshold_compliant_inputs(tmp_path, monkeypatch):
    protocol = tmp_path / "FINAL_PROTOCOL.yaml"; protocol.write_text("status: frozen\n", encoding="utf-8")
    lock = tmp_path / "protocol_sha256.txt"; write_protocol_lock(protocol, lock)
    pytest_marker = tmp_path / "pytest.json"; pytest_marker.write_text(json.dumps({"status": "PASS", "roi_tests": "PASS"}), encoding="utf-8")
    dataset = tmp_path / "dataset"; dataset.mkdir()
    integrity = tmp_path / "integrity.json"; integrity.write_text(json.dumps({"overall_status": "PASS"}), encoding="utf-8")
    alignment = tmp_path / "alignment.json"; alignment.write_text(json.dumps({"status": "unavailable"}), encoding="utf-8")
    descriptors = tmp_path / "descriptors.csv"; pd.DataFrame({"source_id": [1], "f": [1.0]}).to_csv(descriptors, index=False)
    directed = tmp_path / "directed.csv"; pd.DataFrame({"source_split": ["val"], "source_hospital": [1], "target_hospital": [0], "coverage": [.95]}).to_csv(directed, index=False)
    common = tmp_path / "common.csv"; pd.DataFrame({"coverage": [.95]}).to_csv(common, index=False)
    balance = tmp_path / "balance.csv"; pd.DataFrame({"paired_smd": [.05], "n_triplets": [10]}).to_csv(balance, index=False)
    feature = tmp_path / "feature.csv"; pd.DataFrame({"feature": ["f"], "paired_smd": [.05]}).to_csv(feature, index=False)
    donor = tmp_path / "donor.csv"; pd.DataFrame({"row_type": ["donor"], "total_reuse": [20]}).to_csv(donor, index=False)
    slide = tmp_path / "slide.csv"; pd.DataFrame({"row_type": ["slide"], "total_donor_uses": [1]}).to_csv(slide, index=False)
    ledger = tmp_path / "ledger.csv"; pd.DataFrame({"source_id": [1]}).to_csv(ledger, index=False)
    monkeypatch.setattr("medstyleaudit.final_preflight.git_state", lambda repository=".": ("abc", False))
    files = {"integrity_report": integrity, "alignment_report": alignment, "descriptors": descriptors, "source_ledger": ledger, "directed_coverage": directed, "common_support_coverage": common, "matching_balance": balance, "feature_balance": feature, "donor_reuse": donor, "slide_reuse": slide}
    spec = {"files": files, "required_matching_files": [key for key in files if key != "alignment_report"], "pytest_marker": pytest_marker, "required_dataset_files": [dataset], "descriptor_features": ["f"], "expected_hospital_pairs": [["val", 1, 0]], "donor_reuse_cap": 20, "thresholds": {"directed_coverage": .9, "common_support_coverage": .9, "aggregate_abs_paired_smd": .1, "per_feature_abs_smd": .1}, "protocol_path": protocol, "protocol_hash_path": lock, "repository": tmp_path}
    report = run_final_preflight(spec, tmp_path / "final_preflight.json")
    assert report["status"] == "PASS"
    assert report["lesion_aware_status"] == "unavailable"

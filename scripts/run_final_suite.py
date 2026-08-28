"""Resumable, fail-closed orchestration of the frozen MedStyleAudit suite."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from medstyleaudit.protocol import git_state
from medstyleaudit.utils.io import save_json


REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
PHASES = list(range(9))


class FinalSuite:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data = Path(os.environ["MEDSTYLE_DATA_ROOT"]).resolve()
        self.output = Path(os.environ["MEDSTYLE_OUTPUT_ROOT"]).resolve()
        self.repo_id = os.environ["MEDSTYLE_HF_REPO"]
        self.state_path = self.output / "final_run_state.json"
        self.stage_root = self.output / ".stage_state"
        self.log_root = self.output / "logs" / "final_suite"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else {"stages": {}, "p0": "pending", "resnet50_training": "pending", "densenet121_training": "pending", "controls": "pending", "robustness": "pending", "final_test": "locked"}
        self.output.mkdir(parents=True, exist_ok=True); self.stage_root.mkdir(parents=True, exist_ok=True); self.log_root.mkdir(parents=True, exist_ok=True)

    def save_state(self) -> None:
        self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
        save_json(self.state, self.state_path)

    def completed(self, name: str, outputs: list[Path]) -> bool:
        marker = self.stage_root / f"{name}.json"
        if self.args.force or not marker.is_file() or not all(path.exists() for path in outputs):
            return False
        return json.loads(marker.read_text(encoding="utf-8")).get("status") == "completed"

    def run(self, name: str, command: list[str], outputs: list[Path], summary: str | None = None) -> None:
        if self.completed(name, outputs):
            print(f"[resume] {name}")
            return
        self.state["stages"][name] = "running"
        if summary: self.state[summary] = "running"
        self.save_state()
        log_path = self.log_root / f"{name}.log"
        print(f"[run] {name}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
            process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        if process.returncode or not all(path.exists() for path in outputs):
            self.state["stages"][name] = "failed"
            if summary: self.state[summary] = "failed"
            self.save_state()
            raise RuntimeError(f"Stage {name} failed; see {log_path}")
        marker = {"stage": name, "status": "completed", "command": command, "outputs": [str(path) for path in outputs], "completed_at": datetime.now(timezone.utc).isoformat()}
        save_json(marker, self.stage_root / f"{name}.json")
        self.state["stages"][name] = "completed"
        if summary: self.state[summary] = "completed"
        self.save_state()

    def pytest(self) -> None:
        marker = self.output / "protocol" / "pytest_passed.json"
        if self.completed("pytest", [marker]): return
        command = [PYTHON, "-m", "pytest", "-q"]
        self.run("pytest_command", command, [], "p0")
        save_json({"status": "PASS", "roi_tests": "PASS", "command": command, "timestamp": datetime.now(timezone.utc).isoformat()}, marker)
        save_json({"stage": "pytest", "status": "completed", "outputs": [str(marker)]}, self.stage_root / "pytest.json")

    def train(self, backbone: str, config: str, seed: int) -> None:
        directory = self.output / "checkpoints" / backbone / f"seed_{seed:04d}"
        command = [PYTHON, "scripts/05_train_erm.py", "--config", config, "--seed", str(seed), "--device", self.args.device]
        if directory.exists(): command.append("--overwrite")
        if (directory / "last.ckpt").is_file():
            command += ["--resume", str(directory / "last.ckpt")]
        self.run(f"train_{backbone}_{seed:04d}", command, [directory / "best.ckpt", directory / "metrics.csv", self.output / "predictions" / backbone / f"seed_{seed:04d}" / "id_val.parquet", self.output / "predictions" / backbone / f"seed_{seed:04d}" / "ood_val.parquet"], f"{backbone}_training")

    def audit(self, backbone: str, config: str, seed: int, split: str, triplets: Path, roi_only: bool = False) -> None:
        base = self.output / "audits" / ("roi_only" if roi_only else "primary") / ("val" if split == "val" else "test") / backbone / f"seed_{seed:04d}"
        # The audit script's native ordering is backbone/seed/split; pass an explicit final-layout directory.
        command = [PYTHON, "scripts/06_run_primary_audit.py", "--config", "configs/audit/primary.yaml", "--model-config", config, "--checkpoint", str(self.output / "checkpoints" / backbone / f"seed_{seed:04d}" / "best.ckpt"), "--id-logits", str(self.output / "predictions" / backbone / f"seed_{seed:04d}" / "id_val.parquet"), "--triplets", str(triplets), "--split", split, "--seed", str(seed), "--device", self.args.device, "--output-dir", str(base)]
        if roi_only: command.append("--roi-only-control")
        if split == "test": command.append("--allow-final-test")
        command.append("--overwrite")
        self.run(f"audit_{'roi_' if roi_only else ''}{backbone}_{seed:04d}_{split}", command, [base / "directed_hcs.csv", base / "directed_hce.csv", base / "audit_records.parquet"])

    def execute(self) -> None:
        if self.repo_id != "PeiyuanHao/MedStyleAudit-Experiments":
            raise EnvironmentError("MEDSTYLE_HF_REPO must be exactly PeiyuanHao/MedStyleAudit-Experiments")
        commit, dirty = git_state(REPO)
        if dirty: raise RuntimeError("Final suite requires a clean Git working tree")
        protocol_dir = self.output / "protocol"
        self.run("environment", [PYTHON, "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"], [], "p0")
        self.run("protocol_lock", [PYTHON, "scripts/lock_final_protocol.py"], [protocol_dir / "FINAL_PROTOCOL.yaml", protocol_dir / "protocol_sha256.txt", protocol_dir / "git_commit.txt"], "p0")
        self.pytest()
        self.run("data_integrity", [PYTHON, "scripts/00_data_integrity.py", "--config", "configs/data/camelyon17_hf.yaml", "--overwrite"], [self.output / "p0/data_integrity/integrity_report.json", self.output / "p0/data_integrity/wilds_summary.csv"], "p0")
        self.run("lesion_alignment", [PYTHON, "scripts/01_validate_lesion_mapping.py", "--config", "configs/data/camelyon17_hf.yaml", "--overwrite"], [self.output / "p0/data_integrity/alignment_report.json"], "p0")
        self.run("descriptors", [PYTHON, "scripts/02_build_descriptors.py", "--config", "configs/data/camelyon17_hf.yaml", "--overwrite"], [self.output / "p0/descriptors/descriptors.parquet", self.output / "p0/descriptors/mask_stability.parquet"], "p0")
        self.run("matching", [PYTHON, "scripts/03_run_matching.py", "--config", "configs/matching/primary.yaml", "--overwrite"], [self.output / "p0/matching/triplets.parquet", self.output / "p0/matching/feature_balance.csv", self.output / "p0/matching/donor_reuse.csv", self.output / "p0/matching/slide_reuse.csv"], "p0")
        self.run("coverage_gate", [PYTHON, "scripts/04_check_matching_coverage.py", "--overwrite"], [self.output / "p0/matching/coverage_review/coverage_gate.json"], "p0")
        self.run("final_preflight", [PYTHON, "scripts/14_final_preflight.py"], [protocol_dir / "final_preflight.json"], "p0")

        configs = {"resnet50": "configs/models/resnet50_hf.yaml", "densenet121": "configs/models/densenet121_hf.yaml"}
        seeds = [11, 23, 42, 57, 71, 89, 101, 131, 173, 211]
        triplets = self.output / "p0/matching/triplets.parquet"
        self.train("resnet50", configs["resnet50"], 11)
        self.audit("resnet50", configs["resnet50"], 11, "val", triplets)
        for backbone, config in configs.items():
            for seed in seeds: self.train(backbone, config, seed)
        for backbone, config in configs.items():
            for seed in seeds: self.audit(backbone, config, seed, "val", triplets)

        for backbone, config in configs.items():
            for seed in seeds:
                self.audit(backbone, config, seed, "val", triplets, roi_only=True)
                for control in ("context_randomized", "planted_shortcut"):
                    control_config = f"configs/controls/{control}.yaml"
                    root = self.output / "audits" / control / backbone / f"seed_{seed:04d}"
                    expected = root / ("control_status.csv" if control == "context_randomized" else "rho_status.csv")
                    self.run(f"control_{control}_{backbone}_{seed:04d}", [PYTHON, "scripts/07_run_controls.py", "--config", control_config, "--model-config", config, "--seed", str(seed), "--device", self.args.device, "--overwrite"], [expected], "controls")

        for backbone, config in configs.items():
            for seed in seeds:
                checkpoint = self.output / "checkpoints" / backbone / f"seed_{seed:04d}" / "best.ckpt"
                logits = self.output / "predictions" / backbone / f"seed_{seed:04d}" / "id_val.parquet"
                for kind in ("source_buffer", "seam"):
                    root = self.output / "audits" / kind / backbone / f"seed_{seed:04d}"
                    self.run(f"robustness_{kind}_{backbone}_{seed:04d}", [PYTHON, "scripts/08_run_robustness.py", "--config", f"configs/audit/{kind}.yaml", "--checkpoint", str(checkpoint), "--model-config", config, "--id-logits", str(logits), "--triplets", str(triplets), "--seed", str(seed), "--device", self.args.device, "--output-dir", str(root), "--overwrite"], [root / "robustness_summary.csv"], "robustness")
        for level in range(1, 5):
            for target in (0, 3, 4):
                root = self.output / "audits/identification_ladder" / f"level_{level}" / f"target_{target}"
                self.run(f"ladder_{level}_{target}", [PYTHON, "scripts/09_run_identification_ladder.py", "--level", str(level), "--target-hospital", str(target), "--split", "val", "--output-dir", str(root), "--overwrite"], [root / "triplets.parquet", root / "coverage.csv"], "robustness")
            combined = self.output / "audits/identification_ladder" / f"level_{level}" / "triplets.parquet"
            combine_code = "import pandas as pd,sys; from pathlib import Path; root=Path(sys.argv[1]); pd.concat([pd.read_parquet(root/f'target_{h}/triplets.parquet') for h in (0,3,4)],ignore_index=True).to_parquet(root/'triplets.parquet',index=False)"
            self.run(f"ladder_{level}_combine", [PYTHON, "-c", combine_code, str(combined.parent)], [combined], "robustness")
            for backbone, config in configs.items():
                for seed in seeds:
                    base = self.output / "audits/identification_ladder" / f"level_{level}" / "val" / backbone / f"seed_{seed:04d}"
                    command = [PYTHON, "scripts/06_run_primary_audit.py", "--model-config", config, "--checkpoint", str(self.output / "checkpoints" / backbone / f"seed_{seed:04d}/best.ckpt"), "--id-logits", str(self.output / "predictions" / backbone / f"seed_{seed:04d}/id_val.parquet"), "--triplets", str(combined), "--split", "val", "--seed", str(seed), "--device", self.args.device, "--output-dir", str(base), "--overwrite"]
                    self.run(f"ladder_audit_{level}_{backbone}_{seed:04d}", command, [base / "directed_hcs.csv", base / "directed_hce.csv"], "robustness")
        lesion = json.loads((self.output / "p0/data_integrity/alignment_report.json").read_text(encoding="utf-8"))
        lesion_available = str(lesion.get("status", "")).lower() in {"pass", "passed", "validated", "available"}
        lesion_root = self.output / "audits/lesion_aware"
        if lesion_available:
            lesion_matching = lesion_root / "matching"
            self.run("lesion_matching", [PYTHON, "scripts/03_run_matching.py", "--config", "configs/matching/lesion_aware.yaml", "--split", "val", "--output-dir", str(lesion_matching), "--overwrite"], [lesion_matching / "triplets.parquet"])
            for backbone, config in configs.items():
                for seed in seeds:
                    base = lesion_root / "val" / backbone / f"seed_{seed:04d}"
                    command = [PYTHON, "scripts/06_run_primary_audit.py", "--model-config", config, "--checkpoint", str(self.output / "checkpoints" / backbone / f"seed_{seed:04d}/best.ckpt"), "--id-logits", str(self.output / "predictions" / backbone / f"seed_{seed:04d}/id_val.parquet"), "--triplets", str(lesion_matching / "triplets.parquet"), "--split", "val", "--seed", str(seed), "--device", self.args.device, "--output-dir", str(base), "--overwrite"]
                    self.run(f"lesion_audit_{backbone}_{seed:04d}", command, [base / "directed_hcs.csv", base / "directed_hce.csv"], "robustness")
            save_json({"status": "completed"}, lesion_root / "status.json")
        else:
            save_json({"status": "unavailable", "reason": "annotation alignment unavailable"}, lesion_root / "status.json")
        self.run("aggregate_validation", [PYTHON, "scripts/12_aggregate_experiments.py", "--split", "val", "--audit-root", str(self.output / "audits/primary"), "--overwrite"], [self.output / "aggregate/primary/global_hcs.csv", self.output / "aggregate/final/MAIN_RESULTS.csv"])

        if not self.args.allow_final_test:
            self.state["final_test"] = "locked"; self.save_state()
            print("Validation suite complete. Hospital 2 remains locked; resume with --allow-final-test.")
            return
        self.state["final_test"] = "running"; self.save_state()
        final_matching = self.output / "p0/matching_final_test"
        self.run("final_test_matching", [PYTHON, "scripts/03_run_matching.py", "--split", "test", "--allow-final-test", "--prior-triplets", str(self.output / "p0/matching/triplets.parquet"), "--output-dir", str(final_matching), "--overwrite"], [final_matching / "triplets.parquet"])
        self.run("merge_final_test_matching", [PYTHON, "scripts/merge_final_test_matching.py", "--primary", str(self.output / "p0/matching"), "--final-test", str(final_matching)], [self.output / "p0/matching/triplets.parquet", self.output / "p0/matching/feature_balance.csv", self.output / "p0/matching/final_matching_check.json"])
        for backbone, config in configs.items():
            for seed in seeds:
                prediction = self.output / "predictions" / backbone / f"seed_{seed:04d}" / "ood_test.parquet"
                self.run(f"predict_{backbone}_{seed:04d}_test", [PYTHON, "scripts/05_train_erm.py", "--config", config, "--seed", str(seed), "--device", self.args.device, "--predictions-only", "--allow-final-test"], [prediction])
                self.audit(backbone, config, seed, "test", final_matching / "triplets.parquet")
        self.run("aggregate_final_test", [PYTHON, "scripts/12_aggregate_experiments.py", "--split", "test", "--audit-root", str(self.output / "audits/primary"), "--overwrite"], [self.output / "aggregate/primary/global_hcs.csv", self.output / "aggregate/final/MAIN_RESULTS.csv"])
        shutil.copy2(REPO / "docs/HF_DATASET_CARD.md", self.output / "README.md")
        save_json({"status": "ready_for_upload", "protocol_hash": (protocol_dir / "protocol_sha256.txt").read_text().strip(), "git_commit": commit, "artifact_manifest": "MANIFEST.json", "run_state": "logs/final_suite/final_run_state.json"}, self.output / "aggregate/final/experiment_manifest.json")
        shutil.copy2(self.state_path, self.log_root / "final_run_state.json")
        self.run("hf_create", [PYTHON, "scripts/hf_create_repository.py"], [], None)
        self.run("hf_upload", [PYTHON, "scripts/hf_upload_artifacts.py", "--root", str(self.output)], [self.output / "MANIFEST.json"])
        failed_marker = self.output / ".hf_verification_failed"
        try:
            self.run("hf_verify", [PYTHON, "scripts/hf_verify_artifacts.py"], [], None)
            failed_marker.unlink(missing_ok=True)
            (self.output / ".hf_verified").write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
        except Exception:
            failed_marker.write_text(datetime.now(timezone.utc).isoformat(), encoding="utf-8")
            raise
        self.state["final_test"] = "completed"; self.save_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    required = ["MEDSTYLE_DATA_ROOT", "MEDSTYLE_OUTPUT_ROOT", "MEDSTYLE_HF_REPO"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing: parser.error(f"Required environment variables are missing: {missing}")
    FinalSuite(args).execute()


if __name__ == "__main__":
    main()

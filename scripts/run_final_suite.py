"""Resumable, fail-closed orchestration of the frozen MedStyleAudit suite."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from medstyleaudit.protocol import git_state
from medstyleaudit.utils.config import load_config
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
        self.suite_config = load_config(REPO / "configs/final/FINAL_SUITE.yaml")
        self.configs = dict(self.suite_config["backbones"])
        self.all_seeds = [int(seed) for seed in self.suite_config["seeds"]]
        requested_backbones = getattr(args, "backbones", None)
        requested_seeds = getattr(args, "seeds", None)
        self.backbones = requested_backbones or list(self.configs)
        self.seeds = requested_seeds or self.all_seeds
        unknown_backbones = sorted(set(self.backbones) - set(self.configs))
        unknown_seeds = sorted(set(self.seeds) - set(self.all_seeds))
        if unknown_backbones or unknown_seeds:
            raise ValueError(f"Unknown final-suite selection: backbones={unknown_backbones}, seeds={unknown_seeds}")
        configured_devices = getattr(args, "devices", None)
        self.devices = configured_devices or [args.device]
        if not self.devices:
            raise ValueError("At least one device is required")
        self.partial_selection = set(self.backbones) != set(self.configs) or set(self.seeds) != set(self.all_seeds)
        self.state_path = self.output / "final_run_state.json"
        self.stage_root = self.output / ".stage_state"
        self.log_root = self.output / "logs" / "final_suite"
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else {"stages": {}, "p0": "pending", "resnet50_training": "pending", "densenet121_training": "pending", "controls": "pending", "robustness": "pending", "final_test": "locked"}
        self.state_lock = threading.RLock()
        self.output.mkdir(parents=True, exist_ok=True); self.stage_root.mkdir(parents=True, exist_ok=True); self.log_root.mkdir(parents=True, exist_ok=True)

    def save_state(self) -> None:
        with self.state_lock:
            self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_json(self.state, self.state_path)

    def completed(self, name: str, outputs: list[Path], validity: Callable[[], bool] | None = None) -> bool:
        marker = self.stage_root / f"{name}.json"
        if self.args.force or not marker.is_file() or not all(path.exists() for path in outputs):
            return False
        if json.loads(marker.read_text(encoding="utf-8")).get("status") != "completed":
            return False
        return validity is None or bool(validity())

    def run(self, name: str, command: list[str], outputs: list[Path], summary: str | None = None, validity: Callable[[], bool] | None = None) -> None:
        if self.completed(name, outputs, validity):
            print(f"[resume] {name}")
            return
        with self.state_lock:
            self.state["stages"][name] = "running"
            if summary: self.state[summary] = "running"
            self.save_state()
        log_path = self.log_root / f"{name}.log"
        print(f"[run] {name}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
            process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True)
        if process.returncode or not all(path.exists() for path in outputs):
            with self.state_lock:
                self.state["stages"][name] = "failed"
                if summary: self.state[summary] = "failed"
                self.save_state()
            raise RuntimeError(f"Stage {name} failed; see {log_path}")
        marker = {"stage": name, "status": "completed", "command": command, "outputs": [str(path) for path in outputs], "completed_at": datetime.now(timezone.utc).isoformat()}
        save_json(marker, self.stage_root / f"{name}.json")
        with self.state_lock:
            self.state["stages"][name] = "completed"
            if summary: self.state[summary] = "completed"
            self.save_state()

    def parallel(self, jobs: list[Callable[[str], None]]) -> None:
        """Run independent jobs with at most one subprocess assigned to each GPU."""
        if not jobs:
            return
        groups = [[] for _ in self.devices]
        for index, job in enumerate(jobs):
            groups[index % len(groups)].append(job)

        def worker(device: str, assigned: list[Callable[[str], None]]) -> None:
            for job in assigned:
                job(device)

        active = [(device, assigned) for device, assigned in zip(self.devices, groups) if assigned]
        if len(active) == 1:
            worker(*active[0])
            return
        with ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="final-suite-gpu") as executor:
            futures = [executor.submit(worker, device, assigned) for device, assigned in active]
            for future in futures:
                future.result()

    def pytest(self) -> None:
        marker = self.output / "protocol" / "pytest_passed.json"
        commit, _ = git_state(REPO)

        def current() -> bool:
            try:
                return json.loads(marker.read_text(encoding="utf-8")).get("git_commit") == commit
            except (OSError, json.JSONDecodeError):
                return False

        if self.completed("pytest", [marker], current): return
        command = [PYTHON, "-m", "pytest", "-q"]
        self.run("pytest_command", command, [], "p0", validity=lambda: False)
        save_json({"status": "PASS", "roi_tests": "PASS", "command": command, "git_commit": commit, "timestamp": datetime.now(timezone.utc).isoformat()}, marker)
        save_json({"stage": "pytest", "status": "completed", "outputs": [str(marker)]}, self.stage_root / "pytest.json")

    def train(self, backbone: str, config: str, seed: int, device: str) -> None:
        directory = self.output / "checkpoints" / backbone / f"seed_{seed:04d}"
        command = [PYTHON, "scripts/05_train_erm.py", "--config", config, "--seed", str(seed), "--device", device]
        if directory.exists(): command.append("--overwrite")
        if (directory / "last.ckpt").is_file():
            command += ["--resume", str(directory / "last.ckpt")]
        self.run(f"train_{backbone}_{seed:04d}", command, [directory / "best.ckpt", directory / "metrics.csv", self.output / "predictions" / backbone / f"seed_{seed:04d}" / "id_val.parquet", self.output / "predictions" / backbone / f"seed_{seed:04d}" / "ood_val.parquet"], f"{backbone}_training")

    def audit(self, backbone: str, config: str, seed: int, split: str, triplets: Path, device: str, roi_only: bool = False) -> None:
        base = self.output / "audits" / ("roi_only" if roi_only else "primary") / ("val" if split == "val" else "test") / backbone / f"seed_{seed:04d}"
        # The audit script's native ordering is backbone/seed/split; pass an explicit final-layout directory.
        command = [PYTHON, "scripts/06_run_primary_audit.py", "--config", "configs/audit/primary.yaml", "--model-config", config, "--checkpoint", str(self.output / "checkpoints" / backbone / f"seed_{seed:04d}" / "best.ckpt"), "--id-logits", str(self.output / "predictions" / backbone / f"seed_{seed:04d}" / "id_val.parquet"), "--triplets", str(triplets), "--split", split, "--seed", str(seed), "--device", device, "--output-dir", str(base)]
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
        tracked_protocol_hash = (REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()

        def current_protocol_lock() -> bool:
            try:
                return (
                    (protocol_dir / "git_commit.txt").read_text(encoding="utf-8").strip() == commit
                    and (protocol_dir / "protocol_sha256.txt").read_text(encoding="utf-8").strip() == tracked_protocol_hash
                )
            except OSError:
                return False

        self.run("protocol_lock", [PYTHON, "scripts/lock_final_protocol.py"], [protocol_dir / "FINAL_PROTOCOL.yaml", protocol_dir / "protocol_sha256.txt", protocol_dir / "git_commit.txt"], "p0", validity=current_protocol_lock)
        self.pytest()
        self.run("data_integrity", [PYTHON, "scripts/00_data_integrity.py", "--config", "configs/data/camelyon17_hf.yaml", "--overwrite"], [self.output / "p0/data_integrity/integrity_report.json", self.output / "p0/data_integrity/wilds_summary.csv"], "p0")
        self.run("lesion_alignment", [PYTHON, "scripts/01_validate_lesion_mapping.py", "--config", "configs/data/camelyon17_hf.yaml", "--overwrite"], [self.output / "p0/data_integrity/alignment_report.json"], "p0")
        self.run("descriptors", [PYTHON, "scripts/02_build_descriptors.py", "--config", "configs/data/camelyon17_hf.yaml", "--overwrite"], [self.output / "p0/descriptors/descriptors.parquet", self.output / "p0/descriptors/mask_stability.parquet"], "p0")
        self.run("matching", [PYTHON, "scripts/03_run_matching.py", "--config", "configs/matching/primary.yaml", "--overwrite"], [self.output / "p0/matching/triplets.parquet", self.output / "p0/matching/feature_balance.csv", self.output / "p0/matching/donor_reuse.csv", self.output / "p0/matching/slide_reuse.csv"], "p0")
        self.run("coverage_gate", [PYTHON, "scripts/04_check_matching_coverage.py", "--overwrite"], [self.output / "p0/matching/coverage_review/coverage_gate.json"], "p0")
        def current_preflight() -> bool:
            try:
                report = json.loads((protocol_dir / "final_preflight.json").read_text(encoding="utf-8"))
                return report.get("status") == "PASS" and report.get("git_commit") == commit and report.get("protocol_hash") == tracked_protocol_hash
            except (OSError, json.JSONDecodeError):
                return False

        self.run("final_preflight", [PYTHON, "scripts/14_final_preflight.py"], [protocol_dir / "final_preflight.json"], "p0", validity=current_preflight)

        triplets = self.output / "p0/matching/triplets.parquet"
        if "resnet50" in self.backbones and 11 in self.seeds:
            self.train("resnet50", self.configs["resnet50"], 11, self.devices[0])
            self.audit("resnet50", self.configs["resnet50"], 11, "val", triplets, self.devices[0])

        training_jobs = [
            (lambda device, backbone=backbone, config=self.configs[backbone], seed=seed: self.train(backbone, config, seed, device))
            for backbone in self.backbones for seed in self.seeds
        ]
        self.parallel(training_jobs)
        audit_jobs = [
            (lambda device, backbone=backbone, config=self.configs[backbone], seed=seed: self.audit(backbone, config, seed, "val", triplets, device))
            for backbone in self.backbones for seed in self.seeds
        ]
        self.parallel(audit_jobs)

        if self.partial_selection:
            print("Selected primary training/audit shard complete; full controls, aggregation, final test, and upload were not run.")
            return

        roi_jobs = [
            (lambda device, backbone=backbone, config=self.configs[backbone], seed=seed: self.audit(backbone, config, seed, "val", triplets, device, roi_only=True))
            for backbone in self.backbones for seed in self.seeds
        ]
        self.parallel(roi_jobs)

        control_scope = self.suite_config["control_scope"]
        control_backbone, control_seed = str(control_scope["backbone"]), int(control_scope["seed"])
        if control_backbone not in self.configs or control_seed not in self.all_seeds:
            raise ValueError(f"Invalid control scope: {control_backbone}/seed {control_seed}")
        control_jobs = []
        context_root = self.output / "audits/context_randomized" / control_backbone / f"seed_{control_seed:04d}"

        def context_job(device: str) -> None:
            command = [PYTHON, "scripts/07_run_controls.py", "--config", "configs/controls/context_randomized.yaml", "--model-config", self.configs[control_backbone], "--seed", str(control_seed), "--device", device, "--overwrite"]
            self.run(f"control_context_randomized_{control_backbone}_{control_seed:04d}", command, [context_root / "control_status.csv"], "controls")

        control_jobs.append(context_job)
        planted_root = self.output / "audits/planted_shortcut" / control_backbone / f"seed_{control_seed:04d}"
        planted_strengths = [float(value) for value in load_config("configs/controls/planted_shortcut.yaml")["control"]["strengths"]]
        for rho in planted_strengths:
            rho_root = planted_root / f"rho_{rho:.2f}"

            def planted_job(device: str, rho=rho, rho_root=rho_root) -> None:
                command = [PYTHON, "scripts/07_run_controls.py", "--config", "configs/controls/planted_shortcut.yaml", "--model-config", self.configs[control_backbone], "--seed", str(control_seed), "--rho", str(rho), "--device", device, "--overwrite"]
                if self.args.force:
                    command.append("--force-rhos")
                self.run(f"control_planted_{control_backbone}_{control_seed:04d}_{rho:.2f}", command, [rho_root / "directed_hcs.csv", rho_root / "directed_hce.csv"], "controls")

            control_jobs.append(planted_job)
        self.parallel(control_jobs)
        summary_code = "import pandas as pd,sys; from pathlib import Path; root=Path(sys.argv[1]); values=[float(x) for x in sys.argv[2:]]; missing=[x for x in values if not (root/f'rho_{x:.2f}/directed_hcs.csv').is_file()]; assert not missing,missing; pd.DataFrame([{'rho':x,'status':'completed'} for x in values]).to_csv(root/'rho_status.csv',index=False)"
        self.run("control_planted_summary", [PYTHON, "-c", summary_code, str(planted_root), *map(str, planted_strengths)], [planted_root / "rho_status.csv"], "controls")

        robustness_jobs = []
        for backbone, config in self.configs.items():
            for seed in self.all_seeds:
                for kind in ("source_buffer", "seam"):
                    def robustness_job(device: str, backbone=backbone, config=config, seed=seed, kind=kind) -> None:
                        checkpoint = self.output / "checkpoints" / backbone / f"seed_{seed:04d}" / "best.ckpt"
                        logits = self.output / "predictions" / backbone / f"seed_{seed:04d}" / "id_val.parquet"
                        root = self.output / "audits" / kind / backbone / f"seed_{seed:04d}"
                        self.run(f"robustness_{kind}_{backbone}_{seed:04d}", [PYTHON, "scripts/08_run_robustness.py", "--config", f"configs/audit/{kind}.yaml", "--checkpoint", str(checkpoint), "--model-config", config, "--id-logits", str(logits), "--triplets", str(triplets), "--seed", str(seed), "--device", device, "--output-dir", str(root), "--overwrite"], [root / "robustness_summary.csv"], "robustness")
                    robustness_jobs.append(robustness_job)
        self.parallel(robustness_jobs)
        for level in range(1, 5):
            for target in (0, 3, 4):
                root = self.output / "audits/identification_ladder" / f"level_{level}" / f"target_{target}"
                self.run(f"ladder_{level}_{target}", [PYTHON, "scripts/09_run_identification_ladder.py", "--level", str(level), "--target-hospital", str(target), "--split", "val", "--output-dir", str(root), "--overwrite"], [root / "triplets.parquet", root / "coverage.csv"], "robustness")
            combined = self.output / "audits/identification_ladder" / f"level_{level}" / "triplets.parquet"
            combine_code = "import pandas as pd,sys; from pathlib import Path; root=Path(sys.argv[1]); pd.concat([pd.read_parquet(root/f'target_{h}/triplets.parquet') for h in (0,3,4)],ignore_index=True).to_parquet(root/'triplets.parquet',index=False)"
            self.run(f"ladder_{level}_combine", [PYTHON, "-c", combine_code, str(combined.parent)], [combined], "robustness")
            ladder_jobs = []
            for backbone, config in self.configs.items():
                for seed in self.all_seeds:
                    def ladder_job(device: str, level=level, backbone=backbone, config=config, seed=seed, combined=combined) -> None:
                        base = self.output / "audits/identification_ladder" / f"level_{level}" / "val" / backbone / f"seed_{seed:04d}"
                        command = [PYTHON, "scripts/06_run_primary_audit.py", "--model-config", config, "--checkpoint", str(self.output / "checkpoints" / backbone / f"seed_{seed:04d}/best.ckpt"), "--id-logits", str(self.output / "predictions" / backbone / f"seed_{seed:04d}/id_val.parquet"), "--triplets", str(combined), "--split", "val", "--seed", str(seed), "--device", device, "--output-dir", str(base), "--overwrite"]
                        self.run(f"ladder_audit_{level}_{backbone}_{seed:04d}", command, [base / "directed_hcs.csv", base / "directed_hce.csv"], "robustness")
                    ladder_jobs.append(ladder_job)
            self.parallel(ladder_jobs)
        lesion = json.loads((self.output / "p0/data_integrity/alignment_report.json").read_text(encoding="utf-8"))
        lesion_available = str(lesion.get("status", "")).lower() in {"pass", "passed", "validated", "available"}
        lesion_root = self.output / "audits/lesion_aware"
        if lesion_available:
            lesion_matching = lesion_root / "matching"
            self.run("lesion_matching", [PYTHON, "scripts/03_run_matching.py", "--config", "configs/matching/lesion_aware.yaml", "--split", "val", "--output-dir", str(lesion_matching), "--overwrite"], [lesion_matching / "triplets.parquet"])
            lesion_jobs = []
            for backbone, config in self.configs.items():
                for seed in self.all_seeds:
                    def lesion_job(device: str, backbone=backbone, config=config, seed=seed) -> None:
                        base = lesion_root / "val" / backbone / f"seed_{seed:04d}"
                        command = [PYTHON, "scripts/06_run_primary_audit.py", "--model-config", config, "--checkpoint", str(self.output / "checkpoints" / backbone / f"seed_{seed:04d}/best.ckpt"), "--id-logits", str(self.output / "predictions" / backbone / f"seed_{seed:04d}/id_val.parquet"), "--triplets", str(lesion_matching / "triplets.parquet"), "--split", "val", "--seed", str(seed), "--device", device, "--output-dir", str(base), "--overwrite"]
                        self.run(f"lesion_audit_{backbone}_{seed:04d}", command, [base / "directed_hcs.csv", base / "directed_hce.csv"], "robustness")
                    lesion_jobs.append(lesion_job)
            self.parallel(lesion_jobs)
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
        final_test_jobs = []
        for backbone, config in self.configs.items():
            for seed in self.all_seeds:
                def final_test_job(device: str, backbone=backbone, config=config, seed=seed) -> None:
                    prediction = self.output / "predictions" / backbone / f"seed_{seed:04d}" / "ood_test.parquet"
                    self.run(f"predict_{backbone}_{seed:04d}_test", [PYTHON, "scripts/05_train_erm.py", "--config", config, "--seed", str(seed), "--device", device, "--predictions-only", "--allow-final-test"], [prediction])
                    self.audit(backbone, config, seed, "test", final_matching / "triplets.parquet", device)
                final_test_jobs.append(final_test_job)
        self.parallel(final_test_jobs)
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
    parser.add_argument("--devices", nargs="+", default=None, help="GPU devices used in parallel, for example cuda:0 cuda:1")
    parser.add_argument("--backbones", nargs="+", choices=["resnet50", "densenet121"], default=None, help="Run a primary-only backbone shard")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Run a primary-only seed shard")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    required = ["MEDSTYLE_DATA_ROOT", "MEDSTYLE_OUTPUT_ROOT", "MEDSTYLE_HF_REPO"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing: parser.error(f"Required environment variables are missing: {missing}")
    FinalSuite(args).execute()


if __name__ == "__main__":
    main()

"""Resumable, fail-closed orchestration of the reduced final MedStyleAudit suite.

The reduced suite is ResNet-50 only, seeds 11/42/101, hospital 1 as OOD validation
and hospital 2 as the locked final test. Every robustness setting reuses the fixed
audit subset and the same balanced triplets and checkpoint; matching is never
rematched at r=8 or the hard boundary.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from medstyleaudit.protocol import authorize_final_test, git_state
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import save_json

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

BACKBONE = "resnet50"
SEEDS = [11, 42, 101]
POPULATIONS = {"hospital1": "val", "hospital2": "test"}
PRIMARY_SETTINGS = ("primary", "random_paired", "roi_only")
ROBUSTNESS_SETTINGS = ("buffer_r8", "hard_boundary")

PARTIAL_RESUME_MODEL_FIELDS = (
    "architecture",
    "pretrained",
    "input_mean",
    "input_std",
)
PARTIAL_RESUME_TRAINING_FIELDS = (
    "epochs",
    "batch_size",
    "optimizer",
    "learning_rate",
    "weight_decay",
    "scheduler",
    "amp",
    "selection_metric",
    "maximize_metric",
)
PARTIAL_RESUME_DATA_FIELDS = ("version", "backend")


def _partial_resume_error(seed: int, checkpoint: Path, reason: str) -> RuntimeError:
    return RuntimeError(
        f"Refusing to resume seed {seed} from {checkpoint.name} because its provenance does not match "
        f"the current frozen final run: {reason}. Remove or intentionally archive the old partial "
        f"checkpoint/output directory before starting a clean final training run."
    )


def _scientific_training_config(config: dict) -> dict[str, object]:
    return {
        **{f"model.{field}": config.get("model", {}).get(field) for field in PARTIAL_RESUME_MODEL_FIELDS},
        **{f"training.{field}": config.get("training", {}).get(field) for field in PARTIAL_RESUME_TRAINING_FIELDS},
        **{f"data.{field}": config.get("data", {}).get(field) for field in PARTIAL_RESUME_DATA_FIELDS},
    }


def validate_partial_training_resume(
    seed_dir: Path,
    seed: int,
    current_model_config: dict,
    current_git_commit: str,
    current_protocol_hash: str,
) -> Path:
    """Return last.ckpt only when its persisted scientific provenance is current."""
    checkpoint = seed_dir / "last.ckpt"
    run_info_path = seed_dir / "run_info.json"
    config_path = seed_dir / "config_resolved.yaml"
    if not run_info_path.is_file():
        raise _partial_resume_error(seed, checkpoint, "run_info.json is missing")
    if not config_path.is_file():
        raise _partial_resume_error(seed, checkpoint, "config_resolved.yaml is missing")
    try:
        run_info = json.loads(run_info_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise _partial_resume_error(seed, checkpoint, f"run_info.json is invalid ({error})") from error
    try:
        recorded_config = load_config(config_path)
    except (OSError, ValueError) as error:
        raise _partial_resume_error(seed, checkpoint, f"config_resolved.yaml is invalid ({error})") from error

    if run_info.get("git_commit") != current_git_commit:
        raise _partial_resume_error(seed, checkpoint, "git_commit mismatch")
    recorded_hashes = [run_info[key] for key in ("protocol_hash", "protocol_sha256") if run_info.get(key)]
    if not recorded_hashes or any(value != current_protocol_hash for value in recorded_hashes):
        raise _partial_resume_error(seed, checkpoint, "protocol hash mismatch")
    if run_info.get("seed") != seed:
        raise _partial_resume_error(seed, checkpoint, "seed mismatch")
    if run_info.get("experiment") != f"train_erm_{BACKBONE}":
        raise _partial_resume_error(seed, checkpoint, "training experiment/architecture mismatch")
    if run_info.get("dataset_version") != current_model_config.get("data", {}).get("version"):
        raise _partial_resume_error(seed, checkpoint, "dataset version mismatch")

    current_scientific = _scientific_training_config(current_model_config)
    recorded_scientific = _scientific_training_config(recorded_config)
    mismatches = [field for field, value in current_scientific.items() if recorded_scientific[field] != value]
    if mismatches:
        raise _partial_resume_error(seed, checkpoint, f"scientific config mismatch: {', '.join(mismatches)}")
    return checkpoint


class FinalSuite:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.data = Path(os.environ["MEDSTYLE_DATA_ROOT"]).resolve()
        self.output = Path(os.environ["MEDSTYLE_OUTPUT_ROOT"]).resolve()
        self.repo_id = os.environ["MEDSTYLE_HF_REPO"]
        self.suite_config = load_config(REPO / "configs/final/FINAL_SUITE.yaml")
        self.model_config = self.suite_config["backbones"][BACKBONE]
        self.seeds = SEEDS
        self.devices = getattr(args, "devices", None) or [args.device]
        if not self.devices:
            raise ValueError("At least one device is required")
        self.state_path = self.output / "final_run_state.json"
        self.stage_root = self.output / ".stage_state"
        self.log_root = self.output / "logs" / "final_suite"
        default_state = {
            "p0": "pending",
            "train_seed_11": "pending",
            "train_seed_42": "pending",
            "train_seed_101": "pending",
            "hospital1_audit": "pending",
            "hospital1_aggregate": "pending",
            "final_test": "locked",
        }
        self.state = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.is_file() else {"stages": {}, **default_state}
        self.state_lock = threading.RLock()
        for directory in (self.output, self.stage_root, self.log_root):
            directory.mkdir(parents=True, exist_ok=True)

    def save_state(self) -> None:
        with self.state_lock:
            self.state["updated_at"] = datetime.now(timezone.utc).isoformat()
            save_json(self.state, self.state_path)

    def completed(
        self,
        name: str,
        outputs: list[Path],
        validity: Callable[[], bool] | None = None,
        command: list[str] | None = None,
        stage_signature: list[str] | None = None,
    ) -> bool:
        marker = self.stage_root / f"{name}.json"
        if self.args.force or not marker.is_file() or not all(path.exists() for path in outputs):
            return False
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("status") != "completed":
            return False
        signature = stage_signature or command
        if signature is not None:
            commit, dirty = git_state(REPO)
            tracked_hash = (REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
            if dirty or metadata.get("git_commit") != commit:
                return False
            recorded_signature = metadata.get("stage_signature")
            if recorded_signature is None and name.startswith("train_resnet50_"):
                recorded_signature = self.training_stage_signature(metadata.get("command", []))
            if recorded_signature is None:
                recorded_signature = metadata.get("command")
            if metadata.get("protocol_hash") != tracked_hash or recorded_signature != signature:
                return False
        return validity is None or bool(validity())

    def run(
        self,
        name: str,
        command: list[str],
        outputs: list[Path],
        summary: str | None = None,
        validity: Callable[[], bool] | None = None,
        stage_signature: list[str] | None = None,
    ) -> None:
        signature = stage_signature or command
        if self.completed(name, outputs, validity, command, signature):
            print(f"[resume] {name}")
            return
        with self.state_lock:
            self.state["stages"][name] = "running"
            if summary:
                self.state[summary] = "running"
            self.save_state()
        log_path = self.log_root / f"{name}.log"
        print(f"[run] {name}")
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{datetime.now(timezone.utc).isoformat()}] {' '.join(command)}\n")
            process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, text=True, check=False)
        if process.returncode or not all(path.exists() for path in outputs):
            with self.state_lock:
                self.state["stages"][name] = "failed"
                if summary:
                    self.state[summary] = "failed"
                self.save_state()
            raise RuntimeError(f"Stage {name} failed; see {log_path}")
        commit, _ = git_state(REPO)
        tracked_hash = (REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
        save_json({
            "stage": name,
            "status": "completed",
            "command": command,
            "execution_command": command,
            "stage_signature": signature,
            "outputs": [str(path) for path in outputs],
            "git_commit": commit,
            "protocol_hash": tracked_hash,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }, self.stage_root / f"{name}.json")
        with self.state_lock:
            self.state["stages"][name] = "completed"
            if summary:
                self.state[summary] = "completed"
            self.save_state()

    def parallel(self, jobs: list[Callable[[str], None]]) -> None:
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

        if self.completed("pytest", [marker], current):
            return
        command = [PYTHON, "-m", "pytest", "-q"]
        self.run("pytest_command", command, [], "p0", validity=lambda: False)
        save_json({"status": "PASS", "roi_tests": "PASS", "command": command, "git_commit": commit, "timestamp": datetime.now(timezone.utc).isoformat()}, marker)
        save_json({"stage": "pytest", "status": "completed", "outputs": [str(marker)]}, self.stage_root / "pytest.json")

    def checkpoint(self, seed: int) -> Path:
        return self.output / "checkpoints" / BACKBONE / f"seed_{seed:04d}" / "best.ckpt"

    def id_logits(self, seed: int) -> Path:
        return self.output / "predictions" / BACKBONE / f"seed_{seed:04d}" / "id_val.parquet"

    @staticmethod
    def training_stage_signature(command: list[str]) -> list[str]:
        """Normalize a training invocation to its stable scientific arguments."""
        if "scripts/05_train_erm.py" not in command:
            return []
        signature = ["scripts/05_train_erm.py"]
        for option in ("--config", "--seed"):
            if option not in command or command.index(option) + 1 >= len(command):
                return []
            signature.extend([option, command[command.index(option) + 1]])
        return signature

    def triplets_for(self, population: str) -> Path:
        return self.output / "subsets" / f"{population}_triplets.parquet"

    def subset_for(self, population: str) -> Path:
        return self.output / "subsets" / f"{population}.parquet"

    def audit_dir(self, population: str, setting: str, seed: int) -> Path:
        return self.output / "audits" / population / setting / BACKBONE / f"seed_{seed:04d}"

    def train(self, seed: int, device: str) -> None:
        directory = self.output / "checkpoints" / BACKBONE / f"seed_{seed:04d}"
        command = [PYTHON, "scripts/05_train_erm.py", "--config", self.model_config, "--seed", str(seed), "--device", device]
        outputs = [self.checkpoint(seed), directory / "metrics.csv", self.id_logits(seed), self.output / "predictions" / BACKBONE / f"seed_{seed:04d}" / "ood_val.parquet"]
        stage_name = f"train_{BACKBONE}_{seed:04d}"
        stage_signature = self.training_stage_signature(command)
        if self.completed(stage_name, outputs, command=command, stage_signature=stage_signature):
            print(f"[resume] {stage_name}")
            return
        if directory.exists():
            command.append("--overwrite")
        if (directory / "last.ckpt").is_file():
            commit, _ = git_state(REPO)
            protocol_hash = (REPO / "configs/final/protocol_sha256.txt").read_text(encoding="utf-8").strip()
            checkpoint = validate_partial_training_resume(
                directory,
                seed,
                load_config(self.model_config),
                commit,
                protocol_hash,
            )
            command += ["--resume", str(checkpoint)]
        self.run(
            stage_name,
            command,
            outputs,
            f"train_seed_{seed}",
            stage_signature=stage_signature,
        )

    def require_explicit_final_test_unlock(self) -> dict:
        try:
            return authorize_final_test(self.output, REPO / "configs/final/FINAL_PROTOCOL.yaml")
        except PermissionError as error:
            raise PermissionError(
                "Hospital 2 is locked. Run:\n python scripts/15_unlock_final_test.py\n"
                " after Hospital-1 completion."
            ) from error

    def primary_audit(self, population: str, setting: str, seed: int, triplets: Path, device: str) -> None:
        split = POPULATIONS[population]
        base = self.audit_dir(population, setting, seed)
        command = [PYTHON, "scripts/06_run_primary_audit.py", "--config", "configs/audit/primary.yaml", "--model-config", self.model_config, "--checkpoint", str(self.checkpoint(seed)), "--id-logits", str(self.id_logits(seed)), "--triplets", str(triplets), "--split", split, "--seed", str(seed), "--device", device, "--output-dir", str(base)]
        if setting == "roi_only":
            command.append("--roi-only-control")
        if split == "test":
            command.append("--allow-final-test")
        command.append("--overwrite")
        outputs = [
            base / "directed_hcs.csv",
            base / "directed_hce.csv",
            base / "directed_intervals.csv",
            base / "global_intervals.csv",
            base / "audit_records.parquet",
        ]
        self.run(f"audit_{population}_{setting}_{seed:04d}", command, outputs, f"{population}_audit")

    def robustness_audit(self, population: str, setting: str, seed: int, triplets: Path, device: str) -> None:
        base = self.audit_dir(population, setting, seed)
        if setting == "buffer_r8":
            config = "configs/audit/source_buffer.yaml"
        else:
            config = "configs/audit/seam.yaml"
        command = [PYTHON, "scripts/08_run_robustness.py", "--config", config, "--checkpoint", str(self.checkpoint(seed)), "--model-config", self.model_config, "--id-logits", str(self.id_logits(seed)), "--triplets", str(triplets), "--split", POPULATIONS[population], "--seed", str(seed), "--device", device, "--output-dir", str(base), "--overwrite"]
        if POPULATIONS[population] == "test":
            command.append("--allow-final-test")
        self.run(f"robustness_{population}_{setting}_{seed:04d}", command, [base / "robustness_summary.csv"], f"{population}_audit")

    def build_subset(self, population: str) -> None:
        split = POPULATIONS[population]
        triplets = self.output / "p0/matching/triplets.parquet"
        command = [PYTHON, "scripts/04_build_audit_subset.py", "--triplets", str(triplets), "--split", split, "--output", str(self.subset_for(population)), "--triplets-out", str(self.triplets_for(population))]
        if split == "test":
            command.append("--allow-final-test")
        self.run(f"subset_{population}", command, [self.subset_for(population), self.triplets_for(population)], "p0")

    def build_random_paired(self, population: str) -> None:
        split = POPULATIONS[population]
        triplets = self.output / "audits" / population / "random_paired" / "triplets.parquet"
        command = [PYTHON, "scripts/03_random_paired.py", "--config", "configs/matching/primary.yaml", "--subset", str(self.subset_for(population)), "--split", split, "--seed", "42", "--output", str(triplets)]
        if split == "test":
            command.append("--allow-final-test")
        self.run(f"random_paired_{population}", command, [triplets], "p0")

    def aggregate(self, population: str) -> None:
        command = [PYTHON, "scripts/12_aggregate_experiments.py", "--population", population, "--overwrite"]
        if POPULATIONS[population] == "test":
            command.append("--allow-final-test")
        output = self.output / "aggregate" / population
        self.run(
            f"aggregate_{population}",
            command,
            [output / "run_completeness.csv", output / "global_hcs.csv", output / "global_intervals.csv", output / "directed_intervals.csv"],
            f"{population}_aggregate",
        )

    def assemble_final(self, protocol_dir: Path, commit: str) -> None:
        import pandas as pd
        final = self.output / "aggregate" / "final"
        final.mkdir(parents=True, exist_ok=True)
        protocol_hash = (protocol_dir / "protocol_sha256.txt").read_text(encoding="utf-8").strip()

        def read_all(name: str) -> pd.DataFrame:
            frames = []
            for population in POPULATIONS:
                path = self.output / "aggregate" / population / f"{name}.csv"
                if path.is_file():
                    frames.append(pd.read_csv(path))
            return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

        for name in ("global_hcs", "global_hce", "global_intervals", "directed_intervals", "INDIVIDUAL_SEEDS", "run_completeness", "robustness_by_seed", "matching_results"):
            frame = read_all(name)
            if not frame.empty:
                frame.to_csv(final / f"{name}.csv", index=False)

        main = read_all("global_hcs")
        hce = read_all("global_hce")
        if not main.empty and not hce.empty:
            pd.concat([main.assign(summary_family="hcs"), hce.assign(summary_family="hce")], ignore_index=True, sort=False).to_csv(final / "MAIN_RESULTS.csv", index=False)
        completeness = read_all("run_completeness")
        if not completeness.empty:
            completeness.to_csv(final / "run_completeness.csv", index=False)
        robustness = read_all("robustness_by_seed")
        if not robustness.empty:
            robustness.to_csv(final / "ROBUSTNESS_RESULTS.csv", index=False)
        matching = read_all("matching_results")
        if not matching.empty:
            matching.to_csv(final / "MATCHING_RESULTS.csv", index=False)
        individual = read_all("INDIVIDUAL_SEEDS")
        if not individual.empty:
            individual.to_csv(final / "INDIVIDUAL_SEEDS.csv", index=False)
        global_interval_table = read_all("global_intervals")
        if not global_interval_table.empty:
            global_interval_table.to_csv(final / "GLOBAL_INTERVALS.csv", index=False)
        directed_interval_table = read_all("directed_intervals")
        if not directed_interval_table.empty:
            directed_interval_table.to_csv(final / "DIRECTED_INTERVALS.csv", index=False)
        save_json({"status": "ready_for_upload", "protocol_hash": protocol_hash, "git_commit": commit, "artifact_manifest": "MANIFEST.json", "run_state": "logs/final_suite/final_run_state.json"}, final / "experiment_manifest.json")

    def execute(self) -> None:
        if self.repo_id != "PeiyuanHao/MedStyleAudit-Experiments":
            raise OSError("MEDSTYLE_HF_REPO must be exactly PeiyuanHao/MedStyleAudit-Experiments")
        commit, dirty = git_state(REPO)
        if dirty:
            raise RuntimeError("Final suite requires a clean Git working tree")
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
        self.run("matching", [PYTHON, "scripts/03_run_matching.py", "--config", "configs/matching/primary.yaml", "--split", "val", "--overwrite"], [self.output / "p0/matching/triplets.parquet", self.output / "p0/matching/feature_balance.csv", self.output / "p0/matching/donor_reuse.csv", self.output / "p0/matching/slide_reuse.csv"], "p0")
        self.run("coverage_gate", [PYTHON, "scripts/04_check_matching_coverage.py", "--overwrite"], [self.output / "p0/matching/coverage_review/coverage_gate.json"], "p0")

        def current_preflight() -> bool:
            try:
                report = json.loads((protocol_dir / "final_preflight.json").read_text(encoding="utf-8"))
                return report.get("status") == "PASS" and report.get("git_commit") == commit and report.get("protocol_hash") == tracked_protocol_hash
            except (OSError, json.JSONDecodeError):
                return False

        self.run("final_preflight", [PYTHON, "scripts/14_final_preflight.py"], [protocol_dir / "final_preflight.json"], "p0", validity=current_preflight)

        self.build_subset("hospital1")
        self.build_random_paired("hospital1")
        self.state["p0"] = "completed"
        self.save_state()

        training_jobs = [(lambda device, seed=seed: self.train(seed, device)) for seed in self.seeds]
        self.parallel(training_jobs)

        hospital1_triplets = self.triplets_for("hospital1")
        hospital1_jobs = []
        for seed in self.seeds:
            for setting in PRIMARY_SETTINGS:
                setting_triplets = self.output / "audits" / "hospital1" / "random_paired" / "triplets.parquet" if setting == "random_paired" else hospital1_triplets
                hospital1_jobs.append((lambda device, seed=seed, setting=setting, triplets=setting_triplets: self.primary_audit("hospital1", setting, seed, triplets, device)))
            for setting in ROBUSTNESS_SETTINGS:
                hospital1_jobs.append((lambda device, seed=seed, setting=setting: self.robustness_audit("hospital1", setting, seed, hospital1_triplets, device)))
        self.parallel(hospital1_jobs)
        self.aggregate("hospital1")

        if not self.args.allow_final_test:
            self.state["final_test"] = "locked"
            self.save_state()
            print("Validation suite complete. Hospital 2 remains locked; resume with --allow-final-test.")
            return

        self.require_explicit_final_test_unlock()
        self.state["final_test"] = "running"
        self.save_state()

        self.run("final_test_matching", [PYTHON, "scripts/03_run_matching.py", "--split", "test", "--allow-final-test", "--prior-triplets", str(self.output / "p0/matching/triplets.parquet"), "--output-dir", str(self.output / "p0/matching_final_test"), "--overwrite"], [self.output / "p0/matching_final_test/triplets.parquet"])
        self.run("merge_final_test_matching", [PYTHON, "scripts/merge_final_test_matching.py", "--allow-final-test"], [self.output / "p0/matching/triplets.parquet", self.output / "p0/matching/feature_balance.csv", self.output / "p0/matching/final_matching_check.json"])
        self.build_subset("hospital2")
        self.build_random_paired("hospital2")

        final_test_jobs = []
        for seed in self.seeds:
            def predict_test(device: str, seed=seed) -> None:
                prediction = self.output / "predictions" / BACKBONE / f"seed_{seed:04d}" / "ood_test.parquet"
                self.run(f"predict_{BACKBONE}_{seed:04d}_test", [PYTHON, "scripts/05_train_erm.py", "--config", self.model_config, "--seed", str(seed), "--device", device, "--predictions-only", "--allow-final-test"], [prediction])
            final_test_jobs.append(predict_test)
        self.parallel(final_test_jobs)

        hospital2_triplets = self.triplets_for("hospital2")
        hospital2_jobs = []
        for seed in self.seeds:
            for setting in PRIMARY_SETTINGS:
                setting_triplets = self.output / "audits" / "hospital2" / "random_paired" / "triplets.parquet" if setting == "random_paired" else hospital2_triplets
                hospital2_jobs.append((lambda device, seed=seed, setting=setting, triplets=setting_triplets: self.primary_audit("hospital2", setting, seed, triplets, device)))
            for setting in ROBUSTNESS_SETTINGS:
                hospital2_jobs.append((lambda device, seed=seed, setting=setting: self.robustness_audit("hospital2", setting, seed, hospital2_triplets, device)))
        self.parallel(hospital2_jobs)
        self.aggregate("hospital2")

        self.assemble_final(protocol_dir, commit)
        shutil.copy2(REPO / "docs/HF_DATASET_CARD.md", self.output / "README.md")
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
        self.state["final_test"] = "completed"
        self.save_state()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--devices", nargs="+", default=None, help="GPU devices used in parallel, for example cuda:0 cuda:1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume completed stages (default behavior unless --force)")
    parser.add_argument("--allow-final-test", action="store_true")
    args = parser.parse_args()
    required = ["MEDSTYLE_DATA_ROOT", "MEDSTYLE_OUTPUT_ROOT", "MEDSTYLE_HF_REPO"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        parser.error(f"Required environment variables are missing: {missing}")
    FinalSuite(args).execute()


if __name__ == "__main__":
    main()

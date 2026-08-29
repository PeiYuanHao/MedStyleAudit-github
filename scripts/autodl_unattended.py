"""Fail-safe finalization and small-result export for unattended AutoDL runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

GITHUB_MAX_FILE_BYTES = 10 * 1024 * 1024
FORBIDDEN_EXPORT_SUFFIXES = {".ckpt", ".npy", ".npz", ".parquet", ".pt", ".pth"}
PROTOCOL_FILES = (
    "FINAL_PROTOCOL.yaml",
    "protocol_sha256.txt",
    "git_commit.txt",
    "final_preflight.json",
    "final_test_lock.json",
    "autodl_final_status.json",
)
P0_FILES = (
    "directed_coverage.csv",
    "common_support_coverage.csv",
    "matching_balance.csv",
    "feature_balance.csv",
    "donor_reuse_summary.csv",
    "slide_reuse_summary.csv",
    "final_matching_check.json",
)

CommandRunner = Callable[[list[str], Path], int]
GithubPusher = Callable[[Path, Path, str, str], str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def suite_command(phase: str, devices: list[str], python: str = sys.executable) -> list[str]:
    if phase not in {"hospital1", "hospital2"}:
        raise ValueError(f"Unsupported unattended phase: {phase}")
    if not devices:
        raise ValueError("At least one device is required")
    command = [python, "scripts/run_final_suite.py"]
    if phase == "hospital2":
        command.extend(["--resume", "--allow-final-test"])
    command.extend(["--devices", *devices])
    return command


def _candidate_files(root: Path) -> list[Path]:
    candidates = [root / "protocol" / name for name in PROTOCOL_FILES]
    for aggregate in (root / "aggregate" / name for name in ("hospital1", "hospital2", "final")):
        if aggregate.is_dir():
            candidates.extend(path for path in aggregate.rglob("*") if path.is_file())
    candidates.extend(root / "p0" / "matching" / name for name in P0_FILES)
    log_root = root / "logs" / "final_suite"
    if log_root.is_dir():
        candidates.extend(path for path in log_root.rglob("*") if path.is_file())
    candidates.append(root / "MANIFEST.json")
    return sorted(set(candidates))


def stage_github_export(
    output_root: str | Path,
    *,
    max_file_bytes: int = GITHUB_MAX_FILE_BYTES,
    sensitive_values: tuple[str, ...] = (),
) -> tuple[Path, dict[str, Any]]:
    """Stage only small, explicitly scoped result/provenance files."""
    root = Path(output_root).resolve()
    destination = root / "github_export"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    copied, skipped = [], []
    for source in _candidate_files(root):
        if not source.exists():
            continue
        relative = source.relative_to(root)
        size = source.stat().st_size
        reason = None
        if source.is_symlink():
            reason = "symbolic links are not exported"
        elif source.suffix.lower() in FORBIDDEN_EXPORT_SUFFIXES:
            reason = "large/scientific artifact type is reserved for Hugging Face"
        elif size > max_file_bytes:
            reason = f"file exceeds maximum individual size of {max_file_bytes} bytes"
        elif any(value.encode() in source.read_bytes() for value in sensitive_values if value):
            reason = "file contains a configured secret"
        if reason:
            skipped.append({"path": relative.as_posix(), "bytes": size, "reason": reason})
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append({"path": relative.as_posix(), "bytes": size})
    manifest = {
        "created_at": utc_now(),
        "source_root": str(root),
        "max_file_bytes": int(max_file_bytes),
        "copied": copied,
        "skipped": skipped,
    }
    (destination / "export_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return destination, manifest


def _git(command: list[str], cwd: Path) -> None:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Git result export command failed ({command[1]}), exit code {result.returncode}")


def push_github_export(repo_root: Path, export_dir: Path, phase: str, timestamp: str) -> str:
    """Commit an export snapshot from a temporary worktree and push a result branch."""
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    compact_timestamp = parsed.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    branch = f"results/{phase}-{compact_timestamp}"
    run_id = f"{phase}-{compact_timestamp}"
    temporary = Path(tempfile.mkdtemp(prefix="medstyleaudit-github-export-"))
    worktree = temporary / "worktree"
    added = False
    try:
        _git(["git", "worktree", "add", "--detach", str(worktree), "HEAD"], repo_root)
        added = True
        _git(["git", "switch", "-c", branch], worktree)
        archive = worktree / "results" / "final-runs" / run_id
        shutil.copytree(export_dir, archive)
        _git(["git", "add", "-f", str(archive.relative_to(worktree))], worktree)
        _git(["git", "commit", "-m", f"results: archive MedStyleAudit {phase} final run"], worktree)
        _git(["git", "push", "origin", f"HEAD:refs/heads/{branch}"], worktree)
        return branch
    finally:
        if added:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo_root,
                text=True,
                capture_output=True,
                check=False,
            )
        shutil.rmtree(temporary, ignore_errors=True)


def _run_command(command: list[str], cwd: Path) -> int:
    try:
        return subprocess.run(command, cwd=cwd, check=False).returncode
    except OSError:
        return 127


def _write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def finalize_unattended_run(
    *,
    repo_root: Path,
    output_root: Path,
    phase: str,
    suite_exit_code: int,
    started_at: str,
    environment: Mapping[str, str] | None = None,
    command_runner: CommandRunner = _run_command,
    github_pusher: GithubPusher = push_github_export,
    shutdown: bool = True,
) -> dict[str, Any]:
    """Attempt every backup path and shutdown without masking suite status."""
    env = dict(os.environ if environment is None else environment)
    status_path = output_root / "protocol" / "autodl_final_status.json"
    protocol_path = output_root / "protocol" / "protocol_sha256.txt"
    try:
        commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, capture_output=True, check=False
        )
        git_commit = commit_result.stdout.strip() if commit_result.returncode == 0 else None
    except OSError:
        git_commit = None
    status: dict[str, Any] = {
        "status": "success" if suite_exit_code == 0 else "failed",
        "suite_exit_code": int(suite_exit_code),
        "phase": phase,
        "git_commit": git_commit,
        "protocol_sha256": protocol_path.read_text(encoding="utf-8").strip() if protocol_path.is_file() else None,
        "started_at": started_at,
        "finished_at": utc_now(),
        "hf_upload_attempted": False,
        "hf_upload_succeeded": False,
        "hf_verification_succeeded": False,
        "github_export_attempted": False,
        "github_export_succeeded": False,
        "github_result_branch": None,
    }
    _write_status(status_path, status)

    if env.get("HF_TOKEN"):
        status["hf_upload_attempted"] = True
        _write_status(status_path, status)
        upload = [sys.executable, "scripts/hf_upload_artifacts.py", "--root", str(output_root)]
        status["hf_upload_succeeded"] = command_runner(upload, repo_root) == 0
        verify = [sys.executable, "scripts/hf_verify_artifacts.py"]
        status["hf_verification_succeeded"] = command_runner(verify, repo_root) == 0
        status["finished_at"] = utc_now()
        _write_status(status_path, status)

    if env.get("MEDSTYLE_GITHUB_EXPORT", "1") == "1":
        status["github_export_attempted"] = True
        status["github_export_succeeded"] = True  # written into the snapshot; corrected locally on failure
        status["finished_at"] = utc_now()
        _write_status(status_path, status)
        try:
            maximum = int(env.get("MEDSTYLE_GITHUB_MAX_FILE_BYTES", GITHUB_MAX_FILE_BYTES))
            if maximum <= 0:
                raise ValueError("MEDSTYLE_GITHUB_MAX_FILE_BYTES must be positive")
            secrets = tuple(
                value for name in ("HF_TOKEN", "GITHUB_TOKEN", "GH_TOKEN") if (value := env.get(name))
            )
            export_dir, _ = stage_github_export(
                output_root, max_file_bytes=maximum, sensitive_values=secrets
            )
            branch = github_pusher(repo_root, export_dir, phase, status["finished_at"])
            status["github_result_branch"] = branch
        except (OSError, RuntimeError, ValueError, shutil.Error, subprocess.SubprocessError) as error:
            status["github_export_succeeded"] = False
            status["github_export_error"] = str(error)
            try:
                stage_github_export(
                    output_root,
                    max_file_bytes=GITHUB_MAX_FILE_BYTES,
                    sensitive_values=tuple(
                        value
                        for name in ("HF_TOKEN", "GITHUB_TOKEN", "GH_TOKEN")
                        if (value := env.get(name))
                    ),
                )
            except (OSError, RuntimeError, ValueError, shutil.Error) as restage_error:
                status["github_export_restage_error"] = str(restage_error)
        status["finished_at"] = utc_now()
        _write_status(status_path, status)

    if status["hf_upload_attempted"]:
        command_runner(
            [
                sys.executable,
                "scripts/hf_upload_artifacts.py",
                "--file",
                str(status_path),
                "--hf-path",
                "protocol/autodl_final_status.json",
            ],
            repo_root,
        )
    command_runner(["sync"], repo_root)
    if shutdown:
        status["shutdown_attempted"] = True
        status["shutdown_succeeded"] = command_runner(["/usr/bin/shutdown"], repo_root) == 0
        if not status["shutdown_succeeded"]:
            print("ERROR: /usr/bin/shutdown failed; manual AutoDL shutdown is required.", file=sys.stderr)
        status["finished_at"] = utc_now()
        _write_status(status_path, status)
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize-hospital2")
    authorize.add_argument("--output-root", type=Path, required=True)
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--phase", choices=["hospital1", "hospital2"], required=True)
    finalize.add_argument("--suite-exit-code", type=int, required=True)
    finalize.add_argument("--started-at", required=True)
    finalize.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.command == "authorize-hospital2":
        from medstyleaudit.protocol import authorize_final_test

        authorize_final_test(args.output_root, repo_root / "configs/final/FINAL_PROTOCOL.yaml")
        print("Hospital 2 explicit unlock authorization: PASS")
        return
    finalize_unattended_run(
        repo_root=repo_root,
        output_root=args.output_root.resolve(),
        phase=args.phase,
        suite_exit_code=args.suite_exit_code,
        started_at=args.started_at,
    )


if __name__ == "__main__":
    main()

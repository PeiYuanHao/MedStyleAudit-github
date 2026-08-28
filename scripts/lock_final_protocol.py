"""Resolve and lock FINAL_PROTOCOL.yaml into the experiment artifact root."""

from __future__ import annotations

import argparse
from pathlib import Path

from medstyleaudit.protocol import git_state, resolved_protocol, write_protocol_lock
from medstyleaudit.utils.io import save_json, save_yaml
from medstyleaudit.utils.paths import experiment_path
from medstyleaudit.utils.run_metadata import environment_info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/final/FINAL_PROTOCOL.yaml")
    parser.add_argument("--expected-hash", default="configs/final/protocol_sha256.txt")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    output = args.output_dir or experiment_path("protocol")
    output.mkdir(parents=True, exist_ok=True)
    resolved = resolved_protocol(args.protocol)
    locked = output / "FINAL_PROTOCOL.yaml"
    if locked.exists() and resolved_protocol(locked) != resolved:
        raise PermissionError("A different final protocol is already locked in the artifact root")
    from medstyleaudit.protocol import protocol_sha256
    expected_hash = Path(args.expected_hash).read_text(encoding="utf-8").strip()
    actual_hash = protocol_sha256(resolved)
    if expected_hash != actual_hash:
        raise PermissionError(f"Tracked final protocol lock mismatch: expected {expected_hash}, got {actual_hash}")
    save_yaml(resolved, locked)
    digest = write_protocol_lock(locked, output / "protocol_sha256.txt")
    commit, _ = git_state()
    (output / "git_commit.txt").write_text(commit + "\n", encoding="utf-8")
    save_json(environment_info(), output / "environment.json")
    print(digest)


if __name__ == "__main__":
    main()

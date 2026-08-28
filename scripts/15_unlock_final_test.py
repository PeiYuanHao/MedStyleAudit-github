"""Explicitly unlock the frozen Hospital-2 final test.

Unlock only succeeds when the required final-protocol metadata is present and
self-consistent: a tracked protocol hash, a PASS final preflight for that hash,
and a clean Git commit matching the preflight. On success the unlock marker
records the timestamp, git commit, and protocol SHA256.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from medstyleaudit.protocol import git_state, verify_protocol_lock
from medstyleaudit.utils.io import save_json
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default="configs/final/FINAL_PROTOCOL.yaml")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    root = experiment_path("")
    protocol_dir = root / "protocol"
    hash_path = protocol_dir / "protocol_sha256.txt"
    preflight_path = protocol_dir / "final_preflight.json"
    if not Path(args.protocol).is_file() or not hash_path.is_file() or not preflight_path.is_file():
        raise PermissionError("Final test cannot be unlocked without the protocol, protocol hash, and preflight report")
    protocol_hash = verify_protocol_lock(args.protocol, hash_path)
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "PASS" or preflight.get("protocol_hash") != protocol_hash:
        raise PermissionError("Final preflight is not PASS for the current protocol")
    commit, dirty = git_state()
    if dirty or preflight.get("git_commit") != commit:
        raise PermissionError("Git must be clean and match the preflight commit")
    destination = args.output or protocol_dir / "final_test_lock.json"
    payload = {
        "status": "unlocked",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": commit,
        "protocol_sha256": protocol_hash,
    }
    save_json(payload, destination)
    print(f"Final test unlocked: {destination}")


if __name__ == "__main__":
    main()

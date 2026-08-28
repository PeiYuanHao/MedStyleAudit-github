"""Upload changed, derived artifacts and publish MANIFEST.json last."""

from __future__ import annotations

import argparse
from pathlib import Path

from medstyleaudit.artifacts.huggingface import directory_files, upload_artifacts
from medstyleaudit.protocol import git_state
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--file", type=Path, default=None)
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--split", default=None)
    args = parser.parse_args()
    root = args.root or experiment_path("")
    if args.file:
        if not args.hf_path:
            parser.error("--hf-path is required with --file")
        files = [(args.file, args.hf_path)]
    else:
        files = directory_files(root)
    protocol_hash = (root / "protocol" / "protocol_sha256.txt").read_text(encoding="utf-8").strip()
    commit, _ = git_state()
    manifest = upload_artifacts(files, repo_id=args.repo, metadata={"experiment": args.experiment, "backbone": args.backbone, "seed": args.seed, "split": args.split, "git_commit": commit, "protocol_hash": protocol_hash}, manifest_path=root / "MANIFEST.json")
    print(f"Uploaded/verified {len(manifest.entries)} manifest entries")


if __name__ == "__main__":
    main()

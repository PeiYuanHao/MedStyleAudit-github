"""Verify local or remote final artifacts against MANIFEST.json."""

import argparse
from pathlib import Path

from medstyleaudit.artifacts.huggingface import verify_remote_artifacts
from medstyleaudit.artifacts.manifest import ArtifactManifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument("--local-root", type=Path, default=None)
    args = parser.parse_args()
    if args.local_root:
        manifest = ArtifactManifest.read(args.local_root / "MANIFEST.json")
        manifest.verify_local(args.local_root)
    else:
        manifest = verify_remote_artifacts(repo_id=args.repo)
    print(f"PASS: verified {len(manifest.entries)} artifacts")


if __name__ == "__main__":
    main()

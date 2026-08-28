"""Download every manifest artifact and fail on checksum mismatch."""

import argparse
from pathlib import Path

from medstyleaudit.artifacts.huggingface import download_manifest_artifacts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repo", default=None)
    args = parser.parse_args()
    manifest = download_manifest_artifacts(args.destination, repo_id=args.repo)
    print(f"Downloaded and verified {len(manifest.entries)} artifacts")


if __name__ == "__main__":
    main()

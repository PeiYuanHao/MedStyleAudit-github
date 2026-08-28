"""Create the private Hugging Face Dataset artifact repository."""

import argparse

from medstyleaudit.artifacts.huggingface import create_repository


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=None)
    parser.add_argument("--public", action="store_true", help="Explicitly create a public repository instead of the private default")
    args = parser.parse_args()
    print(create_repository(args.repo, private=not args.public))


if __name__ == "__main__":
    main()

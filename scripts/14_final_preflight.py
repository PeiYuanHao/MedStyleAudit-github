"""Run the automated final preflight; exit nonzero on any failed condition."""

from __future__ import annotations

import argparse
from pathlib import Path

from medstyleaudit.final_preflight import run_final_preflight
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/final/FINAL_SUITE.yaml")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    config = load_config(args.config)["preflight"]
    output = args.output or experiment_path("protocol/final_preflight.json")
    report = run_final_preflight(config, output)
    if report["status"] != "PASS":
        failed = ", ".join(item["check"] for item in report["checks"] if item["status"] == "FAIL")
        raise SystemExit(f"Final preflight FAIL: {failed}. See {output}")
    print(f"Final preflight PASS: {output}")


if __name__ == "__main__":
    main()

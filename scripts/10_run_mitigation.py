"""Backward-compatible entry point; mitigation moved to script 11."""

from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import save_json
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    args = common_parser("Prepare secondary context-consistency mitigation", "configs/models/resnet50.yaml").parse_args(); config = load_config(args.config); seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("mitigation")
    run = start_run("context_consistency_mitigation", config, output, seed, overwrite=args.overwrite); save_json({"status": "not_implemented", "note": "Secondary mitigation requires a validated primary audit and a counterfactual training loader. No result is inferred by this setup command."}, output / "mitigation_status.json"); run.complete(status="not_implemented", mitigation_status="not_implemented")


if __name__ == "__main__": main()

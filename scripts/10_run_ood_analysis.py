"""Phase 10: aggregate saved split predictions without retraining."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from medstyleaudit.utils.cli import common_parser, enforce_final_test_guard
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import read_table, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = common_parser("Aggregate ID/OOD prediction performance", "configs/models/resnet50.yaml"); parser.add_argument("--predictions", type=Path, nargs="+", required=True); parser.add_argument("--include-final-test", action="store_true")
    args = parser.parse_args(); config = load_config(args.config); seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("ood_analysis")
    run = start_run("ood_analysis", config, output, seed, overwrite=args.overwrite); rows = []
    for path in args.predictions:
        table = read_table(path); split = str(table["split"].iloc[0]) if "split" in table else path.stem
        if split in {"test", "ood_test"}: enforce_final_test_guard(split, args.allow_final_test and args.include_final_test)
        label = table["label"].astype(int); probability = table["probability"] if "probability" in table else 1 / (1 + __import__("numpy").exp(-table["logit"]))
        rows.append({"split": split, "n": len(table), "accuracy": accuracy_score(label, probability >= .5), "auroc": roc_auc_score(label, probability) if label.nunique() > 1 else float("nan"), "prediction_file": str(path)})
    save_table(pd.DataFrame(rows), output / "ood_metrics.csv"); run.complete(status="completed", n_splits=len(rows))


if __name__ == "__main__": main()

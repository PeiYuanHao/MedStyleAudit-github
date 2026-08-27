"""Phase 1: validate WSI lesion-coordinate mapping before using lesion features."""

from pathlib import Path

import pandas as pd

from medstyleaudit.data.lesion_mapping import lesion_features, project_polygons, read_camelyon_xml, validate_alignment
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import save_json, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = common_parser("Validate CAMELYON17 lesion annotation mapping", "configs/data/camelyon17.yaml")
    parser.add_argument("--mapping-csv", type=Path, default=None)
    args = parser.parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("lesion_mapping")
    mapping_csv = args.mapping_csv or experiment_path("data_integrity/patch_mapping.csv")
    run = start_run("lesion_mapping", config, output, seed, overwrite=args.overwrite)
    required = {"source_id", "label", "x", "y", "annotation_path"}
    if not mapping_csv.exists():
        status = {"status": "unavailable", "reason": f"patch-coordinate mapping not found: {mapping_csv}"}
        save_json(status, output / "alignment_report.json"); run.complete(status="completed", lesion_mapping="unavailable"); return
    mapping = pd.read_csv(mapping_csv)
    missing = required - set(mapping)
    if missing:
        status = {"status": "unavailable", "reason": f"mapping missing columns: {sorted(missing)}"}
        save_json(status, output / "alignment_report.json"); run.complete(status="completed", lesion_mapping="unavailable"); return
    if args.dry_run: mapping = mapping.head(32)
    rows, alignment = [], []
    polygon_cache = {}
    for record in mapping.itertuples(index=False):
        annotation_path = Path(record.annotation_path)
        if not annotation_path.is_file():
            continue
        polygons = polygon_cache.setdefault(str(annotation_path), read_camelyon_xml(annotation_path))
        mask = project_polygons(polygons, (float(record.x), float(record.y)), int(getattr(record, "patch_size", 96)), float(getattr(record, "scale", 1)))
        alignment.append((mask, int(record.label)))
        rows.append({"source_id": record.source_id, **lesion_features(mask)})
    report = validate_alignment(alignment)
    save_json(report, output / "alignment_report.json")
    save_table(pd.DataFrame(rows), output / "lesion_features.csv")
    run.complete(status="completed", lesion_mapping=report["status"])


if __name__ == "__main__": main()

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
    required = {"source_id", "label", "x", "y", "annotation_path", "mapping_status", "patch_size", "scale", "coordinate_level", "orientation", "coordinate_reference"}
    if not mapping_csv.exists():
        status = {"status": "unavailable", "reason": f"patch-coordinate mapping not found: {mapping_csv}"}
        save_json(status, output / "alignment_report.json"); run.complete(status="completed", lesion_mapping="unavailable"); return
    mapping = pd.read_csv(mapping_csv)
    missing = required - set(mapping)
    if missing:
        status = {"status": "unavailable", "reason": f"mapping missing columns: {sorted(missing)}"}
        save_json(status, output / "alignment_report.json"); run.complete(status="completed", lesion_mapping="unavailable"); return
    if args.dry_run: mapping = mapping.head(32)
    eligible = mapping[mapping["mapping_status"] == "mapped"].copy()
    if eligible.empty:
        counts = mapping["mapping_status"].value_counts(dropna=False).to_dict()
        status = {"status": "unavailable", "reason": "no fully mapped WSI/XML patch records", "mapping_status_counts": counts, "n": 0}
        save_json(status, output / "alignment_report.json")
        save_table(pd.DataFrame(columns=["source_id", "peripheral_tumor_presence", "peripheral_tumor_fraction", "peripheral_tumor_distance", "peripheral_largest_component", "transplanted_area_tumor_fraction"]), output / "lesion_features.csv")
        run.complete(status="completed", lesion_mapping="unavailable"); return
    rows, alignment = [], []
    polygon_cache = {}
    for record in eligible.itertuples(index=False):
        annotation_path = Path(record.annotation_path)
        if not annotation_path.is_file():
            continue
        polygons = polygon_cache.setdefault(str(annotation_path), read_camelyon_xml(annotation_path))
        mask = project_polygons(
            polygons,
            (float(record.x), float(record.y)),
            int(record.patch_size),
            float(record.scale),
            str(record.orientation),
            str(record.coordinate_reference),
        )
        alignment.append((mask, int(record.label)))
        rows.append({"source_id": record.source_id, **lesion_features(mask)})
    report = validate_alignment(alignment)
    report["mapped_records"] = len(eligible)
    report["processed_records"] = len(alignment)
    if len(alignment) != len(eligible):
        report = {**report, "status": "unavailable", "reason": "mapped annotation assets disappeared before validation"}
    save_json(report, output / "alignment_report.json")
    save_table(pd.DataFrame(rows), output / "lesion_features.csv")
    run.complete(status="completed", lesion_mapping=report["status"])


if __name__ == "__main__": main()

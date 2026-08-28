"""Phase 0: inspect metadata, slide mapping, and physical clustering."""

from pathlib import Path

import pandas as pd

from medstyleaudit.data.metadata import load_metadata_csv, metadata_summary, validate_integrity
from medstyleaudit.data.slide_mapping import build_patch_mapping, exact_manifest_from_metadata, map_slides
from medstyleaudit.data.wilds_loader import extract_metadata, load_wilds_dataset
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import save_json, save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.seed import seed_everything
from medstyleaudit.utils.paths import configured_output


def main() -> None:
    args = common_parser("Validate Camelyon17-WILDS integrity", "configs/data/camelyon17.yaml").parse_args()
    config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42))
    output = args.output_dir or configured_output(config, "data_integrity")
    run = start_run("data_integrity", config, output, seed, overwrite=args.overwrite)
    seed_everything(seed)
    metadata_path = config["data"].get("metadata_csv")
    frame = load_metadata_csv(metadata_path) if metadata_path else extract_metadata(load_wilds_dataset(config))
    run.logger.info("metadata loaded: %d patches", len(frame))
    if args.dry_run:
        frame = frame.head(int(config["data"].get("dry_run_samples", 128))).copy()
        run.logger.info("dry run restricted to %d metadata rows", len(frame))
    save_table(metadata_summary(frame), output / "wilds_summary.csv")
    save_table(frame, output / "wilds_metadata.csv")
    run.logger.info("metadata tables saved; building exact slide mapping")
    report = validate_integrity(frame, config)
    manifest_path = config["data"].get("slide_manifest_csv")
    wsi_root = Path(config["data"].get("camelyon17_wsi_root", ""))
    mapping_unavailable_reason = "CAMELYON17 WSI root is not installed"
    try:
        manifest = pd.read_csv(manifest_path) if manifest_path else (exact_manifest_from_metadata(frame, wsi_root) if wsi_root.exists() else None)
    except (ValueError, FileNotFoundError) as error:
        manifest = None
        mapping_unavailable_reason = str(error)
    if manifest is not None:
        mapping = map_slides(frame, manifest)
        save_table(mapping, output / "slide_mapping.csv")
        report["slide_mapping_completeness"] = {"status": "PASS" if (mapping["mapping_status"] == "mapped").all() else "FAIL", "mapped": int((mapping["mapping_status"] == "mapped").sum()), "total": len(mapping)}
        report["ambiguous_mappings"] = {"status": "PASS" if not (mapping["mapping_status"] == "ambiguous").any() else "FAIL", "count": int((mapping["mapping_status"] == "ambiguous").sum())}
    else:
        mapping = pd.DataFrame({"wilds_slide_id": sorted(frame["slide_id"].unique()), "mapping_status": "unavailable", "mapping_reason": mapping_unavailable_reason})
        save_table(mapping, output / "slide_mapping.csv")
        report["slide_mapping_completeness"] = {"status": "UNAVAILABLE", "reason": mapping_unavailable_reason}
        report["ambiguous_mappings"] = {"status": "UNAVAILABLE", "reason": "slide mapping unavailable"}
    patch_config = config.get("patch_mapping", {})
    patch_mapping = build_patch_mapping(
        frame,
        mapping,
        annotation_root=patch_config.get("annotation_root"),
        patch_size=patch_config.get("patch_size"),
        scale=patch_config.get("scale"),
        coordinate_level=patch_config.get("coordinate_level"),
        orientation=patch_config.get("orientation"),
        coordinate_reference=patch_config.get("coordinate_reference"),
        show_progress=True,
    )
    save_table(patch_mapping, output / "patch_mapping.csv")
    run.logger.info("patch mapping saved: %d records", len(patch_mapping))
    status_counts = patch_mapping["mapping_status"].value_counts(dropna=False).to_dict()
    report["patch_mapping"] = {
        "status": "PASS" if status_counts.get("mapped", 0) == len(patch_mapping) else "UNAVAILABLE",
        "counts": status_counts,
        "total": len(patch_mapping),
    }
    save_json(report, output / "integrity_report.json")
    run.complete(status="completed", integrity_status=report["overall_status"])


if __name__ == "__main__":
    main()

"""Phase 2: build fixed tissue masks and mask-derived content descriptors."""

from pathlib import Path

import pandas as pd

from medstyleaudit.data.metadata import load_metadata_csv
from medstyleaudit.data.wilds_loader import extract_metadata, load_wilds_dataset
from medstyleaudit.preprocessing.descriptors import descriptor_vector, mask_descriptor
from medstyleaudit.preprocessing.tissue_mask import hed_appearance_perturbation, mask_stability, overlap_metrics, tissue_mask
from medstyleaudit.utils.cli import common_parser
from medstyleaudit.utils.config import load_config
from medstyleaudit.utils.io import save_table
from medstyleaudit.utils.run_metadata import start_run
from medstyleaudit.utils.paths import experiment_path


def main() -> None:
    parser = common_parser("Build primary mask-derived descriptors", "configs/data/camelyon17.yaml")
    parser.add_argument("--metadata-csv", type=Path, default=None)
    args = parser.parse_args(); config = load_config(args.config)
    seed = args.seed if args.seed is not None else int(config.get("seed", 42)); output = args.output_dir or experiment_path("p0/descriptors")
    run = start_run("build_descriptors", config, output, seed, overwrite=args.overwrite)
    dataset = load_wilds_dataset(config)
    metadata = load_metadata_csv(args.metadata_csv) if args.metadata_csv else extract_metadata(dataset)
    if args.dry_run: metadata = metadata.head(int(config["data"].get("dry_run_samples", 128)))
    mask_config = config.get("mask", {})
    rows, stability_rows = [], []
    from tqdm.auto import tqdm
    records = tqdm(metadata.itertuples(index=False), total=len(metadata), desc="Phase 2 descriptors", unit="patch")
    for count, record in enumerate(records):
        image, _, _ = dataset[int(record.source_id)]
        rgb = __import__("numpy").asarray(image.convert("RGB"))
        reference_mask = tissue_mask(rgb, mask_config); descriptor = mask_descriptor(reference_mask)
        rows.append({column: getattr(record, column) for column in metadata.columns if hasattr(record, column)} | descriptor)
        if count < (16 if args.dry_run else 256):
            for stability in mask_stability(rgb, mask_config, [{"saturation_min": mask_config.get("saturation_min", .05) * .9}, {"saturation_min": mask_config.get("saturation_min", .05) * 1.1}]):
                stability_rows.append({"source_id": record.source_id, "hospital_id": record.hospital_id, "kind": "threshold", **stability})
            for perturbation, (h_scale, e_scale) in enumerate([(0.9, 1.0), (1.1, 1.0), (1.0, 0.9), (1.0, 1.1)]):
                perturbed_mask = tissue_mask(hed_appearance_perturbation(rgb, h_scale, e_scale), mask_config)
                perturbed_descriptor = mask_descriptor(perturbed_mask)
                stability_rows.append({"source_id": record.source_id, "hospital_id": record.hospital_id, "kind": "hed", "perturbation": perturbation, "h_scale": h_scale, "e_scale": e_scale, **overlap_metrics(reference_mask, perturbed_mask), "descriptor_l2": float(__import__("numpy").linalg.norm(descriptor_vector(descriptor) - descriptor_vector(perturbed_descriptor)))})
    save_table(pd.DataFrame(rows), output / "descriptors.parquet")
    save_table(pd.DataFrame(stability_rows), output / "mask_stability.parquet")
    run.complete(status="completed", n_descriptors=len(rows))


if __name__ == "__main__": main()

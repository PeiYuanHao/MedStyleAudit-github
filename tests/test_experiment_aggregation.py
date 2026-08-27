import json

import pandas as pd
import yaml

from medstyleaudit.audit.experiment_aggregation import RUN_TABLES, aggregate_experiment_runs


def _write_run(root, architecture, seed, audit_config):
    directory = root / architecture / f"seed_{seed:04d}" / "val"; directory.mkdir(parents=True)
    (directory / "run_info.json").write_text(json.dumps({"status": "completed", "seed": seed, "dataset_version": "v1"}), encoding="utf-8")
    (directory / "config_resolved.yaml").write_text(yaml.safe_dump({"model_architecture": architecture, "audit": audit_config}), encoding="utf-8")
    base = {"source_split": ["val"], "seed": [seed], "backbone": [architecture], "source_hospital": [1], "target_hospital": [0]}
    tables = {
        "directed_hcs": pd.DataFrame({**base, "hcs": [0.2]}),
        "directed_hce": pd.DataFrame({**base, "hce": [-0.1]}),
        "global_hcs_pair_weighted": pd.DataFrame({"source_split": ["val"], "seed": [seed], "backbone": [architecture], "status": ["available"], "hcs_pair_weighted": [0.2]}),
        "global_hcs_common_support": pd.DataFrame({"source_split": ["val"], "seed": [seed], "backbone": [architecture], "status": ["available"], "hcs_common_support": [0.2]}),
        "global_hce_pair_weighted": pd.DataFrame({"source_split": ["val"], "seed": [seed], "backbone": [architecture], "status": ["available"], "hce_pair_weighted": [-0.1]}),
        "global_hce_common_support": pd.DataFrame({"source_split": ["val"], "seed": [seed], "backbone": [architecture], "hce_common_support": [-0.1]}),
        "directed_intervals": pd.DataFrame({**base, "metric": ["hcs"], "estimate": [0.2]}),
    }
    assert set(tables) == set(RUN_TABLES)
    for name, table in tables.items(): table.to_csv(directory / f"{name}.csv", index=False)


def test_aggregator_records_every_missing_seed_and_backbone(tmp_path):
    audit = {"expected_directed_pairs": {"val": [[1, 0]]}, "roi_size": 32}
    root, output = tmp_path / "runs", tmp_path / "combined"
    _write_run(root, "resnet50", 1, audit)
    result = aggregate_experiment_runs(root, output, {"resnet50": {"seeds": [1, 2], "dataset_version": "v1"}, "densenet121": {"seeds": [1], "dataset_version": "v1"}}, split="val", audit_config=audit, expected_pairs={(1, 0)})
    missing = result["missing_runs"]
    assert set(zip(missing["backbone"], missing["seed"])) == {("resnet50", 2), ("densenet121", 1)}
    assert len(result["combined_directed_hcs"]) == 1
    assert all((output / f"{name}.csv").is_file() for name in result)

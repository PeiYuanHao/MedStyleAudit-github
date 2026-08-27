import pandas as pd

from medstyleaudit.audit.directed import aggregate_audit


def test_aggregate_audit_persists_all_scientific_tables(tmp_path):
    predictions = pd.DataFrame({
        "source_id": [1, 2], "source_split": ["val", "val"],
        "source_hospital": [1, 1], "target_hospital": [0, 0],
        "original_logit": [0., 1.], "within_logit": [.2, .8],
        "cross_logit": [.5, 1.7], "common_support": [True, True],
        "seed": [42, 42], "backbone": ["synthetic", "synthetic"],
    })
    outputs = aggregate_audit(predictions, pd.Series([-1., 0., 1., 2.]), tmp_path)
    assert set(outputs) == {
        "audit_records", "source_metrics", "directed_hcs", "directed_hce", "directed_intervals",
        "global_hcs_pair_weighted", "global_hcs_common_support",
        "global_hce_pair_weighted", "global_hce_common_support",
    }
    assert all((tmp_path / f"{name}.csv").is_file() for name in outputs)


def test_main_aggregation_does_not_average_an_absent_configured_pair(tmp_path):
    predictions = pd.DataFrame({
        "source_id": [1], "source_split": ["val"], "source_hospital": [1], "target_hospital": [0],
        "original_logit": [0.0], "within_logit": [0.2], "cross_logit": [0.5],
        "common_support": [True], "seed": [42], "backbone": ["synthetic"],
    })
    outputs = aggregate_audit(predictions, pd.Series([-1.0, 0.0, 1.0]), tmp_path, expected_pairs={(1, 0), (1, 3)})
    assert outputs["global_hcs_pair_weighted"].loc[0, "status"] == "unavailable"
    assert outputs["global_hce_pair_weighted"].loc[0, "status"] == "unavailable"
    assert outputs["global_hcs_common_support"].loc[0, "status"] == "unavailable"
    assert outputs["global_hce_common_support"].loc[0, "status"] == "unavailable"

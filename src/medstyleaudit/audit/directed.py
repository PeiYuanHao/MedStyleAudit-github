"""End-to-end audit table aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from medstyleaudit.utils.io import save_table
from medstyleaudit.statistics.bootstrap import crossed_multiplier_bootstrap
from .hce import common_support_hce, directed_hce, pair_weighted_hce
from .hcs import common_support_hcs, directed_hcs, pair_weighted_hcs
from .metrics import donor_pair_metrics, fit_logit_normalizer, source_metrics


def expected_pairs_for_split(config: dict, split: str) -> set[tuple[int, int]]:
    """Read the prespecified directed pair set without inferring absent pairs."""
    configured = config.get("audit", {}).get("expected_directed_pairs", {}).get(split)
    if not configured:
        raise ValueError(f"No expected directed hospital pairs configured for split={split}")
    pairs = {tuple(map(int, pair)) for pair in configured}
    if any(len(pair) != 2 or pair[0] == pair[1] for pair in pairs):
        raise ValueError(f"Invalid expected directed hospital pairs for split={split}: {configured}")
    return pairs


def _expected_targets(expected_pairs: set[tuple[int, int]] | None) -> dict[int, set[int]] | None:
    if expected_pairs is None:
        return None
    targets: dict[int, set[int]] = {}
    for source, target in expected_pairs:
        targets.setdefault(source, set()).add(target)
    return targets


def aggregate_audit(
    predictions: pd.DataFrame,
    id_logits: pd.Series,
    output_dir: str | Path,
    q_min: float = 1e-3,
    bootstrap_draws: int = 0,
    bootstrap_seed: int = 42,
    expected_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, pd.DataFrame]:
    normalizer = fit_logit_normalizer(id_logits.to_numpy(), q_min)
    sources = source_metrics(predictions, normalizer)
    pairs = donor_pair_metrics(predictions, normalizer)
    merge_keys = ["source_id", "source_split", "source_hospital", "target_hospital"]
    pairs = pairs.merge(sources[merge_keys + ["delta_within", "delta_cross", "HCS_source", "HCE_source"]], on=merge_keys, how="left", validate="many_to_one")
    outputs = {
        "audit_records": pairs,
        "source_metrics": sources,
        "directed_hcs": directed_hcs(sources),
        "directed_hce": directed_hce(sources),
    }
    interval_rows = []
    interval_keys = [column for column in ["source_split", "seed", "backbone", "source_hospital", "target_hospital"] if column in pairs]
    if bootstrap_draws > 0:
        cluster_columns = [column for column in ["source_physical_id", "within_physical_id", "cross_physical_id"] if column in pairs]
        if not cluster_columns:
            cluster_columns = ["source_id"]
        for keys, group in pairs.groupby(interval_keys, dropna=False):
            base = dict(zip(interval_keys, keys if isinstance(keys, tuple) else (keys,)))
            for metric, column in [("hcs", "HCS_pair"), ("hce", "HCE_pair")]:
                result = crossed_multiplier_bootstrap(group, column, cluster_columns, bootstrap_draws, bootstrap_seed)
                interval_rows.append({**base, "metric": metric, **result, "cluster_columns": ";".join(cluster_columns)})
    outputs["directed_intervals"] = pd.DataFrame(interval_rows)
    outputs["global_hcs_pair_weighted"] = pair_weighted_hcs(outputs["directed_hcs"], expected_pairs)
    targets = _expected_targets(expected_pairs)
    outputs["global_hcs_common_support"] = common_support_hcs(sources, targets)
    outputs["global_hce_pair_weighted"] = pair_weighted_hce(outputs["directed_hce"], expected_pairs)
    outputs["global_hce_common_support"] = common_support_hce(sources, targets)
    directory = Path(output_dir)
    for name, table in outputs.items():
        save_table(table, directory / f"{name}.csv")
    return outputs

"""Structured status ledgers for controls that are not end-to-end complete."""

from __future__ import annotations


def context_randomized_stages() -> list[dict[str, str]]:
    return [
        {"stage": "assignment_quota_preparation", "status": "completed", "reason": "fixed joint-cell quotas written"},
        {"stage": "context_randomized_training", "status": "implemented", "reason": "executed by scripts/07_run_controls.py"},
        {"stage": "trained_model_audit", "status": "implemented", "reason": "uses the frozen primary HCS/HCE audit"},
    ]


def planted_shortcut_stages(assignments_prepared: bool) -> list[dict[str, str]]:
    first = {
        "stage": "cue_generator_and_assignment_preparation" if assignments_prepared else "cue_generator",
        "status": "completed" if assignments_prepared else "implemented",
        "reason": "cue utilities and permuted assignment ledger are available" if assignments_prepared else "peripheral corner-cue library function is available",
    }
    rows = [first]
    if not assignments_prepared:
        rows.append({"stage": "cue_assignment_preparation", "status": "unavailable", "reason": "--assignments was not provided"})
    rows.extend([
        {"stage": "cue_conditioned_training", "status": "implemented", "reason": "runs every frozen rho independently"},
        {"stage": "classifier_audit", "status": "implemented", "reason": "uses the frozen primary HCS/HCE audit"},
        {"stage": "hcs_rho_aggregation", "status": "implemented", "reason": "does not assume monotonicity"},
    ])
    return rows

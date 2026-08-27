"""Structured status ledgers for controls that are not end-to-end complete."""

from __future__ import annotations


def context_randomized_stages() -> list[dict[str, str]]:
    return [
        {"stage": "assignment_quota_preparation", "status": "completed", "reason": "fixed joint-cell quotas written"},
        {"stage": "context_randomized_training", "status": "not_implemented", "reason": "behavioral-control training loop is not implemented"},
        {"stage": "trained_model_audit", "status": "not_implemented", "reason": "requires a trained context-randomized model"},
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
        {"stage": "cue_conditioned_training", "status": "not_implemented", "reason": "rho-conditioned classifier training is not implemented"},
        {"stage": "classifier_audit", "status": "not_implemented", "reason": "requires trained cue-conditioned classifiers"},
        {"stage": "hcs_rho_aggregation", "status": "not_implemented", "reason": "requires completed training and audits"},
    ])
    return rows

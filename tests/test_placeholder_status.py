from medstyleaudit.controls.status import context_randomized_stages, planted_shortcut_stages
from medstyleaudit.matching.ladder import advanced_level_status


def test_advanced_ladder_levels_never_look_like_empty_successes():
    unavailable = advanced_level_status(5, ["a"], ["lesion_feature"])
    assert unavailable["status"] == "unavailable" and unavailable["missing_columns"] == ["lesion_feature"]
    pending = advanced_level_status(6, ["appearance_feature"], ["appearance_feature"])
    assert pending["status"] == "not_implemented"


def test_partial_controls_mark_training_and_audit_not_implemented():
    context = {row["stage"]: row["status"] for row in context_randomized_stages()}
    planted = {row["stage"]: row["status"] for row in planted_shortcut_stages(False)}
    assert context["assignment_quota_preparation"] == "completed"
    assert context["context_randomized_training"] == "not_implemented"
    assert context["trained_model_audit"] == "not_implemented"
    assert planted["cue_generator"] == "implemented"
    assert planted["cue_conditioned_training"] == "not_implemented"

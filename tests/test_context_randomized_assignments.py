import pandas as pd

from medstyleaudit.controls.context_randomized import independent_context_assignments


def test_context_assignment_is_deterministic_capped_and_not_label_matched():
    frame = pd.DataFrame({
        "source_id": range(8), "label": [0, 0, 0, 0, 1, 1, 1, 1], "hospital_id": [0, 3] * 4,
        "physical_id": [f"p{i}" for i in range(8)], "slide_id": [f"s{i // 2}" for i in range(8)],
    })
    first, diagnostics = independent_context_assignments(frame, frame, seed=11, donor_reuse_cap=2)
    second, _ = independent_context_assignments(frame, frame, seed=11, donor_reuse_cap=2)
    pd.testing.assert_frame_equal(first, second)
    assert first["donor_source_id"].value_counts().max() <= 2
    assert (first["source_id"] != first["donor_source_id"]).all()
    assert (first["source_label"] != first["donor_label"]).any()
    assert set(diagnostics["source_label"]) == {0, 1}

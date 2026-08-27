import numpy as np

from medstyleaudit.data.lesion_mapping import lesion_features, project_polygons, validate_alignment


def test_polygon_projection_and_peripheral_features():
    polygon = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
    mask = project_polygons([polygon], (0, 0), 96, 1)
    features = lesion_features(mask)
    assert features["peripheral_tumor_presence"] == 1
    assert features["peripheral_tumor_fraction"] > 0


def test_alignment_reports_failure_instead_of_fabricating():
    mask = np.zeros((96, 96), bool)
    report = validate_alignment([(mask, 1)])
    assert report["status"] == "failed"

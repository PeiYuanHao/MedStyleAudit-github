from pathlib import Path

import pytest

from medstyleaudit.utils.paths import configured_output, experiment_path


def test_experiment_path_requires_external_root(monkeypatch):
    monkeypatch.delenv("MEDSTYLE_OUTPUT_ROOT", raising=False)
    with pytest.raises(EnvironmentError):
        experiment_path("metrics")


def test_configured_output_resolves_server_variable(monkeypatch, tmp_path):
    root = tmp_path.resolve()
    monkeypatch.setenv("MEDSTYLE_OUTPUT_ROOT", str(root))
    result = configured_output({"output_dir": "${MEDSTYLE_OUTPUT_ROOT}/matching"}, "unused")
    assert result == root / "matching"

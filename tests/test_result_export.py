import json
from pathlib import Path

from medstyleaudit.utils.io import save_json


def test_result_export_policy_is_covered_by_script_import(tmp_path):
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "13_export_results.py"
    spec = importlib.util.spec_from_file_location("result_export", script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source, repository = tmp_path / "external", tmp_path / "repo"
    source.mkdir(); repository.mkdir()
    save_json({"metric": 1}, source / "summary.json")
    (source / "best.ckpt").write_bytes(b"weights")
    destination = module.export_results(source, repository, "test-run", 1)
    assert (destination / "summary.json").is_file()
    assert not (destination / "best.ckpt").exists()
    manifest = json.loads((destination / "export_manifest.json").read_text(encoding="utf-8"))
    assert any(row["path"] == "best.ckpt" for row in manifest["skipped"])

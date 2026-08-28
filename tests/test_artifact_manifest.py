import json

import pytest

from medstyleaudit.artifacts.checksums import verify_sha256
from medstyleaudit.artifacts.manifest import ArtifactManifest, ManifestEntry, validate_hf_path
from medstyleaudit.artifacts.huggingface import repository_id


def test_manifest_generation_and_checksum_validation(tmp_path):
    artifact = tmp_path / "result.csv"; artifact.write_text("a,b\n1,2\n", encoding="utf-8")
    entry = ManifestEntry.from_file(artifact, "aggregate/final/MAIN_RESULTS.csv", experiment="final", seed=11)
    manifest = ArtifactManifest([entry]); manifest.write(tmp_path / "MANIFEST.json")
    loaded = ArtifactManifest.read(tmp_path / "MANIFEST.json")
    loaded.verify_local()
    data = json.loads((tmp_path / "MANIFEST.json").read_text(encoding="utf-8"))
    assert data["artifacts"][0]["seed"] == 11
    artifact.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_sha256(artifact, entry.sha256)


def test_manifest_rejects_raw_medical_data_paths():
    with pytest.raises(ValueError, match="Raw/source"):
        validate_hf_path("p0/raw/camelyon17/image.png")


def test_exact_hugging_face_namespace_is_the_default(monkeypatch):
    monkeypatch.delenv("MEDSTYLE_HF_REPO", raising=False)
    assert repository_id() == "PeiyuanHao/MedStyleAudit-Experiments"

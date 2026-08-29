import importlib.util
import json
from pathlib import Path

import pytest


def _module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "autodl_unattended.py"
    spec = importlib.util.spec_from_file_location("autodl_unattended", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeCommands:
    def __init__(self, *, upload=0, verify=0, shutdown=0):
        self.calls = []
        self.upload = upload
        self.verify = verify
        self.shutdown = shutdown

    def __call__(self, command, cwd):
        self.calls.append(tuple(command))
        joined = " ".join(command)
        if "hf_upload_artifacts.py" in joined:
            return self.upload
        if "hf_verify_artifacts.py" in joined:
            return self.verify
        if command == ["/usr/bin/shutdown"]:
            return self.shutdown
        return 0

    def called(self, fragment):
        return any(fragment in " ".join(command) for command in self.calls)


def _output(tmp_path):
    root = tmp_path / "output"
    protocol = root / "protocol"
    protocol.mkdir(parents=True)
    (protocol / "protocol_sha256.txt").write_text("protocol-hash\n", encoding="utf-8")
    (protocol / "FINAL_PROTOCOL.yaml").write_text("status: frozen\n", encoding="utf-8")
    return root


def _finalize(tmp_path, *, suite_exit=0, upload=0, verify=0, github_failure=False):
    module = _module()
    output = _output(tmp_path)
    log = output / "logs" / "final_suite" / "autodl_wrapper.log"
    log.parent.mkdir(parents=True)
    log.write_text("redacted test log hf-secret github-secret\n", encoding="utf-8")
    commands = FakeCommands(upload=upload, verify=verify)
    github_calls = []

    def github(repo_root, export_dir, phase, timestamp):
        github_calls.append((repo_root, export_dir, phase, timestamp))
        if github_failure:
            raise RuntimeError("simulated GitHub push failure")
        return "results/hospital1-20260829T000000Z"

    status = module.finalize_unattended_run(
        repo_root=Path(__file__).resolve().parents[1],
        output_root=output,
        phase="hospital1",
        suite_exit_code=suite_exit,
        started_at="2026-08-29T00:00:00+00:00",
        environment={"HF_TOKEN": "hf-secret", "GITHUB_TOKEN": "github-secret"},
        command_runner=commands,
        github_pusher=github,
    )
    return module, output, commands, github_calls, status


@pytest.mark.parametrize("suite_exit", [0, 17])
def test_success_and_failure_both_attempt_backups_and_shutdown(tmp_path, suite_exit):
    _, _, commands, github_calls, status = _finalize(tmp_path, suite_exit=suite_exit)
    assert commands.called("hf_upload_artifacts.py")
    assert github_calls
    assert commands.called("/usr/bin/shutdown")
    assert status["suite_exit_code"] == suite_exit
    assert status["status"] == ("success" if suite_exit == 0 else "failed")


def test_hf_upload_failure_still_attempts_github_and_shutdown(tmp_path):
    _, _, commands, github_calls, status = _finalize(tmp_path, upload=1)
    assert not status["hf_upload_succeeded"]
    assert github_calls
    assert commands.called("/usr/bin/shutdown")


def test_github_push_failure_still_attempts_shutdown(tmp_path):
    _, _, commands, _, status = _finalize(tmp_path, github_failure=True)
    assert not status["github_export_succeeded"]
    assert commands.called("/usr/bin/shutdown")


def test_hf_verification_failure_still_attempts_shutdown(tmp_path):
    _, _, commands, github_calls, status = _finalize(tmp_path, verify=1)
    assert not status["hf_verification_succeeded"]
    assert github_calls
    assert commands.called("/usr/bin/shutdown")


def test_phase_commands_preserve_manual_final_test_unlock():
    module = _module()
    hospital1 = module.suite_command("hospital1", ["cuda:0", "cuda:1"], python="python")
    hospital2 = module.suite_command("hospital2", ["cuda:0", "cuda:1"], python="python")
    assert "--allow-final-test" not in hospital1
    assert "--resume" not in hospital1
    assert "--resume" in hospital2
    assert "--allow-final-test" in hospital2
    assert "15_unlock_final_test.py" not in " ".join(hospital1 + hospital2)
    wrapper = (Path(__file__).resolve().parents[1] / "scripts" / "autodl_run_final_suite.sh").read_text(
        encoding="utf-8"
    )
    assert "15_unlock_final_test.py" not in wrapper


def test_github_export_excludes_files_above_ten_mib(tmp_path):
    module = _module()
    output = _output(tmp_path)
    aggregate = output / "aggregate" / "hospital1"
    aggregate.mkdir(parents=True)
    large = aggregate / "too_large.csv"
    large.write_bytes(b"x" * (module.GITHUB_MAX_FILE_BYTES + 1))
    small = aggregate / "global_hcs.csv"
    small.write_text("metric,value\nhcs,0.1\n", encoding="utf-8")
    export, manifest = module.stage_github_export(output)
    assert (export / "aggregate" / "hospital1" / "global_hcs.csv").is_file()
    assert not (export / "aggregate" / "hospital1" / "too_large.csv").exists()
    skipped = {row["path"]: row["reason"] for row in manifest["skipped"]}
    assert "exceeds maximum" in skipped["aggregate/hospital1/too_large.csv"]


def test_secrets_are_not_written_to_status_or_export(tmp_path):
    _, output, _, _, _ = _finalize(tmp_path)
    status = json.loads((output / "protocol" / "autodl_final_status.json").read_text(encoding="utf-8"))
    assert "HF_TOKEN" not in status
    assert "GITHUB_TOKEN" not in status
    for path in (output / "github_export").rglob("*"):
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            assert "hf-secret" not in text
            assert "github-secret" not in text

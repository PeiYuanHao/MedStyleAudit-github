from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path, PurePosixPath

from huggingface_hub import (
    CommitOperationAdd,
    HfApi,
    hf_hub_download,
)

from medstyleaudit.artifacts.manifest import (
    ArtifactManifest,
    ManifestEntry,
    validate_hf_path,
)

ROOT = Path("/root/autodl-tmp/medstyleaudit-experiments")
REPO = "PeiyuanHao/MedStyleAudit-Experiments"
REPO_TYPE = "dataset"

STATE = Path("/root/autodl-tmp/hf_final_upload_state.json")
LOCAL_MANIFEST = ROOT / "MANIFEST.json"

EXPECTED_SELECTED_FILES = 675

MAX_BATCH_FILES = 20
MAX_BATCH_BYTES = 400 * 1024**2
BETWEEN_COMMITS_SECONDS = 8
MAX_RETRIES = 5

TOKEN = os.environ.get("HF_TOKEN")

if not TOKEN:
    raise RuntimeError("HF_TOKEN is not set")

if os.environ.get("MEDSTYLE_HF_REPO") != REPO:
    raise RuntimeError(
        f"MEDSTYLE_HF_REPO must be exactly {REPO}"
    )

api = HfApi(token=TOKEN)


# ============================================================
# Helpers
# ============================================================

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size

    h = hashlib.sha1()
    h.update(f"blob {size}\0".encode())

    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)

    return h.hexdigest()


def infer_metadata(remote: str, commit: str, protocol_hash: str | None):
    parts = PurePosixPath(remote).parts

    experiment = (
        parts[1]
        if len(parts) > 1
        and parts[0] in {"p0", "audits", "aggregate"}
        else parts[0]
    )

    backbone = next(
        (x for x in parts if x in {"resnet50", "densenet121"}),
        None,
    )

    seed_part = next(
        (x for x in parts if re.fullmatch(r"seed_\d+", x)),
        None,
    )

    seed = (
        int(seed_part.split("_", 1)[1])
        if seed_part
        else None
    )

    split = next(
        (
            x for x in parts
            if x in {
                "train",
                "id_val",
                "val",
                "test",
                "ood_val",
                "ood_test",
            }
        ),
        None,
    )

    return {
        "experiment": experiment,
        "backbone": backbone,
        "seed": seed,
        "split": split,
        "git_commit": commit,
        "protocol_hash": protocol_hash,
    }


def save_state(data):
    STATE.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_commit_with_retry(operations, message):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return api.create_commit(
                repo_id=REPO,
                repo_type=REPO_TYPE,
                operations=operations,
                commit_message=message,
            )

        except Exception as exc:
            print(
                f"UPLOAD ERROR attempt={attempt}/{MAX_RETRIES}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

            if attempt == MAX_RETRIES:
                raise

            delay = min(
                15 * (2 ** (attempt - 1)),
                240,
            )

            print(
                f"BACKOFF {delay}s",
                flush=True,
            )

            time.sleep(delay)


# ============================================================
# Auth/repo preflight
# ============================================================

print("=" * 72)
print("HF FINAL SAFE UPLOAD")
print("=" * 72)

who = api.whoami()
info = api.repo_info(
    REPO,
    repo_type=REPO_TYPE,
)

print("user    =", who["name"])
print("repo    =", info.id)
print("private =", info.private)

if not info.private:
    raise RuntimeError("HF repository is not private")


# ============================================================
# Exactly the approved artifact roots
# ============================================================

roots = [
    ROOT / "checkpoints",
    ROOT / "predictions",
    ROOT / "subsets",
    ROOT / "audits/hospital1",
    ROOT / "audits/hospital2",
    ROOT / "aggregate/hospital1",
    ROOT / "aggregate/hospital2",
    ROOT / "aggregate/final",
    ROOT / "protocol",
]

selected: list[tuple[Path, str]] = []

for source_root in roots:
    if not source_root.is_dir():
        raise RuntimeError(
            f"Required artifact root missing: {source_root}"
        )

    for path in sorted(source_root.rglob("*")):
        if not path.is_file():
            continue

        rel_parts = path.relative_to(ROOT).parts

        if path.name in {"MANIFEST.json", "last.ckpt"}:
            continue

        if any(part.startswith(".") for part in rel_parts):
            continue

        remote = path.relative_to(ROOT).as_posix()

        # Project's own source-data safety guard.
        remote = validate_hf_path(remote)

        selected.append((path, remote))


if len(selected) != EXPECTED_SELECTED_FILES:
    raise RuntimeError(
        f"Expected exactly {EXPECTED_SELECTED_FILES} approved files, "
        f"found {len(selected)}"
    )

total_bytes = sum(
    path.stat().st_size
    for path, _ in selected
)

print("approved files =", len(selected))
print("approved bytes =", total_bytes)
print(
    "approved GiB   =",
    f"{total_bytes / 1024**3:.3f}",
)

print("RAW/SOURCE DATA GUARD: PASS")


# ============================================================
# Provenance
# ============================================================

commit = subprocess.check_output(
    [
        "git",
        "-C",
        "/root/MedStyleAudit-github",
        "rev-parse",
        "HEAD",
    ],
    text=True,
).strip()

protocol_file = (
    ROOT
    / "protocol"
    / "protocol_sha256.txt"
)

protocol_hash = (
    protocol_file.read_text(
        encoding="utf-8"
    ).strip()
    if protocol_file.is_file()
    else None
)

print("git commit    =", commit)
print("protocol hash =", protocol_hash)


# ============================================================
# Load existing remote manifest
# ============================================================

remote_files_initial = set(
    api.list_repo_files(
        REPO,
        repo_type=REPO_TYPE,
    )
)

if "MANIFEST.json" in remote_files_initial:
    remote_manifest_path = hf_hub_download(
        repo_id=REPO,
        repo_type=REPO_TYPE,
        filename="MANIFEST.json",
        token=TOKEN,
    )

    remote_manifest = ArtifactManifest.read(
        remote_manifest_path
    )

    print(
        "existing manifest artifacts =",
        len(remote_manifest.entries),
    )
else:
    remote_manifest = ArtifactManifest()
    print("existing manifest artifacts = 0")


old = {
    entry.hf_path: entry
    for entry in remote_manifest.entries
}

merged = ArtifactManifest(
    remote_manifest.entries
)


# ============================================================
# Hash approved files
# ============================================================

entries = []
changed = []
unchanged = []

print()
print("HASHING 675 APPROVED FILES...", flush=True)

for index, (path, remote) in enumerate(selected, 1):

    metadata = infer_metadata(
        remote,
        commit,
        protocol_hash,
    )

    entry = ManifestEntry.from_file(
        path,
        remote,
        **metadata,
    )

    entries.append(entry)
    merged.add(entry)

    previous = old.get(remote)

    if (
        previous is not None
        and previous.sha256 == entry.sha256
        and previous.bytes == entry.bytes
    ):
        unchanged.append(entry)
    else:
        changed.append(entry)

    if index % 25 == 0 or index == len(selected):
        print(
            f"HASH {index}/{len(selected)}",
            flush=True,
        )


print()
print("UPLOAD PLAN")
print("unchanged =", len(unchanged))
print("changed   =", len(changed))
print(
    "changed GiB =",
    f"{sum(x.bytes for x in changed) / 1024**3:.3f}",
)


# ============================================================
# Resume state
# ============================================================

state = {
    "repo": REPO,
    "status": "running",
    "uploaded": {},
}

if STATE.is_file():
    try:
        previous_state = json.loads(
            STATE.read_text(encoding="utf-8")
        )

        if previous_state.get("repo") == REPO:
            state["uploaded"] = previous_state.get(
                "uploaded",
                {},
            )

            print(
                "resume records =",
                len(state["uploaded"]),
            )
    except Exception:
        pass


pending = []

for entry in changed:
    remembered = state["uploaded"].get(
        entry.hf_path
    )

    if (
        remembered
        and remembered.get("sha256") == entry.sha256
        and remembered.get("bytes") == entry.bytes
    ):
        print(
            "RESUME SKIP:",
            entry.hf_path,
            flush=True,
        )
        continue

    pending.append(entry)


print("pending uploads =", len(pending))


# ============================================================
# Build safe batches
# ============================================================

batches = []
current = []
current_bytes = 0

for entry in pending:
    if (
        current
        and (
            len(current) >= MAX_BATCH_FILES
            or current_bytes + entry.bytes > MAX_BATCH_BYTES
        )
    ):
        batches.append(current)
        current = []
        current_bytes = 0

    current.append(entry)
    current_bytes += entry.bytes

if current:
    batches.append(current)


print("upload batches =", len(batches))


# ============================================================
# Upload changed artifacts
# ============================================================

for batch_index, batch in enumerate(batches, 1):

    batch_bytes = sum(
        x.bytes for x in batch
    )

    print()
    print(
        f"UPLOAD BATCH {batch_index}/{len(batches)} "
        f"| files={len(batch)} "
        f"| MiB={batch_bytes / 1024**2:.2f}",
        flush=True,
    )

    operations = [
        CommitOperationAdd(
            path_in_repo=entry.hf_path,
            path_or_fileobj=entry.local_path,
        )
        for entry in batch
    ]

    create_commit_with_retry(
        operations,
        (
            f"Final experiment artifacts "
            f"{batch_index}/{len(batches)}"
        ),
    )

    for entry in batch:
        state["uploaded"][entry.hf_path] = {
            "sha256": entry.sha256,
            "bytes": entry.bytes,
        }

    save_state(state)

    print(
        f"UPLOAD BATCH {batch_index}/{len(batches)} PASS",
        flush=True,
    )

    if batch_index != len(batches):
        time.sleep(BETWEEN_COMMITS_SECONDS)


# ============================================================
# Manifest LAST
# ============================================================

print()
print("WRITING MERGED MANIFEST...", flush=True)

merged.write(
    LOCAL_MANIFEST
)

manifest_op = CommitOperationAdd(
    path_in_repo="MANIFEST.json",
    path_or_fileobj=str(LOCAL_MANIFEST),
)

create_commit_with_retry(
    [manifest_op],
    "Publish final verified artifact manifest",
)

print(
    "MANIFEST LAST UPLOAD: PASS",
    flush=True,
)


# ============================================================
# Remote verification
#
# LFS:
#   compare remote SHA256
#
# ordinary Git object:
#   compare exact Git blob SHA1 to local content
#
# This avoids downloading 2.8 GiB again.
# ============================================================

print()
print("REMOTE VERIFY START...", flush=True)

remote_objects = {}

for obj in api.list_repo_tree(
    REPO,
    repo_type=REPO_TYPE,
    recursive=True,
    expand=True,
):
    path = getattr(obj, "path", None)
    size = getattr(obj, "size", None)

    if path is not None and size is not None:
        remote_objects[path] = obj


verified = 0

for index, entry in enumerate(entries, 1):

    obj = remote_objects.get(
        entry.hf_path
    )

    if obj is None:
        raise RuntimeError(
            f"REMOTE MISSING: {entry.hf_path}"
        )

    if int(obj.size) != int(entry.bytes):
        raise RuntimeError(
            f"REMOTE SIZE MISMATCH: {entry.hf_path} "
            f"local={entry.bytes} remote={obj.size}"
        )

    lfs = getattr(obj, "lfs", None)

    remote_lfs_sha = None

    if isinstance(lfs, dict):
        remote_lfs_sha = lfs.get("sha256")
    elif lfs is not None:
        remote_lfs_sha = getattr(
            lfs,
            "sha256",
            None,
        )

    if remote_lfs_sha:
        if remote_lfs_sha != entry.sha256:
            raise RuntimeError(
                f"REMOTE SHA256 MISMATCH: {entry.hf_path}"
            )

    else:
        blob_id = getattr(
            obj,
            "blob_id",
            None,
        )

        local_blob = git_blob_sha1(
            Path(entry.local_path)
        )

        if blob_id != local_blob:
            raise RuntimeError(
                f"REMOTE GIT BLOB MISMATCH: {entry.hf_path}"
            )

    verified += 1

    if index % 25 == 0 or index == len(entries):
        print(
            f"VERIFY {index}/{len(entries)}",
            flush=True,
        )


# Verify remote manifest itself
remote_manifest_path = hf_hub_download(
    repo_id=REPO,
    repo_type=REPO_TYPE,
    filename="MANIFEST.json",
    token=TOKEN,
    force_download=True,
)

published = ArtifactManifest.read(
    remote_manifest_path
)

published_by_path = {
    x.hf_path: x
    for x in published.entries
}

for entry in entries:
    remote_entry = published_by_path.get(
        entry.hf_path
    )

    if remote_entry is None:
        raise RuntimeError(
            f"MANIFEST MISSING ENTRY: {entry.hf_path}"
        )

    if (
        remote_entry.sha256 != entry.sha256
        or remote_entry.bytes != entry.bytes
    ):
        raise RuntimeError(
            f"MANIFEST MISMATCH: {entry.hf_path}"
        )


state["status"] = "completed"
state["verified_files"] = verified
state["manifest_artifact_count"] = len(
    published.entries
)

save_state(state)


print()
print("=" * 72)
print("HF FINAL BACKUP: PASS")
print(f"selected files verified = {verified}")
print(
    "remote manifest artifacts =",
    len(published.entries),
)
print("repo =", REPO)
print("=" * 72)

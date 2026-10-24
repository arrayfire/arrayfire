"""Avoid repeated fixture downloads during a SuiteSparse collection outage."""

import hashlib
import io
import json
import pathlib
import re
import sys
import tarfile
import urllib.request

COLLECTION = "https://sparse.tamu.edu/"


def restore_bundle(repo, cache, expected, metadata):
    archive = repo / ".github/fixtures/matrixmarket.tar.gz"
    if not archive.exists():
        return False
    if metadata is None:
        raise RuntimeError("Bundled matrix fixtures have no integrity manifest")
    blob = archive.read_bytes()
    if hashlib.sha256(blob).hexdigest() != metadata["archive_sha256"]:
        raise RuntimeError("Bundled matrix archive checksum mismatch")
    payloads = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as bundle:
        members = bundle.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != expected:
            raise RuntimeError("Bundled matrix archive contains unexpected paths")
        for member in members:
            spec = metadata["files"][member.name]
            total += member.size
            if not member.isfile() or member.size != spec["bytes"] or total > 64 * 1024 * 1024:
                raise RuntimeError("Invalid bundled matrix archive member")
            payload = bundle.extractfile(member).read()
            if hashlib.sha256(payload).hexdigest() != spec["sha256"]:
                raise RuntimeError("Bundled matrix file checksum mismatch")
            payloads[member.name] = payload
    for relative, payload in payloads.items():
        target = cache / relative
        if not target.resolve().is_relative_to(cache.resolve()):
            raise RuntimeError("Matrix cache path escapes its root")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".mtx.tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)
    return True


def check_cache(repo, cache, fetch=None):
    manifest = repo / "test/CMakeModules/download_sparse_datasets.cmake"
    matrices = re.findall(
        r'^\s*mtxDownload\("([^"/]+)"\s+"([^"/]+)"\)',
        manifest.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not matrices:
        raise RuntimeError("No matrix fixtures found in the CMake manifest")
    expected = {f"{group}/{name}/{name}.mtx" for name, group in matrices}
    manifest = repo / ".github/fixtures/matrixmarket.json"
    metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None
    if metadata is not None and set(metadata["files"]) != expected:
        raise RuntimeError("Bundled matrix manifest does not match the CMake fixture list")
    missing = []
    for name, group in matrices:
        matrix = cache / group / name / (name + ".mtx")
        if (not matrix.is_file() or matrix.stat().st_size == 0
                or (metadata is not None and hashlib.sha256(matrix.read_bytes()).hexdigest()
                    != metadata["files"][f"{group}/{name}/{name}.mtx"]["sha256"])):
            missing.append(group + "/" + name)
    if not missing:
        return "All matrix fixtures are cached; no provider request needed."
    if restore_bundle(repo, cache, expected, metadata):
        return "Restored verified bundled matrix fixtures; no provider request needed."

    if fetch is None:
        def fetch():
            with urllib.request.urlopen(COLLECTION, timeout=20) as response:
                return response.read(256 * 1024).decode("utf-8", errors="replace")
    if "COLLECTION IS OFFLINE" in fetch().upper():
        raise RuntimeError(
            "SuiteSparse reports that the collection is offline. "
            "The matrix fixture cache is incomplete. Restore a complete cache "
            "or rerun after the provider recovers; matrix tests remain enabled. "
            + COLLECTION
        )
    return "Fixture cache is incomplete; the provider reports no collection outage."


if __name__ == "__main__":
    repo = pathlib.Path(__file__).resolve().parents[2]
    try:
        print(check_cache(repo, repo / "build/extern/matrixmarket"))
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)

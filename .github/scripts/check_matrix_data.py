"""Avoid repeated fixture downloads during a SuiteSparse collection outage."""

import pathlib
import re
import sys
import urllib.request

COLLECTION = "https://sparse.tamu.edu/"


def check_cache(repo, cache, fetch=None):
    manifest = repo / "test/CMakeModules/download_sparse_datasets.cmake"
    matrices = re.findall(
        r'^\s*mtxDownload\("([^"/]+)"\s+"([^"/]+)"\)',
        manifest.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if not matrices:
        raise RuntimeError("No matrix fixtures found in the CMake manifest")
    missing = []
    for name, group in matrices:
        matrix = cache / group / name / (name + ".mtx")
        if not matrix.is_file() or matrix.stat().st_size == 0:
            missing.append(group + "/" + name)
    if not missing:
        return "All matrix fixtures are cached; no provider request needed."

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

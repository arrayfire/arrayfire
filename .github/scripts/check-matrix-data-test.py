#!/usr/bin/env python3
"""Tests of check-matrix-data.py: it restores the bundled Matrix Market data, fails naming the
file when the bundle, a matrix or the tests' list of matrices differs from the manifest, and the
CI configurations point FetchContent at every bundled matrix.
"""
import gzip
import io
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
SCRIPTS = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
SCRIPT = SCRIPTS / "check-matrix-data.py"
FIXTURES = ROOT / ".github/fixtures"


def run(fixtures, source, build):
    process = subprocess.run([sys.executable, "-I", str(SCRIPT), "--fixtures", str(fixtures),
                              str(source), str(build)], capture_output=True, text=True, timeout=120)
    return process.returncode, process.stdout + process.stderr


def copy_fixtures(tmp):
    fixtures = tmp / "fixtures"
    shutil.copytree(FIXTURES, fixtures)
    return fixtures


def repack(fixtures, change):
    """Rewrites the bundle with change(members) applied to {name: bytes}, keeping the manifest."""
    archive = fixtures / "matrixmarket.tar.gz"
    with tarfile.open(archive, "r:gz") as bundle:
        members = {m.name: bundle.extractfile(m).read() for m in bundle.getmembers()}
    change(members)
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w") as bundle:
        for name, content in sorted(members.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            bundle.addfile(info, io.BytesIO(content))
    archive.write_bytes(gzip.compress(data.getvalue(), mtime=0))


def flip_matrix_byte(members):
    content = bytearray(members["HB/bcsstm02/bcsstm02.mtx"])
    content[-2] ^= 0x01
    members["HB/bcsstm02/bcsstm02.mtx"] = bytes(content)


def flip_archive_byte(fixtures):
    archive = fixtures / "matrixmarket.tar.gz"
    blob = bytearray(archive.read_bytes())
    blob[len(blob) // 2] ^= 0x01
    archive.write_bytes(bytes(blob))


def source_listing(tmp, extra=""):
    """A source tree whose tests download the repository's matrices, plus extra lines."""
    source = tmp / "source"
    module = source / "test/CMakeModules/download_sparse_datasets.cmake"
    module.parent.mkdir(parents=True)
    module.write_text((ROOT / "test/CMakeModules/download_sparse_datasets.cmake").read_text() + extra)
    return source


def case_restores(tmp):
    code, output = run(FIXTURES, ROOT, tmp / "build")
    restored = sorted(str(p.relative_to(tmp / "build/extern/matrixmarket"))
                      for p in (tmp / "build/extern/matrixmarket").rglob("*.mtx"))
    again_code, again = run(FIXTURES, ROOT, tmp / "build")
    ok = (code == 0 and len(restored) == 9 and "9 restored" in output
          and again_code == 0 and "0 restored, 9 already in place" in again)
    return ok, f"{output.strip()} | rerun: {again.strip()}"


def case_matrix_byte(tmp):
    fixtures = copy_fixtures(tmp)
    repack(fixtures, flip_matrix_byte)
    code, output = run(fixtures, ROOT, tmp / "build")
    ok = (code == 1 and "HB/bcsstm02/bcsstm02.mtx: 1884 bytes with SHA-256" in output
          and "matrixmarket.tar.gz: SHA-256" in output and not (tmp / "build").exists())
    return ok, output.strip()


def case_archive_byte(tmp):
    fixtures = copy_fixtures(tmp)
    flip_archive_byte(fixtures)
    code, output = run(fixtures, ROOT, tmp / "build")
    ok = code == 1 and "matrixmarket.tar.gz:" in output and not (tmp / "build").exists()
    return ok, output.strip()


def case_missing_matrix(tmp):
    fixtures = copy_fixtures(tmp)
    repack(fixtures, lambda members: members.pop("QCD/conf6_0-4x4-30/conf6_0-4x4-30.mtx"))
    code, output = run(fixtures, ROOT, tmp / "build")
    ok = code == 1 and "QCD/conf6_0-4x4-30/conf6_0-4x4-30.mtx: matrixmarket.json lists it, but" in output
    return ok, output.strip()


def case_new_test_matrix(tmp):
    source = source_listing(tmp, 'mtxDownload("Si2" "PARSEC")\n')
    code, output = run(FIXTURES, source, tmp / "build")
    ok = code == 1 and "PARSEC/Si2/Si2.mtx: the tests use it, but matrixmarket.json doesn't list it" in output
    return ok, output.strip()


def case_configurations(tmp):
    """The manifest, the tests' list, matrixmarket.cmake and the ci preset name the same matrices."""
    manifest = {path.rsplit("/", 1)[0] for path in
                json.loads((FIXTURES / "matrixmarket.json").read_text())["files"]}
    tests = {f"{group}/{name}" for name, group in re.findall(
        r'^\s*mtxDownload\("([^"/]+)"\s+"([^"/]+)"\)',
        (ROOT / "test/CMakeModules/download_sparse_datasets.cmake").read_text(), re.MULTILINE)}
    cache = set(re.search(r"foreach\(matrix\s+([^)]*)\)", (SCRIPTS / "matrixmarket.cmake").read_text())
                .group(1).split())
    preset = next(p for p in json.loads((ROOT / "CMakePresets.json").read_text())["configurePresets"]
                  if p["name"] == "ci")["cacheVariables"]
    overrides = {key: value for key, value in preset.items()
                 if key.startswith("FETCHCONTENT_SOURCE_DIR_MTXDOWNLOAD_")}
    expected = {"FETCHCONTENT_SOURCE_DIR_" + ("MTXDOWNLOAD_" + m.replace("/", "_")).upper():
                "${sourceDir}/build/${presetName}/extern/matrixmarket/" + m for m in manifest}
    ok = (len(manifest) == 9 and manifest == tests == cache and overrides == expected
          and preset.get("AF_TEST_WITH_MTX_FILES") == "ON")
    return ok, (f"manifest {len(manifest)}, tests {len(tests)}, matrixmarket.cmake {len(cache)}, "
                f"ci preset {len(overrides)} matrices; AF_TEST_WITH_MTX_FILES={preset.get('AF_TEST_WITH_MTX_FILES')}")


CASES = [
    ("the bundle restores its 9 matrices, and a second run finds them in place", case_restores),
    ("one changed byte in a matrix fails, naming the matrix and the archive", case_matrix_byte),
    ("one changed byte in the compressed archive fails, naming the archive", case_archive_byte),
    ("a matrix missing from the archive fails, naming it", case_missing_matrix),
    ("a matrix the tests use but the bundle lacks fails, naming it", case_new_test_matrix),
    ("the manifest, the tests, matrixmarket.cmake and the ci preset agree", case_configurations),
]


def main():
    results = []
    for name, case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            ok, detail = case(pathlib.Path(tmp))
        results.append(ok)
        print(f"{'ok  ' if ok else 'FAIL'} {name}\n     " + detail.replace("\n", "\n     "))
    print(f"{sum(results)} of {len(results)} cases match")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())

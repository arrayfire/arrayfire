#!/usr/bin/env python3
"""Tests of check-matrix-data.py: with no arrayfire-data commit pinned it turns the Matrix Market
tests off; with one it downloads, checks and restores the matrices, and fails naming the file
when a download or a check fails; and the CI configurations agree with the manifest.

Most cases serve small made-up matrices from a temporary directory (file://). With
AF_CI_ARRAYFIRE_DATA naming a local arrayfire-data checkout, the case for a correct source uses
the repository's manifest and that checkout's matrixmarket/ instead.
"""
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
SCRIPTS = pathlib.Path(__file__).resolve().parent
ROOT = SCRIPTS.parents[1]
SCRIPT = SCRIPTS / "check-matrix-data.py"
FIXTURES = ROOT / ".github/fixtures"
COMMIT = "0123456789abcdef0123456789abcdef01234567"
MATRICES = {
    ("tiny", "HB"): "%%MatrixMarket matrix coordinate real general\n% id: 1\n2 2 1\n1 1 1.5\n",
    ("small", "QCD"): "%%MatrixMarket matrix coordinate complex general\n% id: 2\n2 2 2\n1 1 1 0\n2 2 0 1\n",
}
spec = importlib.util.spec_from_file_location("check_matrix_data", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def path(name, group):
    return f"{group}/{name}/{name}.mtx"


def run(fixtures, tree, build, *options):
    process = subprocess.run([sys.executable, "-I", str(SCRIPT), "--fixtures", str(fixtures), *options,
                              str(tree), str(build)], capture_output=True, text=True, timeout=120)
    return process.returncode, process.stdout, process.stderr


def made_up(tmp, commit=COMMIT):
    """Fixtures pinning commit for MATRICES, the matrices served under tmp/served, and a source
    tree whose tests use them; returns (fixtures, tree, the --source URL)."""
    fixtures, served, tree = tmp / "fixtures", tmp / "served", tmp / "tree"
    files = {}
    for (name, group), text in MATRICES.items():
        data = text.encode()
        files[path(name, group)] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        (served / group / name).mkdir(parents=True)
        (served / path(name, group)).write_bytes(data)
    fixtures.mkdir()
    (fixtures / "matrixmarket.json").write_text(json.dumps(
        {"source": {"repository": "arrayfire/arrayfire-data", "path": "matrixmarket/", "commit": commit},
         "files": files}))
    module = tree / "test/CMakeModules/download_sparse_datasets.cmake"
    module.parent.mkdir(parents=True)
    module.write_text("".join(f'mtxDownload("{name}" "{group}")\n' for name, group in MATRICES))
    return fixtures, tree, served.as_uri() + "/"


def case_unpinned(tmp):
    """The repository's manifest with its commit emptied: OFF, one line, nothing restored."""
    fixtures = tmp / "fixtures"
    shutil.copytree(FIXTURES, fixtures)
    manifest = json.loads((fixtures / "matrixmarket.json").read_text())
    manifest["source"]["commit"] = ""
    (fixtures / "matrixmarket.json").write_text(json.dumps(manifest))
    code, out, err = run(fixtures, ROOT, tmp / "build")
    ok = (code == 0 and out == "OFF\n" and not (tmp / "build").exists()
          and err == "Matrix Market data: no arrayfire-data commit pinned yet; the Matrix Market tests are off\n")
    return ok, f"exit {code}, stdout {out!r}, stderr: {err.strip()}"


def case_correct(tmp):
    """A correct source: ON, every matrix restored, and a rerun finds them in place."""
    local = os.environ.get("AF_CI_ARRAYFIRE_DATA")
    if local:
        fixtures, tree = FIXTURES, ROOT
        source = (pathlib.Path(local).resolve() / "matrixmarket").as_uri() + "/"
        count = len(json.loads((FIXTURES / "matrixmarket.json").read_text())["files"])
    else:
        fixtures, tree, source = made_up(tmp)
        count = len(MATRICES)
    code, out, err = run(fixtures, tree, tmp / "build", "--source", source)
    restored = sorted((tmp / "build/extern/matrixmarket").rglob("*.mtx"))
    again_code, again_out, again = run(fixtures, tree, tmp / "build", "--source", source)
    ok = (code == 0 and out == "ON\n" and len(restored) == count
          and f"{count} matrices match matrixmarket.json ({count} downloaded from {source}, 0 from the cache); "
              f"{count} restored, 0 already in place" in err
          and again_code == 0 and again_out == "ON\n" and f"0 restored, {count} already in place" in again)
    where = f"the arrayfire-data checkout {local}" if local else "made-up matrices"
    return ok, f"{where}: exit {code}, stdout {out!r}, {err.strip()} | rerun: {again.strip()}"


def case_pinned_url(tmp):
    """A pinned commit means raw.githubusercontent.com at that commit; nothing is downloaded here."""
    url = check.pinned_url({"repository": "arrayfire/arrayfire-data", "path": "matrixmarket/", "commit": COMMIT})
    expected = f"https://raw.githubusercontent.com/arrayfire/arrayfire-data/{COMMIT}/matrixmarket/"
    return url == expected, url


def case_branch_name(tmp):
    fixtures, tree, source = made_up(tmp, commit="master")
    code, out, err = run(fixtures, tree, tmp / "build")
    ok = code == 1 and out == "" and "the commit must be a full 40-character SHA, not 'master'" in err
    return ok, f"exit {code}: {err.strip()}"


def case_changed_byte(tmp):
    fixtures, tree, source = made_up(tmp)
    served = tmp / "served" / path("tiny", "HB")
    data = bytearray(served.read_bytes())
    data[-2] ^= 0x01
    served.write_bytes(bytes(data))
    code, out, err = run(fixtures, tree, tmp / "build", "--source", source)
    ok = (code == 1 and out == "" and not (tmp / "build").exists()
          and f"HB/tiny/tiny.mtx: {len(data)} bytes with SHA-256 {hashlib.sha256(data).hexdigest()} from "
              f"{source}HB/tiny/tiny.mtx, but matrixmarket.json says {len(data)} bytes with" in err
          and "QCD/small" not in err)
    return ok, f"exit {code}: {err.strip()}"


def case_wrong_size(tmp):
    fixtures, tree, source = made_up(tmp)
    served = tmp / "served" / path("small", "QCD")
    size = served.stat().st_size
    served.write_bytes(served.read_bytes() + b"3 3 1 0\n")
    code, out, err = run(fixtures, tree, tmp / "build", "--source", source)
    ok = (code == 1 and out == "" and not (tmp / "build").exists()
          and f"QCD/small/small.mtx: more than {size} bytes with SHA-256" in err
          and f"but matrixmarket.json says {size} bytes" in err)
    return ok, f"exit {code}: {err.strip()}"


def case_missing_file(tmp):
    fixtures, tree, source = made_up(tmp)
    (tmp / "served" / path("tiny", "HB")).unlink()
    code, out, err = run(fixtures, tree, tmp / "build", "--source", source)
    ok = (code == 1 and out == "" and not (tmp / "build").exists()
          and f"HB/tiny/tiny.mtx: downloading {source}HB/tiny/tiny.mtx failed (" in err)
    return ok, f"exit {code}: {err.strip()}"


def case_unreachable(tmp):
    """A server that refuses the connection (the discard port): both files named, no retry."""
    fixtures, tree, source = made_up(tmp)
    code, out, err = run(fixtures, tree, tmp / "build", "--source", "http://127.0.0.1:9/matrixmarket/")
    ok = (code == 1 and out == "" and not (tmp / "build").exists()
          and "HB/tiny/tiny.mtx: downloading http://127.0.0.1:9/matrixmarket/HB/tiny/tiny.mtx failed (" in err
          and "QCD/small/small.mtx: downloading http://127.0.0.1:9/matrixmarket/QCD/small/small.mtx failed (" in err)
    return ok, f"exit {code}: {err.strip()}"


def case_cache(tmp):
    """Verified downloads are kept; a later run uses them, and downloads again one that changed."""
    fixtures, tree, source = made_up(tmp)
    cache = tmp / "cache"
    first = run(fixtures, tree, tmp / "build1", "--source", source, "--cache", str(cache))
    shutil.move(tmp / "served", tmp / "elsewhere")
    second = run(fixtures, tree, tmp / "build2", "--source", source, "--cache", str(cache))
    shutil.move(tmp / "elsewhere", tmp / "served")
    (cache / path("tiny", "HB")).write_text("not the matrix\n")
    third = run(fixtures, tree, tmp / "build3", "--source", source, "--cache", str(cache))
    ok = (first[0] == 0 and "(2 downloaded from" in first[2]
          and second[:2] == (0, "ON\n") and "(0 downloaded from" in second[2] and "2 from the cache" in second[2]
          and third[:2] == (0, "ON\n") and "(1 downloaded from" in third[2] and "1 from the cache" in third[2]
          and (cache / path("tiny", "HB")).read_text() == MATRICES[("tiny", "HB")])
    return ok, " | ".join(result[2].strip() for result in (first, second, third))


def case_new_test_matrix(tmp):
    fixtures, tree, source = made_up(tmp)
    with open(tree / "test/CMakeModules/download_sparse_datasets.cmake", "a") as module:
        module.write('mtxDownload("Si2" "PARSEC")\n')
    code, out, err = run(fixtures, tree, tmp / "build", "--source", source)
    ok = code == 1 and "PARSEC/Si2/Si2.mtx: the tests use it, but matrixmarket.json doesn't list it" in err
    return ok, f"exit {code}: {err.strip()}"


def case_configurations(tmp):
    """The manifest pins arrayfire-data's matrixmarket/, and it, the tests' list,
    matrixmarket.cmake and the ci preset name the same matrices and take the switch from ci.sh."""
    manifest = json.loads((FIXTURES / "matrixmarket.json").read_text())
    listed = {name.rsplit("/", 1)[0] for name in manifest["files"]}
    tests = {f"{group}/{name}" for name, group in re.findall(
        r'^\s*mtxDownload\("([^"/]+)"\s+"([^"/]+)"\)',
        (ROOT / "test/CMakeModules/download_sparse_datasets.cmake").read_text(), re.MULTILINE)}
    text = (SCRIPTS / "matrixmarket.cmake").read_text()
    cache = set(re.search(r"foreach\(matrix\s+([^)]*)\)", text).group(1).split())
    preset = next(p for p in json.loads((ROOT / "CMakePresets.json").read_text())["configurePresets"]
                  if p["name"] == "ci")["cacheVariables"]
    overrides = {key: value for key, value in preset.items()
                 if key.startswith("FETCHCONTENT_SOURCE_DIR_MTXDOWNLOAD_")}
    expected = {"FETCHCONTENT_SOURCE_DIR_" + ("MTXDOWNLOAD_" + m.replace("/", "_")).upper():
                "${sourceDir}/build/${presetName}/extern/matrixmarket/" + m for m in listed}
    source = manifest["source"]
    ok = (len(listed) == 9 and listed == tests == cache and overrides == expected
          and source["repository"] == "arrayfire/arrayfire-data" and source["path"] == "matrixmarket/"
          and re.fullmatch(r"|[0-9a-f]{40}", source["commit"]) is not None
          and preset.get("AF_TEST_WITH_MTX_FILES") == "$env{AF_CI_MTX_FILES}"
          and 'set(AF_TEST_WITH_MTX_FILES "$ENV{AF_CI_MTX_FILES}" CACHE BOOL "" FORCE)' in text)
    return ok, (f"manifest {len(listed)}, tests {len(tests)}, matrixmarket.cmake {len(cache)}, ci preset "
                f"{len(overrides)} matrices; source {source}; AF_TEST_WITH_MTX_FILES from AF_CI_MTX_FILES "
                f"in the preset ({preset.get('AF_TEST_WITH_MTX_FILES')}) and in matrixmarket.cmake")


CASES = [
    ("no commit pinned: OFF, with its line, and nothing downloaded or restored", case_unpinned),
    ("a correct source: ON, every matrix restored, and a rerun finds them in place", case_correct),
    ("a pinned commit downloads from raw.githubusercontent.com at that commit", case_pinned_url),
    ("a branch name instead of a commit SHA fails", case_branch_name),
    ("one changed byte fails, naming the file", case_changed_byte),
    ("a wrong size fails, naming the file", case_wrong_size),
    ("a missing file fails, naming the file and its URL", case_missing_file),
    ("an unreachable server fails, naming every file", case_unreachable),
    ("the cache keeps verified downloads and replaces one that changed", case_cache),
    ("a matrix the tests use but the manifest lacks fails, naming it", case_new_test_matrix),
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

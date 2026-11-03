#!/usr/bin/env python3
"""Gets the Matrix Market test data from a pinned arrayfire-data commit, verifies it and
restores it into a build tree, before CMake configures it.

  check-matrix-data.py [--fixtures <dir>] [--source <url>] [--cache <dir>] <source dir> <build dir>

The tests' test/CMakeModules/download_sparse_datasets.cmake downloads its matrices from the
SuiteSparse Matrix Collection, whose storage refuses every download (403 AccessDenied).
ArrayFire's arrayfire-data repository holds the same files, unchanged, under matrixmarket/;
matrixmarket.json pins the commit to take them from, and records each matrix's size and
SHA-256.

It prints ON or OFF on stdout, the value for AF_TEST_WITH_MTX_FILES, and its report on stderr:
- OFF while matrixmarket.json pins no commit: the Matrix Market tests stay off, and nothing
  is downloaded.
- ON once every matrix is downloaded from
  https://raw.githubusercontent.com/<repository>/<commit>/<path>, or from --source instead
  (a URL such as file:///path/to/arrayfire-data/matrixmarket/), matches the manifest, and is
  restored into <build dir>/extern/matrixmarket/<group>/<name>/. The CI configurations point
  FetchContent there (FETCHCONTENT_SOURCE_DIR_MTXDOWNLOAD_<GROUP>_<NAME>, in
  matrixmarket.cmake and the "ci" preset), so the configure downloads nothing.
- It fails instead, naming each file at fault, when a download fails, when a file doesn't
  match the manifest, or when the source tree's tests use other matrices than it lists.
--cache keeps the verified downloads in a directory, and uses those that still match.
"""
import argparse
import hashlib
import json
import pathlib
import re
import sys
import urllib.request

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
MANIFEST = "matrixmarket.json"
RAW = "https://raw.githubusercontent.com/{repository}/{commit}/{path}"
TIMEOUT = 60


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tested_matrices(source):
    """The matrices the source tree's tests download, as <group>/<name>/<name>.mtx."""
    text = (source / "test/CMakeModules/download_sparse_datasets.cmake").read_text(encoding="utf-8")
    pairs = re.findall(r'^\s*mtxDownload\("([^"/]+)"\s+"([^"/]+)"\)', text, re.MULTILINE)
    return {f"{group}/{name}/{name}.mtx" for name, group in pairs}


def pinned_url(source):
    """The URL of the pinned commit's matrixmarket/ directory, or None while none is pinned."""
    commit = source["commit"]
    if not commit:
        return None
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError(f"{MANIFEST}: the commit must be a full 40-character SHA, not '{commit}'")
    return RAW.format(repository=source["repository"], commit=commit, path=source["path"])


def matches(data, spec):
    return data is not None and len(data) == spec["bytes"] and sha256(data) == spec["sha256"]


def place(root, path):
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError(f"{path}: outside {root}")
    return target


def write(target, data):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".partial")
    partial.write_bytes(data)
    partial.replace(target)


def fetch(files, base, cache):
    """The matrices, {path: content}, and how many came from the cache, once each matches the
    manifest; else raises ValueError with one line per file at fault."""
    matrices, cached, errors = {}, 0, []
    for path, spec in sorted(files.items()):
        kept = place(cache, path) if cache else None
        if kept and kept.is_file() and matches(kept.read_bytes(), spec):
            matrices[path] = kept.read_bytes()
            cached += 1
            continue
        url = base.rstrip("/") + "/" + path
        try:
            with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
                data = response.read(spec["bytes"] + 1)
        except (OSError, ValueError) as error:
            errors.append(f"{path}: downloading {url} failed ({error})")
            continue
        if not matches(data, spec):
            size = len(data) if len(data) <= spec["bytes"] else f"more than {spec['bytes']}"
            errors.append(f"{path}: {size} bytes with SHA-256 {sha256(data)} from {url}, but "
                          f"{MANIFEST} says {spec['bytes']} bytes with {spec['sha256']}")
            continue
        matrices[path] = data
        if kept:
            write(kept, data)
    if errors:
        raise ValueError("\n".join(errors))
    return matrices, cached


def restore(matrices, build):
    """Writes the matrices under <build>/extern/matrixmarket; returns how many it wrote."""
    root = build / "extern/matrixmarket"
    written = 0
    for path, data in sorted(matrices.items()):
        target = place(root, path)
        if target.is_file() and sha256(target.read_bytes()) == sha256(data):
            continue
        write(target, data)
        written += 1
    return written


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", type=pathlib.Path, default=FIXTURES)
    parser.add_argument("--source", help="the URL to download <group>/<name>/<name>.mtx from instead")
    parser.add_argument("--cache", type=pathlib.Path, help="a directory that keeps the verified downloads")
    parser.add_argument("tree", type=pathlib.Path, metavar="source-dir")
    parser.add_argument("build", type=pathlib.Path, metavar="build-dir")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((args.fixtures / MANIFEST).read_text(encoding="utf-8"))
        base = args.source or pinned_url(manifest["source"])
        if base is None:
            print("Matrix Market data: no arrayfire-data commit pinned yet; the Matrix Market tests are off",
                  file=sys.stderr)
            print("OFF")
            return 0
        files = manifest["files"]
        wanted = tested_matrices(args.tree)
        errors = [f"{path}: the tests use it, but {MANIFEST} doesn't list it" for path in sorted(wanted - set(files))]
        errors += [f"{path}: {MANIFEST} lists it, but the tests don't use it" for path in sorted(set(files) - wanted)]
        if errors:
            raise ValueError("\n".join(errors))
        matrices, cached = fetch(files, base, args.cache)
        written = restore(matrices, args.build)
    except (OSError, ValueError, KeyError) as error:
        print(f"The Matrix Market test data is not usable:\n{error}", file=sys.stderr)
        return 1
    print(f"Matrix Market data: {len(matrices)} matrices match {MANIFEST} ({len(matrices) - cached} downloaded "
          f"from {base}, {cached} from the cache); {written} restored, {len(matrices) - written} already "
          f"in place, in {args.build / 'extern/matrixmarket'}", file=sys.stderr)
    print("ON")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

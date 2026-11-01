#!/usr/bin/env python3
"""Verifies the Matrix Market test data bundled in .github/fixtures and restores it into a
build tree, before CMake configures it.

  check-matrix-data.py [--fixtures <dir>] <source dir> <build dir>

The tests' test/CMakeModules/download_sparse_datasets.cmake downloads its matrices from the
SuiteSparse Matrix Collection, which refuses CI's downloads (HTTP 403). The bundle,
matrixmarket.tar.gz, holds those matrices; matrixmarket.json records the archive's and each
matrix's SHA-256 and size. The check fails, naming each file at fault, when the archive or a
matrix doesn't match the manifest, or when the source tree's tests list other matrices than
the bundle holds. The matrices go to <build dir>/extern/matrixmarket/<group>/<name>/, where
the CI configurations point FetchContent (FETCHCONTENT_SOURCE_DIR_MTXDOWNLOAD_<GROUP>_<NAME>,
in matrixmarket.cmake and the "ci" preset), so that the configure downloads none of them.
"""
import argparse
import hashlib
import io
import json
import pathlib
import re
import sys
import tarfile
import zlib

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"
ARCHIVE = "matrixmarket.tar.gz"
MANIFEST = "matrixmarket.json"
LIMIT = 64 * 1024 * 1024  # the most a valid bundle may unpack to


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def tested_matrices(source):
    """The matrices the source tree's tests download, as <group>/<name>/<name>.mtx."""
    text = (source / "test/CMakeModules/download_sparse_datasets.cmake").read_text(encoding="utf-8")
    pairs = re.findall(r'^\s*mtxDownload\("([^"/]+)"\s+"([^"/]+)"\)', text, re.MULTILINE)
    return {f"{group}/{name}/{name}.mtx" for name, group in pairs}


def verify(fixtures, wanted):
    """The bundle's matrices, {path: content}, once each matches the manifest; else raises
    ValueError with one line per file at fault."""
    archive = fixtures / ARCHIVE
    manifest = json.loads((fixtures / MANIFEST).read_text(encoding="utf-8"))
    files = manifest["files"]
    errors = [f"{path}: the tests use it, but {MANIFEST} doesn't list it"
              for path in sorted(wanted - set(files))]
    errors += [f"{path}: {MANIFEST} lists it, but the tests don't use it"
               for path in sorted(set(files) - wanted)]
    blob = archive.read_bytes()
    if sha256(blob) != manifest["archive_sha256"]:
        errors.append(f"{ARCHIVE}: SHA-256 {sha256(blob)}, but {MANIFEST} says {manifest['archive_sha256']}")
    matrices = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as bundle:
            total = 0
            for member in bundle.getmembers():
                spec = files.get(member.name)
                if spec is None or not member.isfile():
                    errors.append(f"{member.name}: in {ARCHIVE}, but not a file {MANIFEST} lists")
                    continue
                total += member.size
                if total > LIMIT:
                    errors.append(f"{ARCHIVE}: unpacks to more than {LIMIT} bytes")
                    break
                data = bundle.extractfile(member).read()
                if len(data) != spec["bytes"] or sha256(data) != spec["sha256"]:
                    errors.append(f"{member.name}: {len(data)} bytes with SHA-256 {sha256(data)}, but "
                                  f"{MANIFEST} says {spec['bytes']} bytes with {spec['sha256']}")
                matrices[member.name] = data
        errors += [f"{path}: {MANIFEST} lists it, but {ARCHIVE} lacks it"
                   for path in sorted(set(files) - set(matrices))]
    except (tarfile.TarError, OSError, EOFError, zlib.error) as error:
        errors.append(f"{ARCHIVE}: not a readable gzip tar archive ({error})")
    if errors:
        raise ValueError("\n".join(errors))
    return matrices


def restore(matrices, build):
    """Writes the matrices under <build>/extern/matrixmarket; returns how many it wrote."""
    root = (build / "extern/matrixmarket").resolve()
    written = 0
    for path, data in sorted(matrices.items()):
        target = (root / path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"{path}: outside {root}")
        if target.is_file() and sha256(target.read_bytes()) == sha256(data):
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".partial")
        partial.write_bytes(data)
        partial.replace(target)
        written += 1
    return written


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixtures", type=pathlib.Path, default=FIXTURES)
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("build", type=pathlib.Path)
    args = parser.parse_args(argv)
    try:
        matrices = verify(args.fixtures, tested_matrices(args.source))
        written = restore(matrices, args.build)
    except (OSError, ValueError, KeyError) as error:
        print(f"The Matrix Market test data in {args.fixtures} is not usable:\n{error}", file=sys.stderr)
        return 1
    print(f"Matrix Market test data: {len(matrices)} matrices match {MANIFEST}; "
          f"{written} restored, {len(matrices) - written} already in place, in "
          f"{args.build / 'extern/matrixmarket'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

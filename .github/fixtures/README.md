# Matrix Market test data

`matrixmarket.tar.gz` holds the matrices that the tests download in
`test/CMakeModules/download_sparse_datasets.cmake`. They are public scientific
datasets from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/),
which refuses downloads from CI. Each file keeps its original header, author
credits and collection metadata.

`matrixmarket.json` records each matrix's source URL, the SHA-256 of the file and
of its original archive, its size, dimensions and entry count, and the SHA-256 of
the bundle. Before every build, `ci.sh` runs `.github/scripts/check-matrix-data.py`,
which checks the bundle and every matrix against it, fails naming any file that
differs, and restores the matrices into the build tree. The CI configurations point
the tests' download at them (`.github/scripts/matrixmarket.cmake` and the `ci`
preset), so the configure downloads nothing.

When the tests' list of matrices changes, update the bundle, the manifest,
`matrixmarket.cmake` and the `ci` preset together; `check-matrix-data-test.py`
checks that they agree.

The bundle and the manifest come from ArrayFire's fix/restore-ci branch.

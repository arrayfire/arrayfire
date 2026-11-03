# Matrix Market test data

The Matrix Market tests read matrices that `test/CMakeModules/download_sparse_datasets.cmake`
downloads from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/), whose storage
refuses every download. ArrayFire's [arrayfire-data](https://github.com/arrayfire/arrayfire-data)
repository holds the same files, unchanged, under `matrixmarket/`, with their license
(CC BY 4.0) and citations.

`matrixmarket.json` pins the arrayfire-data commit that CI downloads them from, and records
each matrix's size and SHA-256, and its original SuiteSparse URL and archive SHA-256. Before
every build, `ci.sh` runs `.github/scripts/check-matrix-data.py`:

- **While `commit` is empty,** it prints that no commit is pinned, and the Matrix Market
  tests are off (`AF_TEST_WITH_MTX_FILES=OFF`).
- **With a commit,** it downloads each matrix from
  `https://raw.githubusercontent.com/arrayfire/arrayfire-data/<commit>/matrixmarket/`, checks
  it against the manifest, and restores it into the build tree. The tests are then on, and the
  CI configurations point their download at the restored files
  (`.github/scripts/matrixmarket.cmake` and the `ci` preset), so the configure downloads
  nothing. A failed download or check fails the build, naming the file.

To turn the tests on, set `commit` to a full commit SHA of arrayfire-data that has
`matrixmarket/`. When the tests' list of matrices changes, update the manifest,
`matrixmarket.cmake` and the `ci` preset together; `check-matrix-data-test.py` checks that
they agree.

The manifest comes from ArrayFire's fix/restore-ci branch.

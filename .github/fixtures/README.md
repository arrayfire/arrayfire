# Matrix fixtures for CI

`matrixmarket.tar.gz` contains the Matrix Market fixtures listed in
`test/CMakeModules/download_sparse_datasets.cmake`. They are public scientific
datasets from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/).
The original matrix headers, author credits, and collection metadata remain
in each file.

The bundle comes from an existing ArrayFire download cache. Each file matches
its retained upstream archive. `matrixmarket.json` records the source URLs,
archive and file SHA-256 checksums, dimensions, and entry counts. CI verifies
the bundle before restoring it and also reuses the GitHub Actions cache.

Keep the fixture list, bundle, and manifest together when changing these tests.
The bundle lets the existing tests run during collection outages without
repeated downloads from the provider.

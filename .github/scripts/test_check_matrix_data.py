import hashlib
import io
import json
import pathlib
import tarfile
import tempfile
import unittest

from check_matrix_data import check_cache


class MatrixCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = pathlib.Path(self.temp.name)
        manifest = self.repo / "test/CMakeModules/download_sparse_datasets.cmake"
        manifest.parent.mkdir(parents=True)
        manifest.write_text('mtxDownload("Trec4" "JGD_Kocay")\n', encoding="utf-8")
        self.cache = self.repo / "cache"

    def test_complete_cache_works_without_network_during_outage(self):
        matrix = self.cache / "JGD_Kocay/Trec4/Trec4.mtx"
        matrix.parent.mkdir(parents=True)
        matrix.write_text("fixture", encoding="utf-8")

        def no_network():
            self.fail("A complete cache must not query the provider")

        self.assertIn("cached", check_cache(self.repo, self.cache, no_network))

    def test_empty_fixture_does_not_hide_provider_outage(self):
        matrix = self.cache / "JGD_Kocay/Trec4/Trec4.mtx"
        matrix.parent.mkdir(parents=True)
        matrix.touch()
        with self.assertRaisesRegex(RuntimeError, "tests remain enabled"):
            check_cache(self.repo, self.cache, lambda: "The collection is offline")

    def make_bundle(self):
        directory = self.repo / ".github/fixtures"
        directory.mkdir(parents=True)
        archive = directory / "matrixmarket.tar.gz"
        payload = b"matrix fixture"
        relative = "JGD_Kocay/Trec4/Trec4.mtx"
        with tarfile.open(archive, "w:gz") as bundle:
            member = tarfile.TarInfo(relative)
            member.size = len(payload)
            bundle.addfile(member, io.BytesIO(payload))
        (directory / "matrixmarket.json").write_text(json.dumps({
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "files": {relative: {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}}
        }), encoding="utf-8")
        return archive, payload

    def test_verified_bundle_restores_without_network(self):
        _, payload = self.make_bundle()

        def no_network():
            self.fail("The bundle must work during a provider outage")

        self.assertIn("bundled", check_cache(self.repo, self.cache, no_network))
        self.assertEqual((self.cache / "JGD_Kocay/Trec4/Trec4.mtx").read_bytes(), payload)

    def test_corrupt_bundle_is_rejected_before_cache_writes(self):
        archive, _ = self.make_bundle()
        archive.write_bytes(b"corrupt archive")
        with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
            check_cache(self.repo, self.cache)
        self.assertFalse(self.cache.exists())


if __name__ == "__main__":
    unittest.main()

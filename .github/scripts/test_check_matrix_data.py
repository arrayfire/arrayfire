import pathlib
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


if __name__ == "__main__":
    unittest.main()

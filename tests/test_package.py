import json, unittest, zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from pipeline.contracts import Artifact, Manifest
from pipeline.package import PackageConfig, package_artifacts


class PackageTest(unittest.TestCase):
    def _manifest(self, d):
        m = Manifest(job_id="job1")
        for name in ("design", "reconstruct", "shell"):
            p = Path(d) / f"{name}.bin"
            p.write_bytes(name.encode())
            m.add(Artifact.create(name, p, {"note": name}))
        return m

    def test_zip_contains_artifacts_manifest_notice(self):
        with TemporaryDirectory() as d:
            m = self._manifest(d)
            out = Path(d) / "pkg.zip"
            art = package_artifacts(m, out)
            self.assertEqual(art.stage, "package")
            self.assertTrue(out.exists())
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
            self.assertTrue(any(n.endswith("design.bin") for n in names))
            self.assertTrue(any(n.endswith("shell.bin") for n in names))
            self.assertIn("manifest.json", names)
            self.assertIn("NOTICE.md", names)

    def test_manifest_json_is_valid_and_readable(self):
        with TemporaryDirectory() as d:
            m = self._manifest(d)
            out = Path(d) / "pkg.zip"
            package_artifacts(m, out)
            with zipfile.ZipFile(out) as z:
                data = json.loads(z.read("manifest.json").decode("utf-8"))
            self.assertEqual(data["job_id"], "job1")
            self.assertIn("shell", data["artifacts"])

    def test_stage_filter(self):
        with TemporaryDirectory() as d:
            m = self._manifest(d)
            out = Path(d) / "pkg.zip"
            package_artifacts(m, out, PackageConfig(include_stages=("shell",)))
            with zipfile.ZipFile(out) as z:
                names = z.namelist()
            self.assertFalse(any(n.endswith("design.bin") for n in names))
            self.assertTrue(any(n.endswith("shell.bin") for n in names))

    def test_missing_file_is_skipped(self):
        with TemporaryDirectory() as d:
            m = self._manifest(d)
            (Path(d) / "design.bin").unlink()
            out = Path(d) / "pkg.zip"
            art = package_artifacts(m, out)
            self.assertTrue(art.sha256)

    def test_artifact_hash_matches_file(self):
        with TemporaryDirectory() as d:
            m = self._manifest(d)
            out = Path(d) / "pkg.zip"
            art = package_artifacts(m, out)
            from pipeline.contracts import file_sha256
            self.assertEqual(art.sha256, file_sha256(out))


if __name__ == "__main__":
    unittest.main()

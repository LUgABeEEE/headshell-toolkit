import tempfile, unittest
from pathlib import Path
from pipeline.contracts import Artifact, Manifest, STAGES, file_sha256

class ContractsTest(unittest.TestCase):
    def test_artifact_and_manifest_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.bin"; p.write_bytes(b"hello")
            a = Artifact.create("design", p, {"k": 1})
            self.assertEqual(a.sha256, file_sha256(p))
            m = Manifest(job_id="j1"); m.add(a); m.status = "completed"
            q = Path(d) / "m.json"; m.save(q)
            m2 = Manifest.load(q)
            self.assertEqual(m2.artifacts["design"].sha256, a.sha256)
            self.assertTrue(m2.is_done("design"))

    def test_unknown_stage_rejected(self):
        with self.assertRaises(ValueError):
            Artifact(stage="bogus", path="x", sha256="0"*64).validate()

    def test_tamper_detected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.bin"; p.write_bytes(b"hello")
            a = Artifact.create("design", p)
            m = Manifest(job_id="j"); m.add(a)
            p.write_bytes(b"tampered")
            self.assertFalse(m.is_done("design"))

if __name__ == "__main__":
    unittest.main()

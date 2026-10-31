import tempfile, unittest
from pathlib import Path
from pipeline.contracts import Artifact, Manifest
from pipeline.runner import PipelineRunner

class FakeAdapter:
    def __init__(self, stage, fail=False):
        self.stage = stage; self.fail = fail; self.calls = 0
    def run(self, context):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        p = Path(context["workdir"]) / f"{self.stage}.bin"
        p.write_bytes(self.stage.encode())
        return Artifact.create(self.stage, p)

class RunnerTest(unittest.TestCase):
    def test_runs_and_skips_on_resume(self):
        with tempfile.TemporaryDirectory() as d:
            a = FakeAdapter("design"); b = FakeAdapter("shell")
            r = PipelineRunner(Path(d)/"m.json", {"design": a, "shell": b})
            m = r.run(("design", "shell"))
            self.assertEqual(m.status, "completed")
            self.assertTrue((Path(d)/"m.json").exists())
            r2 = PipelineRunner(Path(d)/"m.json", {"design": a, "shell": b})
            m2 = r2.run(("design", "shell"))
            self.assertEqual(a.calls, 1); self.assertEqual(b.calls, 1)
            self.assertEqual(m2.status, "completed")

    def test_failure_marks_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            bad = FakeAdapter("shell", fail=True)
            r = PipelineRunner(Path(d)/"m.json", {"design": FakeAdapter("design"), "shell": bad})
            with self.assertRaises(RuntimeError):
                r.run(("design", "shell"))
            m = Manifest.load(Path(d)/"m.json")
            self.assertEqual(m.status, "failed")
            self.assertIn("boom", m.error or "")

    def test_missing_adapter(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                PipelineRunner(Path(d)/"m.json", {}).run(("design",))

if __name__ == "__main__":
    unittest.main()

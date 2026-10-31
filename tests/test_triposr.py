import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adapters.reconstruct.triposr import TripoSRAdapter, TripoSRConfig


class FakeRunner:
    def __init__(self, exit_code=0, write_obj=True, write_bytes=b"v 0 0 0\n"):
        self.exit_code = exit_code
        self.write_obj = write_obj
        self.write_bytes = write_bytes
        self.calls = []

    def __call__(self, cmd, timeout):
        self.calls.append((list(cmd), timeout))
        if self.write_obj:
            out = None
            for i, token in enumerate(cmd):
                if token in ("--output-dir", "-o", "--out") and i + 1 < len(cmd):
                    out = Path(cmd[i + 1])
            if out is None:
                out = Path(cmd[-1]).parent
            out.mkdir(parents=True, exist_ok=True)
            (out / "mesh.obj").write_bytes(self.write_bytes)
        return self.exit_code


class TripoSRTest(unittest.TestCase):
    def _adapter(self, runner):
        return TripoSRAdapter(TripoSRConfig(python="py", script="run.py", timeout=42), runner=runner)

    def test_build_command_contains_image_and_output(self):
        cmd = self._adapter(FakeRunner()).build_command("a.png", "out/")
        joined = " ".join(cmd)
        self.assertIn("a.png", joined)
        self.assertIn("out", joined)
        self.assertEqual(cmd[0], "py")

    def test_run_writes_artifact(self):
        with TemporaryDirectory() as d:
            img = Path(d) / "a.png"; img.write_bytes(b"png")
            runner = FakeRunner()
            art = self._adapter(runner).run({"workdir": d, "input_image": str(img)})
            self.assertEqual(art.stage, "reconstruct")
            self.assertTrue(Path(art.path).exists())
            self.assertEqual(len(art.sha256), 64)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(runner.calls[0][1], 42)

    def test_image_from_design_artifact(self):
        with TemporaryDirectory() as d:
            img = Path(d) / "a.png"; img.write_bytes(b"png")
            from pipeline.contracts import Artifact
            art = Artifact.create("design", img)
            from pipeline.contracts import Manifest
            m = Manifest(job_id="j"); m.add(art)
            out = self._adapter(FakeRunner()).run({"workdir": d, "manifest": m})
            self.assertEqual(out.stage, "reconstruct")

    def test_nonzero_exit_raises(self):
        with TemporaryDirectory() as d:
            img = Path(d) / "a.png"; img.write_bytes(b"png")
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner(exit_code=1)).run({"workdir": d, "input_image": str(img)})

    def test_missing_obj_raises(self):
        with TemporaryDirectory() as d:
            img = Path(d) / "a.png"; img.write_bytes(b"png")
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner(write_obj=False)).run({"workdir": d, "input_image": str(img)})

    def test_empty_obj_raises(self):
        with TemporaryDirectory() as d:
            img = Path(d) / "a.png"; img.write_bytes(b"png")
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner(write_bytes=b"")).run({"workdir": d, "input_image": str(img)})

    def test_no_image_raises(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner()).run({"workdir": d})


if __name__ == "__main__":
    unittest.main()

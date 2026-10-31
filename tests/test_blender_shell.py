import json, struct, unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from adapters.shell.blender_shell import BlenderShellAdapter, BlenderShellConfig

CUBE = [
    ((-1,-1,-1),(-1,1,-1),(1,1,-1)), ((-1,-1,-1),(1,1,-1),(1,-1,-1)),
    ((-1,-1,1),(1,-1,1),(1,1,1)), ((-1,-1,1),(1,1,1),(-1,1,1)),
    ((-1,-1,-1),(1,-1,-1),(1,-1,1)), ((-1,-1,-1),(1,-1,1),(-1,-1,1)),
    ((-1,1,-1),(-1,1,1),(1,1,1)), ((-1,1,-1),(1,1,1),(1,1,-1)),
    ((-1,-1,-1),(-1,-1,1),(-1,1,1)), ((-1,-1,-1),(-1,1,1),(-1,1,-1)),
    ((1,-1,-1),(1,1,-1),(1,1,1)), ((1,-1,-1),(1,1,1),(1,-1,1)),
]


def write_stl(path, tris):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"\0" * 80); f.write(struct.pack("<I", len(tris)))
        for a, b, c in tris:
            f.write(struct.pack("<12fH", 0, 0, 0, *a, *b, *c, 0))


class FakeRunner:
    def __init__(self, exit_code=0, write_stl_out=True, tris=None):
        self.exit_code = exit_code
        self.write_stl_out = write_stl_out
        self.tris = CUBE if tris is None else tris
        self.calls = []
        self.config_seen = None

    def __call__(self, cmd, timeout):
        self.calls.append((list(cmd), timeout))
        cfg_path = Path(cmd[cmd.index("--config") + 1])
        self.config_seen = json.loads(cfg_path.read_text(encoding="utf-8"))
        if self.write_stl_out:
            write_stl(Path(self.config_seen["out"]), self.tris)
        return self.exit_code


class BlenderShellTest(unittest.TestCase):
    def _adapter(self, runner, **params):
        cfg = BlenderShellConfig(blender="bl", script="k.py", timeout=99, params=params)
        return BlenderShellAdapter(cfg, runner=runner)

    def test_build_config_merges_params(self):
        new = self._adapter(FakeRunner(), liner_mm=8.0).build_config("o.stl", "out.stl", "i.stl")
        self.assertEqual(new["outer"], "o.stl")
        self.assertEqual(new["inner"], "i.stl")
        self.assertEqual(new["out"], "out.stl")
        self.assertEqual(new["liner_mm"], 8.0)

    def test_run_produces_verified_artifact(self):
        with TemporaryDirectory() as d:
            outer = Path(d) / "outer.stl"; write_stl(outer, CUBE)
            runner = FakeRunner()
            art = self._adapter(runner, expected_bbox_mm=[2, 2, 2]).run(
                {"workdir": d, "outer_mesh": str(outer)})
            self.assertEqual(art.stage, "shell")
            self.assertTrue(Path(art.path).exists())
            self.assertEqual(runner.calls[0][1], 99)
            self.assertEqual(art.meta.get("watertight"), True)

    def test_nonzero_exit_raises(self):
        with TemporaryDirectory() as d:
            outer = Path(d) / "outer.stl"; write_stl(outer, CUBE)
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner(exit_code=2)).run({"workdir": d, "outer_mesh": str(outer)})

    def test_missing_stl_raises(self):
        with TemporaryDirectory() as d:
            outer = Path(d) / "outer.stl"; write_stl(outer, CUBE)
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner(write_stl_out=False)).run({"workdir": d, "outer_mesh": str(outer)})

    def test_verify_failure_raises(self):
        with TemporaryDirectory() as d:
            outer = Path(d) / "outer.stl"; write_stl(outer, CUBE)
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner(tris=CUBE[:-2])).run({"workdir": d, "outer_mesh": str(outer)})

    def test_no_outer_mesh_raises(self):
        with TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                self._adapter(FakeRunner()).run({"workdir": d})

    def test_run_uses_manifest_reconstruct_artifact(self):
        from pipeline.contracts import Artifact, Manifest
        with TemporaryDirectory() as d:
            obj = Path(d) / "m.obj"; obj.write_bytes(b"v 0 0 0\n")
            m = Manifest(job_id="j"); m.add(Artifact.create("reconstruct", obj))
            art = self._adapter(FakeRunner()).run({"workdir": d, "manifest": m})
            self.assertEqual(art.stage, "shell")


if __name__ == "__main__":
    unittest.main()

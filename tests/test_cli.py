import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from cli.headshell import main
from tests.fixtures import write_cube_stl


def run(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class CliTest(unittest.TestCase):
    def test_version(self):
        code, out = run(["version"])
        self.assertEqual(code, 0)
        self.assertIn("headshell", out.lower())

    def test_params_validate(self):
        full = {"schema": "headshell/params@0", "ipd_mm": 64.0, "head_width_mm": 220.0,
                "wall_mm": 2.5, "liner_mm": 6.0, "vent_diameter_mm": 5.0,
                "vent_spacing_mm": 14.0, "split": {"plane": "coronal", "offset_mm": -15.0}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "p.json"
            # schema-only is not a buildable set any more: the contract rejects it
            path.write_text(json.dumps({"schema": "headshell/params@0"}))
            code, out = run(["params-validate", str(path)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["ok"], False)

            path.write_text(json.dumps(full))
            code, out = run(["params-validate", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out)["ok"], True)

            path.write_text(json.dumps({**full, "ipd_mm": 999}))
            code, out = run(["params-validate", str(path)])
            self.assertEqual(code, 1)
            self.assertEqual(json.loads(out)["ok"], False)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_cube_stl(Path(tmp) / "cube.stl", 10.0)
            code, out = run(["stats", str(path)])
            self.assertEqual(code, 0)
            data = json.loads(out)
            self.assertTrue(data["watertight"])
            self.assertAlmostEqual(data["volume_cm3"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()


class CliEdgeCaseTest(unittest.TestCase):
    def test_no_command(self):
        self.assertEqual(run([])[0], 2)

    def test_bad_file(self):
        code, out = run(["stats", "/nonexistent/file.stl"])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads(out)["ok"])

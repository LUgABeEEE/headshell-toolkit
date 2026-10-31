import tempfile
import unittest
from pathlib import Path

from core.geometry import bbox_mm, mesh_stats, parse_binary_stl, slice_width_mm
from tests.fixtures import cube_triangles, write_binary_stl, write_cube_stl


class GeometryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = write_cube_stl(Path(self.tmp.name) / "cube.stl", 10.0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse(self):
        tris = parse_binary_stl(self.path)
        self.assertEqual(len(tris), 12)
        self.assertEqual(len(tris[0]), 3)
        self.assertAlmostEqual(tris[0][0][0], 0.0)

    def test_bbox(self):
        mn, mx, size = bbox_mm(cube_triangles(10.0))
        self.assertEqual(size, (10.0, 10.0, 10.0))
        self.assertEqual(mn, (0.0, 0.0, 0.0))
        self.assertEqual(mx, (10.0, 10.0, 10.0))

    def test_stats_closed_cube(self):
        stats = mesh_stats(cube_triangles(10.0))
        self.assertEqual(stats["triangles"], 12)
        self.assertEqual(stats["unique_vertices"], 8)
        self.assertEqual(stats["boundary_edges"], 0)
        self.assertEqual(stats["nonmanifold_edges"], 0)
        self.assertTrue(stats["watertight"])
        self.assertAlmostEqual(stats["volume_cm3"], 1.0, places=6)
        self.assertEqual(tuple(stats["bbox_mm"]), (10.0, 10.0, 10.0))

    def test_open_mesh_not_watertight(self):
        tris = cube_triangles(10.0)[:-1]
        stats = mesh_stats(tris)
        self.assertEqual(stats["boundary_edges"], 3)
        self.assertFalse(stats["watertight"])

    def test_slice_width(self):
        self.assertAlmostEqual(slice_width_mm(cube_triangles(10.0), 5.0), 10.0)
        self.assertEqual(slice_width_mm(cube_triangles(10.0), 99.0), 0.0)


if __name__ == "__main__":
    unittest.main()


class StlEdgeCaseTest(unittest.TestCase):
    def test_trailing_bytes_tolerated(self):
        import struct
        from tests.fixtures import cube_triangles, write_binary_stl
        with tempfile.TemporaryDirectory() as tmp:
            raw = write_binary_stl(Path(tmp) / "c.stl", cube_triangles(10.0))
            with raw.open("ab") as f:
                f.write(b"padding!")
            self.assertEqual(len(parse_binary_stl(raw)), 12)

    def test_truncated_rejected(self):
        from tests.fixtures import cube_triangles, write_binary_stl
        with tempfile.TemporaryDirectory() as tmp:
            raw = write_binary_stl(Path(tmp) / "c.stl", cube_triangles(10.0))
            data = raw.read_bytes()
            raw.write_bytes(data[:-10])
            with self.assertRaises(ValueError):
                parse_binary_stl(raw)

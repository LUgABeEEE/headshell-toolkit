import struct, tempfile, unittest
from pathlib import Path
from pipeline.verify import verify_stl

CUBE = [
    # bottom (z-)
    ((-1,-1,-1),(-1,1,-1),(1,1,-1)), ((-1,-1,-1),(1,1,-1),(1,-1,-1)),
    # top (z+)
    ((-1,-1,1),(1,-1,1),(1,1,1)), ((-1,-1,1),(1,1,1),(-1,1,1)),
    # front (y-)
    ((-1,-1,-1),(1,-1,-1),(1,-1,1)), ((-1,-1,-1),(1,-1,1),(-1,-1,1)),
    # back (y+)
    ((-1,1,-1),(-1,1,1),(1,1,1)), ((-1,1,-1),(1,1,1),(1,1,-1)),
    # left (x-)
    ((-1,-1,-1),(-1,-1,1),(-1,1,1)), ((-1,-1,-1),(-1,1,1),(-1,1,-1)),
    # right (x+)
    ((1,-1,-1),(1,1,-1),(1,1,1)), ((1,-1,-1),(1,1,1),(1,-1,1)),
]

def write_stl(path, tris):
    with open(path, 'wb') as f:
        f.write(b'\0'*80); f.write(struct.pack('<I', len(tris)))
        for a,b,c in tris:
            f.write(struct.pack('<12fH', 0,0,0, *a, *b, *c, 0))

class VerifyTest(unittest.TestCase):
    def test_cube_ok(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cube.stl'; write_stl(p, CUBE)
            r=verify_stl(p, expected_bbox_mm=(2.0,2.0,2.0), bbox_tolerance_mm=0.01)
            self.assertTrue(r.ok, r.errors)

    def test_bbox_mismatch(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'cube.stl'; write_stl(p, CUBE)
            r=verify_stl(p, expected_bbox_mm=(10.0,10.0,10.0), bbox_tolerance_mm=0.01)
            self.assertFalse(r.ok)

    def test_open_mesh_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'open.stl'; write_stl(p, CUBE[:-2])
            r=verify_stl(p)
            self.assertFalse(r.ok)

if __name__ == '__main__':
    unittest.main()

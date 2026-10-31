import unittest

from core.print.clearance import bbox_clearance


class ClearanceTest(unittest.TestCase):
    def test_passes(self):
        report = bbox_clearance((240.0, 260.0, 340.0), (166.0, 206.0, 246.0), 10.0)
        self.assertTrue(report.ok)
        self.assertAlmostEqual(report.gaps_mm[0], 37.0)
        self.assertAlmostEqual(report.gaps_mm[1], 27.0)
        self.assertAlmostEqual(report.gaps_mm[2], 47.0)

    def test_fails(self):
        report = bbox_clearance((200.0, 220.0, 300.0), (190.0, 210.0, 290.0), 10.0)
        self.assertFalse(report.ok)
        self.assertAlmostEqual(report.gaps_mm[0], 5.0)

    def test_non_finite(self):
        with self.assertRaises(ValueError):
            bbox_clearance((float("nan"), 200.0, 300.0), (190.0, 210.0, 290.0), 10.0)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            bbox_clearance((0.0, 200.0, 300.0), (190.0, 210.0, 290.0), 10.0)
        with self.assertRaises(ValueError):
            bbox_clearance((200.0, 220.0, 300.0), (190.0, 210.0, 290.0), -1.0)


if __name__ == "__main__":
    unittest.main()

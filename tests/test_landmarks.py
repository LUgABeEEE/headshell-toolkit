import math
import unittest

from core.fit.landmarks import Landmark, combined_score, match_by_name, rmse


class LandmarkTest(unittest.TestCase):
    def test_match_by_name(self):
        a = [Landmark("eye_l", 0, 0), Landmark("eye_r", 1, 1)]
        b = [Landmark("eye_r", 2, 2), Landmark("eye_l", 3, 3)]
        pairs = match_by_name(a, b)
        self.assertEqual([p[0].name for p in pairs], ["eye_l", "eye_r"])
        self.assertEqual(pairs[0][1].x, 3)

    def test_match_errors(self):
        with self.assertRaises(ValueError):
            match_by_name([Landmark("a", 0, 0), Landmark("a", 1, 1)], [Landmark("a", 0, 0)])
        with self.assertRaises(ValueError):
            match_by_name([Landmark("a", 0, 0)], [Landmark("b", 0, 0)])

    def test_rmse(self):
        target = [Landmark("a", 0, 0), Landmark("b", 0, 0)]
        predicted = [Landmark("a", 3, 4), Landmark("b", 6, 8)]
        self.assertAlmostEqual(rmse(target, predicted), math.sqrt(62.5), places=6)

    def test_weighted_rmse(self):
        target = [Landmark("a", 0, 0, 1.0), Landmark("b", 0, 0, 3.0)]
        predicted = [Landmark("a", 3, 4), Landmark("b", 6, 8)]
        self.assertAlmostEqual(rmse(target, predicted), math.sqrt(325.0 / 4.0), places=6)

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            rmse([], [])

    def test_combined_score(self):
        target = [Landmark("a", 0, 0)]
        predicted = [Landmark("a", 0, 0)]
        self.assertAlmostEqual(combined_score(target, predicted, 0.5), 0.1, places=6)
        with self.assertRaises(ValueError):
            combined_score(target, predicted, 1.5)


if __name__ == "__main__":
    unittest.main()


class NegativeWeightTest(unittest.TestCase):
    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError):
            rmse([Landmark("a", 0, 0, -1.0)], [Landmark("a", 0, 0)])

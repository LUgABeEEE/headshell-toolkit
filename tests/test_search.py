import unittest

from core.fit.search import coordinate_descent


def quadratic(p):
    return (p["x"] - 2.0) ** 2 + (p["y"] + 1.0) ** 2


class SearchTest(unittest.TestCase):
    def test_converges(self):
        result = coordinate_descent(
            {"x": 0.0, "y": 0.0},
            {"x": (-5.0, 5.0), "y": (-5.0, 5.0)},
            {"x": 1.0, "y": 1.0},
            quadratic,
            rounds=4,
        )
        self.assertAlmostEqual(result.best_params["x"], 2.0, delta=0.6)
        self.assertAlmostEqual(result.best_params["y"], -1.0, delta=0.6)
        self.assertLess(result.best_score, 0.5)
        self.assertGreater(len(result.history), 0)
        self.assertTrue(all("params" in h and "score" in h and "round" in h for h in result.history))

    def test_bounds_clipped(self):
        result = coordinate_descent(
            {"x": 100.0},
            {"x": (-1.0, 1.0)},
            {"x": 10.0},
            lambda p: abs(p["x"]),
            rounds=1,
        )
        self.assertLessEqual(result.best_params["x"], 1.0)
        self.assertGreaterEqual(result.best_params["x"], -1.0)

    def test_maximize(self):
        result = coordinate_descent(
            {"x": 0.0},
            {"x": (-5.0, 5.0)},
            {"x": 1.0},
            lambda p: -((p["x"]) ** 2),
            rounds=3,
            maximize=True,
        )
        self.assertAlmostEqual(result.best_params["x"], 0.0, delta=0.3)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            coordinate_descent({"x": 0.0}, {"y": (0, 1)}, {"x": 1.0}, quadratic)
        with self.assertRaises(ValueError):
            coordinate_descent({"x": 0.0}, {"x": (0, 1)}, {"x": 1.0}, quadratic, rounds=0)


if __name__ == "__main__":
    unittest.main()


class SearchValidationTest(unittest.TestCase):
    def test_bad_shrink_and_step(self):
        with self.assertRaises(ValueError):
            coordinate_descent({"x": 0.0}, {"x": (0, 1)}, {"x": 1.0}, quadratic, shrink=0)
        with self.assertRaises(ValueError):
            coordinate_descent({"x": 0.0}, {"x": (0, 1)}, {"x": 0.0}, quadratic)

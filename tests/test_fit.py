import unittest

from core.fit.fit import fit_measurements
from core.fit.measure import Measurement


def predict(p):
    return {
        "ipd_mm": Measurement("ipd_mm", 60.0 + 10.0 * p["x"]),
        "face_width_mm": Measurement("face_width_mm", 140.0 + 20.0 * p["y"]),
    }


class FitMeasurementsTest(unittest.TestCase):
    def test_recovers_parameters(self):
        target = predict({"x": 0.5, "y": -0.5})
        result = fit_measurements(
            target,
            predict,
            initial={"x": 0.0, "y": 0.0},
            bounds={"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            steps={"x": 0.5, "y": 0.5},
            rounds=4,
        )
        self.assertAlmostEqual(result.best_params["x"], 0.5, delta=0.16)
        self.assertAlmostEqual(result.best_params["y"], -0.5, delta=0.16)
        self.assertLess(result.best_score, 0.02)
        self.assertEqual(set(result.measurements), {"ipd_mm", "face_width_mm"})
        self.assertGreater(len(result.history), 0)

    def test_empty_target(self):
        with self.assertRaises(ValueError):
            fit_measurements({}, predict, {"x": 0.0, "y": 0.0},
                             {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                             {"x": 0.5, "y": 0.5})


if __name__ == "__main__":
    unittest.main()


class FitNameMismatchTest(unittest.TestCase):
    def test_predict_names_must_match_target(self):
        target = {"other_mm": Measurement("other_mm", 1.0)}
        with self.assertRaises(ValueError):
            fit_measurements(target, predict, {"x": 0.0, "y": 0.0},
                             {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                             {"x": 0.5, "y": 0.5}, rounds=1)

import unittest

from core.fit.measure import Measurement, measure_face, measurement_distance


def landmarks(scale=1.0):
    return {
        "eye_l": (20.0, 50.0),
        "eye_r": (80.0, 50.0),
        "jaw_l": (0.0, 80.0),
        "jaw_r": (100.0, 80.0),
        "mouth_l": (40.0, 90.0),
        "mouth_r": (60.0, 90.0),
        "brow_mid": (50.0, 20.0),
        "chin": (50.0, 120.0),
    }


class MeasureFaceTest(unittest.TestCase):
    def test_measure_face(self):
        m = measure_face(landmarks(), scale_mm_per_px=1.0)
        self.assertAlmostEqual(m["ipd_mm"].value_mm, 60.0)
        self.assertAlmostEqual(m["face_width_mm"].value_mm, 100.0)
        self.assertAlmostEqual(m["mouth_width_mm"].value_mm, 20.0)
        self.assertAlmostEqual(m["face_height_mm"].value_mm, 100.0)
        self.assertEqual(set(m), {"ipd_mm", "face_width_mm", "mouth_width_mm", "face_height_mm"})

    def test_scale(self):
        m = measure_face(landmarks(), scale_mm_per_px=2.0)
        self.assertAlmostEqual(m["ipd_mm"].value_mm, 120.0)
        self.assertAlmostEqual(m["face_height_mm"].value_mm, 200.0)

    def test_missing_landmark(self):
        data = landmarks()
        del data["chin"]
        with self.assertRaises(ValueError):
            measure_face(data)

    def test_bad_scale(self):
        with self.assertRaises(ValueError):
            measure_face(landmarks(), scale_mm_per_px=0.0)


class MeasurementDistanceTest(unittest.TestCase):
    def test_zero_for_identical(self):
        a = measure_face(landmarks())
        self.assertAlmostEqual(measurement_distance(a, a), 0.0)

    def test_relative_weighted(self):
        target = {
            "a": Measurement("a", 100.0, 1.0),
            "b": Measurement("b", 100.0, 3.0),
        }
        predicted = {
            "a": Measurement("a", 110.0, 1.0),  # 10% error
            "b": Measurement("b", 100.0, 3.0),  # exact
        }
        self.assertAlmostEqual(measurement_distance(target, predicted), 0.05, places=6)

    def test_absolute(self):
        target = {"a": Measurement("a", 100.0, 1.0)}
        predicted = {"a": Measurement("a", 110.0, 1.0)}
        self.assertAlmostEqual(measurement_distance(target, predicted, relative=False), 10.0)

    def test_errors(self):
        with self.assertRaises(ValueError):
            measurement_distance({}, {})
        with self.assertRaises(ValueError):
            measurement_distance({"a": Measurement("a", 1.0)}, {})
        with self.assertRaises(ValueError):
            measurement_distance(
                {"a": Measurement("a", 1.0, -1.0)}, {"a": Measurement("a", 1.0)}
            )


class MeasurementDistanceEdgeTest(unittest.TestCase):
    def test_zero_total_weight(self):
        target = {"a": Measurement("a", 100.0, 0.0)}
        with self.assertRaises(ValueError):
            measurement_distance(target, {"a": Measurement("a", 100.0)})

    def test_name_mismatch(self):
        with self.assertRaises(ValueError):
            measurement_distance({"a": Measurement("a", 1.0)}, {"b": Measurement("b", 1.0)})

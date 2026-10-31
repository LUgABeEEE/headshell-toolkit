import unittest

from core.fit.inner_proxy import HeadMeasurements, fit_reference_head


class InnerProxyTest(unittest.TestCase):
    def test_scale_and_liner(self):
        spec = fit_reference_head(
            (280.0, 277.0, 376.0),
            HeadMeasurements(width_mm=150.0, depth_mm=190.0, face_length_mm=230.0, ipd_mm=64.0, liner_mm=6.0),
        )
        self.assertAlmostEqual(spec.scale[0], 150.0 / 280.0, places=6)
        self.assertAlmostEqual(spec.scale[1], 190.0 / 277.0, places=6)
        self.assertAlmostEqual(spec.scale[2], 230.0 / 376.0, places=6)
        self.assertAlmostEqual(spec.bbox_mm[0], 162.0, places=6)
        self.assertAlmostEqual(spec.bbox_mm[1], 202.0, places=6)
        self.assertAlmostEqual(spec.bbox_mm[2], 242.0, places=6)
        self.assertEqual(spec.liner_mm, 6.0)

    def test_invalid_reference(self):
        with self.assertRaises(ValueError):
            fit_reference_head((0.0, 277.0, 376.0), HeadMeasurements(150, 190, 230, 64, 6))

    def test_invalid_measurements(self):
        with self.assertRaises(ValueError):
            fit_reference_head((280, 277, 376), HeadMeasurements(width_mm=400, depth_mm=190, face_length_mm=230, ipd_mm=64, liner_mm=6))
        with self.assertRaises(ValueError):
            fit_reference_head((280, 277, 376), HeadMeasurements(150, 190, 230, 64, liner_mm=99))

    def test_non_finite(self):
        with self.assertRaises(ValueError):
            fit_reference_head((float("inf"), 277, 376), HeadMeasurements(150, 190, 230, 64, 6))

    def test_circumference_range(self):
        with self.assertRaises(ValueError):
            fit_reference_head((280, 277, 376), HeadMeasurements(150, 190, 230, 64, 6, circumference_mm=100))


if __name__ == "__main__":
    unittest.main()

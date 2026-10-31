import unittest

from core.params import Params, SplitSpec


FULL = dict(ipd_mm=64.0, head_width_mm=220.0, wall_mm=2.5, liner_mm=6.0,
            vent_diameter_mm=5.0, vent_spacing_mm=14.0,
            split=SplitSpec(plane="coronal", offset_mm=-15.0))


class ParamsTest(unittest.TestCase):
    def test_no_defaults_are_shipped(self):
        # The published contract must not carry a buildable parameter set.
        errors = Params().validate()
        self.assertTrue(errors)
        for name in ("ipd_mm", "head_width_mm", "wall_mm"):
            self.assertTrue(any(name in e for e in errors), name)

    def test_fully_specified_is_valid(self):
        self.assertEqual(Params(**FULL).validate(), [])

    def test_range_errors(self):
        p = Params(**{**FULL, "ipd_mm": 20, "wall_mm": 99, "jaw_width": 2.0})
        errors = p.validate()
        self.assertTrue(any("ipd_mm" in e for e in errors))
        self.assertTrue(any("wall_mm" in e for e in errors))
        self.assertTrue(any("jaw_width" in e for e in errors))

    def test_round_trip(self):
        p = Params(**{**FULL, "cheek_fullness": 0.5, "ipd_mm": 66,
                      "split": SplitSpec(plane="coronal", offset_mm=-10)})
        self.assertEqual(Params.from_dict(p.to_dict()), p)

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            Params.from_dict({"schema": "headshell/params@0", "nope": 1})

    def test_fabrication_ranges(self):
        errors = Params(**{**FULL, "liner_mm": 80, "vent_diameter_mm": 99,
                           "vent_spacing_mm": 400}).validate()
        self.assertTrue(any("liner_mm" in e for e in errors))
        self.assertTrue(any("vent_diameter_mm" in e for e in errors))
        self.assertTrue(any("vent_spacing_mm" in e for e in errors))

    def test_invalid_split(self):
        errors = Params(**{**FULL, "split": SplitSpec(plane="diagonal", offset_mm=9000)}).validate()
        self.assertEqual(len(errors), 2)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from source_code.fragment_physics import THORParameters, THORMaterial, air_drag


class TestFragmentPhysics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steel = THORMaterial("physical_data/THOR_materials/steel.csv")
        cls.glass = THORMaterial("physical_data/THOR_materials/glass.csv")
        cls.concrete = THORMaterial("physical_data/THOR_materials/concrete.csv")

    def test_load_csv_parameters(self):
        # Verify steel.csv loads the exact Project THOR TR-47 constants
        self.assertEqual(self.steel.velocity_params, THORParameters(6.399, 0.889, -0.945, 1.262, 0.019))
        self.assertEqual(self.steel.mass_params, THORParameters(-2.507, 0.138, 0.835, 0.143, 0.761))

    def test_manual_initialization_and_validation(self):
        # Verify initializing with explicit THORParameters works and missing args raises ValueError
        vp = THORParameters(6.399, 0.889, -0.945, 1.262, 0.019)
        mp = THORParameters(-2.507, 0.138, 0.835, 0.143, 0.761)
        mat = THORMaterial(velocity_params=vp, mass_params=mp)
        self.assertEqual(mat.velocity_params, vp)
        with self.assertRaises(ValueError):
            THORMaterial()

    def test_zero_or_negative_inputs_return_zero(self):
        self.assertEqual(self.steel.exit_velocity_and_mass(0.0, 0.01, 1e-4, 0.005, 1.0), (0.0, 0.0))
        self.assertEqual(self.steel.exit_velocity_and_mass(-100.0, 0.01, 1e-4, 0.005, 1.0), (0.0, 0.0))
        self.assertEqual(self.steel.exit_velocity_and_mass(1500.0, 0.0, 1e-4, 0.005, 1.0), (0.0, 0.0))
        self.assertEqual(self.steel.exit_velocity_and_mass(1500.0, 0.01, 1e-4, 0.005, 0.0), (0.0, 0.0))

    def test_zero_thickness_preserves_velocity_and_mass(self):
        # Passing through d = 0 m should cause zero velocity loss and zero mass loss
        v_out, m_out = self.steel.exit_velocity_and_mass(1500.0, 0.01, 1e-4, 0.0, 1.0)
        self.assertAlmostEqual(v_out, 1500.0, places=6)
        self.assertAlmostEqual(m_out, 0.01, places=6)

    def test_perforation_reduces_velocity_and_mass(self):
        # A 10g fragment at 1500 m/s perforating 5mm of steel loses both velocity and mass but still exits
        v_out, m_out = self.steel.exit_velocity_and_mass(1500.0, 0.01, 1e-4, 0.005, 1.0)
        self.assertGreater(v_out, 0.0)
        self.assertLess(v_out, 1500.0)
        self.assertGreater(m_out, 0.0)
        self.assertLess(m_out, 0.01)

    def test_exact_imperial_formula_match(self):
        # Verify SI-to-Imperial conversion matches direct calculation in Imperial units
        v_in, m_in, sigma, d, cos_theta = 1200.0, 0.008, 8e-5, 0.004, 0.8660254
        v_out, m_out = self.glass.exit_velocity_and_mass(v_in, m_in, sigma, d, cos_theta)

        v_imp = v_in * 3.28084
        m_imp = m_in * 15432.358
        s_imp = sigma * 1550.0031
        d_imp = d * 39.370079
        expected_v = (v_imp - (10.0 ** 3.743) * ((s_imp * d_imp) ** 0.705) * (m_imp ** -0.723) * (v_imp ** 0.465) / (cos_theta ** 0.690)) / 3.28084
        expected_m = (m_imp - (10.0 ** -5.926) * ((s_imp * d_imp) ** 0.305) * (m_imp ** 0.429) * (v_imp ** 1.819) / (cos_theta ** 0.747)) / 15432.358
        self.assertAlmostEqual(v_out, expected_v, places=6)
        self.assertAlmostEqual(m_out, expected_m, places=6)

    def test_stopped_fragment_returns_zero_velocity_and_mass(self):
        # A slow/light fragment hitting a thick 10cm steel plate is completely stopped (v_out = 0, m_out = 0)
        v_out, m_out = self.steel.exit_velocity_and_mass(200.0, 0.002, 1e-4, 0.10, 1.0)
        self.assertEqual((v_out, m_out), (0.0, 0.0))

    def test_increasing_thickness_monotonicity(self):
        # Thicker plate must result in lower (or equal) exit velocity and residual mass
        v1, m1 = self.concrete.exit_velocity_and_mass(1500.0, 0.01, 1e-4, 0.02, 1.0)
        v2, m2 = self.concrete.exit_velocity_and_mass(1500.0, 0.01, 1e-4, 0.05, 1.0)
        self.assertLess(v2, v1)
        self.assertLess(m2, m1)

    def test_oblique_impact_increases_resistance(self):
        # Entering at 60 degrees (cos = 0.5) reduces exit velocity and residual mass more than normal impact (cos = 1.0)
        v_norm, m_norm = self.steel.exit_velocity_and_mass(1500.0, 0.01, 1e-4, 0.003, 1.0)
        v_oblq, m_oblq = self.steel.exit_velocity_and_mass(1500.0, 0.01, 1e-4, 0.003, 0.5)
        self.assertLess(v_oblq, v_norm)
        self.assertLess(m_oblq, m_norm)

    def test_heavier_fragment_retains_more_velocity(self):
        # For the same presented area sigma and initial velocity, a heavier fragment loses less velocity
        v_light, _ = self.steel.exit_velocity_and_mass(1500.0, 0.005, 1e-4, 0.004, 1.0)
        v_heavy, _ = self.steel.exit_velocity_and_mass(1500.0, 0.020, 1e-4, 0.004, 1.0)
        self.assertGreater(v_heavy, v_light)

    def test_air_drag_zero_distance_or_density(self):
        # Zero distance x=0 or zero air density rho=0 leaves velocity unchanged
        self.assertAlmostEqual(air_drag(1000.0, 1.0, 1.225, 7850.0, 1.2, 0.01, 0.0), 1000.0, places=6)
        self.assertAlmostEqual(air_drag(1000.0, 1.0, 0.0, 7850.0, 1.2, 0.01, 50.0), 1000.0, places=6)

    def test_air_drag_zero_or_negative_mass_and_velocity(self):
        self.assertEqual(air_drag(0.0, 1.0, 1.225, 7850.0, 1.2, 0.01, 10.0), 0.0)
        self.assertEqual(air_drag(1000.0, 1.0, 1.225, 7850.0, 1.2, 0.0, 10.0), 0.0)

    def test_air_drag_distance_and_mass_scaling(self):
        # Velocity decays exponentially with distance, and heavier fragments decelerate more slowly in air
        v_10m = air_drag(1500.0, 1.0, 1.225, 7850.0, 1.2, 0.01, 10.0)
        v_20m = air_drag(1500.0, 1.0, 1.225, 7850.0, 1.2, 0.01, 20.0)
        self.assertLess(v_20m, v_10m)
        # Check exponential property: v(20m) / v_in == (v(10m) / v_in)^2
        self.assertAlmostEqual(v_20m / 1500.0, (v_10m / 1500.0) ** 2, places=6)

        v_heavy_10m = air_drag(1500.0, 1.0, 1.225, 7850.0, 1.2, 0.05, 10.0)
        self.assertGreater(v_heavy_10m, v_10m)


if __name__ == "__main__":
    unittest.main()

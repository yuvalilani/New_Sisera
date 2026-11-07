import unittest
import numpy as np
import matplotlib.tri as mtri

from source_code.fragment_physics import THORMaterial, air_drag
from source_code.ray_tracing import RayTracer
from source_code.pkill import PKillTable
from source_code.pkill_map import PKillMap2D, convex_hull_2d


def _polygon_area(poly: np.ndarray) -> float:
    return 0.5 * float(abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))))


class TestDeterministicPKill(unittest.TestCase):
    """Exact, deterministic tests for single-ray pkill calculations and 2D target polygon projection."""

    @classmethod
    def setUpClass(cls):
        cls.rt_ball = RayTracer("tests/models/steel_covered_ball.obj")
        cls.rt_double_wall = RayTracer("tests/models/double_wall.obj")
        cls.pkill_table = PKillTable("physical_data/pkill/death.csv")
        cls.ball_map = PKillMap2D(cls.rt_ball, cls.pkill_table, [0.0, 0.0, 0.0], "Target", 0.06, 2300.0, threshold=0.05)

    def test_steel_covered_ball_ray_sections(self):
        center_sections = self.rt_ball.get_path_sections([0.0, 0.0, 0.0], [3.0, 0.0, 0.0])
        self.assertEqual([s.object_name for s in center_sections], ["Air", "Steel", "Target", "Steel", "Air"])
        self.assertAlmostEqual(center_sections[1].length, 0.02, delta=0.003)
        self.assertLess(center_sections[1].entry_angle, 5.0)

        oblique_sections = self.rt_ball.get_path_sections([0.0, 0.0, 0.0], [4.0, 0.0, 0.70])
        self.assertIn("Target", [s.object_name for s in oblique_sections])
        self.assertGreater(oblique_sections[1].length, center_sections[1].length * 1.25)
        self.assertGreater(oblique_sections[1].entry_angle, 35.0)

    def test_exact_single_ray_pkill_vs_direction_on_ball(self):
        # Exact reference outputs for m0=0.06 kg, v0=2300.0 m/s across specific (phi, cos(theta)) directions
        expected = [
            (0.0, 0.0, 0.4912586177),
            (0.0, 0.08, 0.4682580012),
            (0.0, 0.14, 0.3716440896),
            (0.10, -0.10, 0.3748663669),
            (0.0, 0.24, 0.0),
            (0.5, 0.5, 0.0),
        ]
        for phi, cos_theta, exp_p in expected:
            actual_p = self.ball_map.calculate_direction_pkill(phi, cos_theta)
            self.assertAlmostEqual(actual_p, exp_p, places=6, msg=f"Mismatch at phi={phi}, cos_theta={cos_theta}")

    def test_exact_single_ray_pkill_vs_initial_mass_and_velocity(self):
        # Exact reference outputs at ball center (phi=0, cos(theta)=0) for various (mass, velocity) pairs
        expected = [
            (0.06, 1500.0, 0.3045242357),
            (0.06, 2000.0, 0.4344573154),
            (0.06, 2500.0, 0.4663523169),
            (0.06, 3000.0, 0.0),
            (0.12, 3000.0, 0.4696641446),
        ]
        for m0, v0, exp_p in expected:
            pm = PKillMap2D(self.rt_ball, self.pkill_table, [0.0, 0.0, 0.0], "Target", m0, v0, threshold=0.2)
            self.assertAlmostEqual(pm.calculate_direction_pkill(0.0, 0.0), exp_p, places=6, msg=f"Mismatch for m0={m0}, v0={v0}")

    def test_double_wall_matches_analytical_step_by_step_chain(self):
        # Explosion at [0, -3, 1.5] towards "Steel" wall (y in [1, 2]) behind "Concrete" wall (y in [-2, -1])
        dw_map = PKillMap2D(self.rt_double_wall, self.pkill_table, [0.0, -3.0, 1.5], "Steel", 0.50, 2500.0, threshold=0.2)
        concrete = THORMaterial("physical_data/THOR_materials/concrete.csv")

        # Perpendicular ray along +Y (phi = pi/2, cos_theta = 0): 1m Air -> 1m Concrete (0 deg) -> 2m Air -> Steel
        v1 = air_drag(2500.0, 1.0, 1.225, 7850.0, 1.2, 0.50, 1.0)
        sigma = 1.2 * ((0.50 / 7850.0) ** (2.0 / 3.0))
        v2, m2 = concrete.exit_velocity_and_mass(v1, 0.50, sigma, 1.0, 1.0)
        v3 = air_drag(v2, 1.0, 1.225, 7850.0, 1.2, m2, 2.0)
        expected_perp = float(self.pkill_table.get_probability(m2, v3))

        actual_perp = dw_map.calculate_direction_pkill(float(np.pi / 2.0), 0.0)
        self.assertAlmostEqual(actual_perp, expected_perp, places=6)
        self.assertAlmostEqual(actual_perp, 0.5730076452, places=6)

        # Oblique ray at (phi = pi/2 - 0.3, cos_theta = 0.1)
        actual_oblq = dw_map.calculate_direction_pkill(float(np.pi / 2.0 - 0.3), 0.1)
        self.assertAlmostEqual(actual_oblq, 0.5671294628, places=6)

    def test_target_polygon_projection_and_wrap_around(self):
        # Verify exact projected polygon area and bounds for explosion at origin
        self.assertAlmostEqual(_polygon_area(self.ball_map.polygon), 0.19853081, places=6)
        self.assertAlmostEqual(float(np.min(self.ball_map.polygon[:, 0])), -0.25209119, places=6)
        self.assertAlmostEqual(float(np.max(self.ball_map.polygon[:, 0])), 0.25261793, places=6)

        # Explosion at [4, 0, 0] looking in -X direction (where raw atan2 jumps across +/- pi) must unwrap cleanly
        back_map = PKillMap2D(self.rt_ball, self.pkill_table, [4.0, 0.0, 0.0], "Target", 0.06, 2300.0, threshold=0.2)
        self.assertAlmostEqual(_polygon_area(back_map.polygon), 0.19853081, places=5)

    def test_invalid_target_name_raises_value_error(self):
        with self.assertRaises(ValueError):
            PKillMap2D(self.rt_ball, self.pkill_table, [0.0, 0.0, 0.0], "NonExistentTarget", 0.06, 2300.0)


class TestPKillMapConvergence(unittest.TestCase):
    """Convergence-dependent tests for the 2D adaptive pkill map (independent of specific mesh layout)."""

    @classmethod
    def setUpClass(cls):
        cls.rt_ball = RayTracer("tests/models/steel_covered_ball.obj")
        cls.pkill_table = PKillTable("physical_data/pkill/death.csv")
        cls.coarse_map = PKillMap2D(cls.rt_ball, cls.pkill_table, [0.0, 0.0, 0.0], "Target", 0.06, 2300.0, threshold=0.15)
        cls.fine_map = PKillMap2D(cls.rt_ball, cls.pkill_table, [0.0, 0.0, 0.0], "Target", 0.06, 2300.0, threshold=0.01)

    def test_tighter_threshold_improves_interpolation_accuracy(self):
        def compute_grid_mae(pm: PKillMap2D) -> float:
            plot_pts = np.vstack([pm.points, pm.polygon])
            plot_probs = np.concatenate([pm.probabilities, np.zeros(len(pm.polygon))])
            interp = mtri.LinearTriInterpolator(mtri.Triangulation(plot_pts[:, 0], plot_pts[:, 1]), plot_probs)
            errs = []
            for r in np.linspace(0.02, 0.22, 8):
                for a in np.linspace(0.2, 5.8, 8):
                    phi, ct = float(r * np.cos(a)), float(r * np.sin(a))
                    errs.append(abs(pm.calculate_direction_pkill(phi, ct) - float(interp(phi, ct))))
            return float(np.mean(errs))

        mae_coarse = compute_grid_mae(self.coarse_map)
        mae_fine = compute_grid_mae(self.fine_map)

        self.assertGreater(len(self.fine_map.points), len(self.coarse_map.points))
        self.assertLess(mae_fine, mae_coarse * 0.5)
        self.assertLess(mae_fine, 0.015)

    def test_converged_2d_integral_matches_reference(self):
        # The integrated pkill over the 2D (phi, cos(theta)) polygon should converge to ~0.0421 regardless of triangulation details
        plot_pts = np.vstack([self.fine_map.points, self.fine_map.polygon])
        plot_probs = np.concatenate([self.fine_map.probabilities, np.zeros(len(self.fine_map.polygon))])
        tri = mtri.Triangulation(plot_pts[:, 0], plot_pts[:, 1])
        v0, v1, v2 = plot_pts[tri.triangles[:, 0]], plot_pts[tri.triangles[:, 1]], plot_pts[tri.triangles[:, 2]]
        areas = 0.5 * np.abs((v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1]))
        integral = float(np.sum(areas * np.mean(plot_probs[tri.triangles], axis=1)))
        self.assertAlmostEqual(integral, 0.0421, delta=0.004)

    def test_sampled_points_spacing_and_quadrant_symmetry(self):
        pts = self.fine_map.points
        probs = self.fine_map.probabilities

        # Pairwise distances between sampled points must respect min_dist
        diff = pts[:, None, :] - pts[None, :, :]
        dist_matrix = np.linalg.norm(diff, axis=-1) + np.eye(len(pts))
        self.assertGreaterEqual(float(np.min(dist_matrix)), self.fine_map.min_dist - 1e-9)

        # Average pkill across the 4 quadrants around (0, 0) should be symmetric
        q1 = float(np.mean(probs[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]))
        q2 = float(np.mean(probs[(pts[:, 0] < 0) & (pts[:, 1] >= 0)]))
        q3 = float(np.mean(probs[(pts[:, 0] < 0) & (pts[:, 1] < 0)]))
        q4 = float(np.mean(probs[(pts[:, 0] >= 0) & (pts[:, 1] < 0)]))
        for q in (q2, q3, q4):
            self.assertAlmostEqual(q1, q, delta=0.05)


if __name__ == "__main__":
    unittest.main()

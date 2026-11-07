import unittest
import numpy as np
from source_code.ray_tracing import RayTracer, PathSection


class TestGetPathSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wall = RayTracer("tests/models/wall.obj")
        cls.double_wall = RayTracer("tests/models/double_wall.obj")
        cls.layered_wall = RayTracer("tests/models/layered_wall.obj")
        cls.nested_box = RayTracer("tests/models/nested_box.obj")
        cls.houses = RayTracer("tests/models/houses_and_people.obj")

    def assertSectionsMatch(self, actual: list[PathSection], expected: list[tuple[float, str, float]], places: int = 4):
        self.assertEqual(len(actual), len(expected), f"Expected {len(expected)} sections, got {len(actual)}: {actual}")
        for idx, (sec, (exp_len, exp_name, exp_angle)) in enumerate(zip(actual, expected)):
            self.assertEqual(sec.object_name, exp_name, f"Section {idx}: expected name '{exp_name}', got '{sec.object_name}'")
            self.assertAlmostEqual(sec.length, exp_len, places=places, msg=f"Section {idx}: length mismatch")
            self.assertAlmostEqual(sec.entry_angle, exp_angle, places=places, msg=f"Section {idx}: entry_angle mismatch")

    def test_miss_all_objects(self):
        # Ray passes completely above the wall (height=3.0)
        sections = self.wall.get_path_sections([0.0, -2.0, 5.0], [0.0, 2.0, 5.0])
        self.assertSectionsMatch(sections, [(4.0, "Air", 0.0)])

    def test_zero_length_ray(self):
        # Start and end points are identical
        sections = self.wall.get_path_sections([0.0, 0.0, 1.0], [0.0, 0.0, 1.0])
        self.assertEqual(sections, [])

    def test_perpendicular_entry(self):
        # Ray passes perpendicular (0 degrees) through wall (y in [-0.2, 0.2])
        sections = self.wall.get_path_sections([0.0, -2.0, 1.5], [0.0, 2.0, 1.5])
        self.assertSectionsMatch(sections, [(1.8, "Air", 0.0), (0.4, "Concrete", 0.0), (1.8, "Air", 0.0)])

    def test_45_degree_entry(self):
        # Ray enters at 45 degrees in the X-Y plane
        sections = self.wall.get_path_sections([-2.0, -2.0, 1.5], [2.0, 2.0, 1.5])
        s2 = float(np.sqrt(2.0))
        self.assertSectionsMatch(sections, [(1.8 * s2, "Air", 0.0), (0.4 * s2, "Concrete", 45.0), (1.8 * s2, "Air", 0.0)])

    def test_60_degree_entry(self):
        # Ray enters at 60 degrees relative to surface normal (dx = sqrt(3) * dy)
        s3 = float(np.sqrt(3.0))
        sections = self.wall.get_path_sections([-2.0 * s3, -2.0, 1.5], [2.0 * s3, 2.0, 1.5])
        self.assertSectionsMatch(sections, [(3.6, "Air", 0.0), (0.8, "Concrete", 60.0), (3.6, "Air", 0.0)])

    def test_3d_diagonal_oblique_angle(self):
        # Ray moves simultaneously in X, Y, and Z: dx=2, dy=4, dz=2 (direction = [1, 2, 1] / sqrt(6))
        sections = self.wall.get_path_sections([-1.0, -2.0, 0.5], [1.0, 2.0, 2.5])
        s6 = float(np.sqrt(6.0))
        expected_angle = float(np.degrees(np.arccos(2.0 / s6)))
        self.assertSectionsMatch(sections, [(0.9 * s6, "Air", 0.0), (0.2 * s6, "Concrete", expected_angle), (0.9 * s6, "Air", 0.0)])

    def test_vertical_z_axis_traversal(self):
        # Ray goes straight down through the top face (z=3) and out the bottom face (z=0) of the wall
        sections = self.wall.get_path_sections([0.0, 0.0, 4.0], [0.0, 0.0, -1.0])
        self.assertSectionsMatch(sections, [(1.0, "Air", 0.0), (3.0, "Concrete", 0.0), (1.0, "Air", 0.0)])

    def test_lengthwise_x_axis_traversal(self):
        # Ray goes lengthwise through the side faces of the wall from x=-6 to x=6 (wall x in [-5, 5])
        sections = self.wall.get_path_sections([-6.0, 0.0, 1.5], [6.0, 0.0, 1.5])
        self.assertSectionsMatch(sections, [(1.0, "Air", 0.0), (10.0, "Concrete", 0.0), (1.0, "Air", 0.0)])

    def test_start_inside_material_and_exit(self):
        # Start at y = 0.0 (inside Concrete wall [-0.2, 0.2]) and exit to y = 2.0
        sections = self.wall.get_path_sections([0.0, 0.0, 1.5], [0.0, 2.0, 1.5])
        self.assertSectionsMatch(sections, [(0.2, "Concrete", 0.0), (1.8, "Air", 0.0)])

    def test_start_and_end_inside_same_material(self):
        # Both points are completely inside the Concrete wall ([-0.2, 0.2]) with no surface intersections between them
        sections = self.wall.get_path_sections([-1.0, 0.0, 1.5], [1.0, 0.0, 1.5])
        self.assertSectionsMatch(sections, [(2.0, "Concrete", 0.0)])

    def test_enter_and_end_inside_material(self):
        # Start in Air at y = -2.0 and stop inside the Concrete wall at y = 0.0
        sections = self.wall.get_path_sections([0.0, -2.0, 1.5], [0.0, 0.0, 1.5])
        self.assertSectionsMatch(sections, [(1.8, "Air", 0.0), (0.2, "Concrete", 0.0)])

    def test_enter_at_angle_and_end_inside_material(self):
        # Start in Air and stop inside the wall at a 45-degree angle
        sections = self.wall.get_path_sections([-2.0, -2.0, 1.5], [0.0, 0.0, 1.5])
        s2 = float(np.sqrt(2.0))
        self.assertSectionsMatch(sections, [(1.8 * s2, "Air", 0.0), (0.2 * s2, "Concrete", 45.0)])

    def test_start_and_end_exactly_on_boundaries(self):
        # Start directly on the front face (y = -0.2) and end directly on the back face (y = 0.2)
        sections = self.wall.get_path_sections([0.0, -0.2, 1.5], [0.0, 0.2, 1.5])
        self.assertSectionsMatch(sections, [(0.4, "Concrete", 0.0)])

    def test_start_on_boundary_and_exit_into_air(self):
        # Start directly on the entry boundary (y = -0.2) and continue out past the back boundary (y = 0.2) into Air (y = 2.0)
        sections = self.wall.get_path_sections([0.0, -0.2, 1.5], [0.0, 2.0, 1.5])
        self.assertSectionsMatch(sections, [(0.4, "Concrete", 0.0), (1.8, "Air", 0.0)])

    def test_start_on_boundary_corner_vertex_and_exit(self):
        # Start directly on a corner vertex (-5.0, -0.2, 0.0) of the wall and pass through to (0.0, 2.0, 1.5)
        sections = self.wall.get_path_sections([-5.0, -0.2, 0.0], [0.0, 2.0, 1.5])
        self.assertEqual([s.object_name for s in sections], ["Concrete", "Air"])

    def test_multiple_separated_layers(self):
        # Pass through Concrete wall ([-2, -1]), Air gap ([-1, 1]), and Steel wall ([1, 2])
        sections = self.double_wall.get_path_sections([0.0, -3.0, 1.5], [0.0, 3.0, 1.5])
        self.assertSectionsMatch(sections, [(1.0, "Air", 0.0), (1.0, "Concrete", 0.0), (2.0, "Air", 0.0), (1.0, "Steel", 0.0), (1.0, "Air", 0.0)])

    def test_reverse_direction_symmetry(self):
        # Reversing start and end on a symmetric normal-incidence path reverses the section order
        forward = self.double_wall.get_path_sections([0.0, -3.0, 1.5], [0.0, 3.0, 1.5])
        backward = self.double_wall.get_path_sections([0.0, 3.0, 1.5], [0.0, -3.0, 1.5])
        expected_backward = [(s.length, s.object_name, s.entry_angle) for s in reversed(forward)]
        self.assertSectionsMatch(backward, expected_backward)

    def test_start_inside_first_wall_end_inside_second_wall(self):
        # Start inside Concrete wall (y = -1.5) and end inside Steel wall (y = 1.5)
        sections = self.double_wall.get_path_sections([0.0, -1.5, 1.5], [0.0, 1.5, 1.5])
        self.assertSectionsMatch(sections, [(0.5, "Concrete", 0.0), (2.0, "Air", 0.0), (0.5, "Steel", 0.0)])

    def test_flush_touching_layers(self):
        # Pass through flush Glass ([-1, 0]) and Wood ([0, 1]) layers with no air gap between them at 45 degrees
        sections = self.layered_wall.get_path_sections([-2.0, -2.0, 1.5], [2.0, 2.0, 1.5])
        s2 = float(np.sqrt(2.0))
        self.assertSectionsMatch(sections, [(1.0 * s2, "Air", 0.0), (1.0 * s2, "Glass", 45.0), (1.0 * s2, "Wood", 45.0), (1.0 * s2, "Air", 0.0)])

    def test_flush_layers_reverse_direction(self):
        # Pass through Wood ([0, 1]) then Glass ([-1, 0]) in reverse
        sections = self.layered_wall.get_path_sections([0.0, 2.0, 1.5], [0.0, -2.0, 1.5])
        self.assertSectionsMatch(sections, [(1.0, "Air", 0.0), (1.0, "Wood", 0.0), (1.0, "Glass", 0.0), (1.0, "Air", 0.0)])

    def test_nested_materials_full_traversal(self):
        # Outer Concrete ([-2, 2]) enclosing inner Lead core ([-1, 1]) at z = 1.0
        sections = self.nested_box.get_path_sections([0.0, 5.0, 1.0], [0.0, -5.0, 1.0])
        self.assertSectionsMatch(sections, [(3.0, "Air", 0.0), (1.0, "Concrete", 0.0), (2.0, "Lead", 0.0), (1.0, "Concrete", 0.0), (3.0, "Air", 0.0)])

    def test_nested_materials_angled_traversal(self):
        # Enter outer Concrete and inner Lead at 45 degrees in the X-Y plane
        sections = self.nested_box.get_path_sections([-2.5, -3.0, 2.0], [3.5, 3.0, 2.0])
        s2 = float(np.sqrt(2.0))
        self.assertSectionsMatch(sections, [(1.0 * s2, "Air", 0.0), (1.0 * s2, "Concrete", 45.0), (1.5 * s2, "Lead", 45.0), (1.0 * s2, "Concrete", 45.0), (1.5 * s2, "Air", 0.0)])

    def test_grazing_inner_surface(self):
        # At z = 0.5, the ray skims the exact bottom face of the Lead core without entering its interior
        sections = self.nested_box.get_path_sections([0.0, 5.0, 0.5], [0.0, -5.0, 0.5])
        self.assertSectionsMatch(sections, [(3.0, "Air", 0.0), (4.0, "Concrete", 0.0), (3.0, "Air", 0.0)])

    def test_start_inside_nested_inner_core(self):
        # Start at the center of the inner Lead core (0, 0, 2) and exit through Concrete and into Air
        sections = self.nested_box.get_path_sections([0.0, 0.0, 2.0], [0.0, 3.0, 2.0])
        self.assertSectionsMatch(sections, [(1.0, "Lead", 0.0), (1.0, "Concrete", 0.0), (1.0, "Air", 0.0)])

    def test_start_in_outer_shell_end_in_inner_core(self):
        # Start in the outer Concrete shell (y = -1.5) and end inside the Lead core (y = 0.0)
        sections = self.nested_box.get_path_sections([0.0, -1.5, 2.0], [0.0, 0.0, 2.0])
        self.assertSectionsMatch(sections, [(0.5, "Concrete", 0.0), (1.0, "Lead", 0.0)])

    def test_custom_free_space_name(self):
        # Verify that custom free_space_name (e.g., 'Vacuum') is used instead of 'Air'
        sections = self.wall.get_path_sections([0.0, -1.0, 1.5], [0.0, 1.0, 1.5], free_space_name="Vacuum")
        self.assertSectionsMatch(sections, [(0.8, "Vacuum", 0.0), (0.4, "Concrete", 0.0), (0.8, "Vacuum", 0.0)])

    def test_houses_and_people_complex_mesh(self):
        # Sanity check on the full houses_and_people model: sum of section lengths must equal total ray distance
        p0 = np.array([7.5, -5.0, 2.0])
        p1 = np.array([7.5, 20.0, 2.0])
        sections = self.houses.get_path_sections(p0, p1)
        total_len = sum(s.length for s in sections)
        self.assertAlmostEqual(total_len, float(np.linalg.norm(p1 - p0)), places=4)
        self.assertSectionsMatch(sections, [(5.0, "Air", 0.0), (0.2, "Concrete", 0.0), (3.7, "Air", 0.0), (0.2, "Concrete", 0.0), (6.8, "Air", 0.0), (0.2, "Concrete", 0.0), (8.9, "Air", 0.0)])

    def test_houses_corner_start_diagonal_to_person(self):
        # Ray starting at building corner [0, 0, 0] and ending inside Person_2 at [14, 1, 9]
        p0 = np.array([0.0, 0.0, 0.0])
        p1 = np.array([14.0, 1.0, 9.0])
        sections = self.houses.get_path_sections(p0, p1)
        self.assertEqual([s.object_name for s in sections], ["Concrete", "Air", "Concrete", "Air", "Concrete", "Air", "Concrete", "Air", "Concrete", "Air", "Person_2"])
        self.assertAlmostEqual(sum(s.length for s in sections), float(np.linalg.norm(p1 - p0)), places=4)

    def test_separate_people_boxes_in_houses_model(self):
        # Ray passing vertically from z=4.0 to z=10.0 at (x=13.94, y=0.91) hits Person_1 (z in [4.75, 5.75]), Concrete floor (z in [7.4, 7.6]), and Person_2 (z in [8.27, 9.27])
        sections = self.houses.get_path_sections([13.94, 0.91, 4.0], [13.94, 0.91, 10.0])
        person_names = [s.object_name for s in sections if s.object_name.startswith("Person_")]
        self.assertEqual(person_names, ["Person_1", "Person_2"])


if __name__ == "__main__":
    unittest.main()

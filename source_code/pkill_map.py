import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from source_code.fragment_physics import THORMaterial, air_drag
from source_code.pkill import PKillTable
from source_code.ray_tracing import RayTracer


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    """Compute the counter-clockwise 2D convex hull of a set of 2D points."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def build_half(p_list: np.ndarray) -> list[np.ndarray]:
        hull = []
        for p in p_list:
            while len(hull) >= 2 and (hull[-1][0] - hull[-2][0]) * (p[1] - hull[-2][1]) - (hull[-1][1] - hull[-2][1]) * (p[0] - hull[-2][0]) <= 0.0:
                hull.pop()
            hull.append(p)
        return hull[:-1]

    return np.array(build_half(pts) + build_half(pts[::-1]), dtype=float)


class PKillMap2D:
    def __init__(self, ray_tracer: RayTracer, pkill_table: PKillTable, explosion_point: list[float] | np.ndarray, target_name: str, initial_mass: float, initial_velocity: float, materials: dict[str, THORMaterial] | None = None, threshold: float = 0.01, c_d: float = 1.0, air_density: float = 1.225, fragment_density: float = 7850.0, shape_factor: float = 1.2, min_dist: float = 0.02, max_points: int = 300):
        self.ray_tracer = ray_tracer
        self.pkill_table = pkill_table
        self.explosion_point = np.asarray(explosion_point, dtype=float)
        self.target_name = target_name
        self.initial_mass = float(initial_mass)
        self.initial_velocity = float(initial_velocity)
        self.threshold = float(threshold)
        self.c_d = float(c_d)
        self.air_density = float(air_density)
        self.fragment_density = float(fragment_density)
        self.shape_factor = float(shape_factor)
        self.min_dist = float(min_dist)
        self.max_points = int(max_points)

        if materials is None:
            materials = {name: THORMaterial(f"physical_data/THOR_materials/{name}.csv") for name in ("steel", "concrete", "glass")}
        self.materials = {k.lower(): v for k, v in materials.items()}

        self.target_sub = next((sub for name, sub, _ in self.ray_tracer.submeshes if name == self.target_name), None)
        if self.target_sub is None:
            raise ValueError(f"Target '{self.target_name}' not found in RayTracer submeshes.")

        self.ref_phi = 0.0
        self.ray_length = self.ray_tracer.max_span
        self.polygon = self._build_target_polygon()
        self.points, self.probabilities = self._build_adaptive_map()

    def _build_target_polygon(self) -> np.ndarray:
        """Project target mesh vertices onto (phi, cos(theta)) and return their 2D convex hull polygon."""
        vecs = np.asarray(self.target_sub.points, dtype=float) - self.explosion_point
        dists = np.linalg.norm(vecs, axis=1)
        unit_vecs = vecs / dists[:, None]
        mean_vec = np.mean(unit_vecs, axis=0)
        self.ref_phi = float(np.arctan2(mean_vec[1], mean_vec[0]))
        self.ray_length = float(np.max(dists)) + 0.5

        raw_phi = np.arctan2(unit_vecs[:, 1], unit_vecs[:, 0])
        phi = self.ref_phi + np.arctan2(np.sin(raw_phi - self.ref_phi), np.cos(raw_phi - self.ref_phi))
        cos_theta = np.clip(unit_vecs[:, 2], -1.0, 1.0)
        return convex_hull_2d(np.column_stack([phi, cos_theta]))

    def calculate_direction_pkill(self, phi: float, cos_theta: float) -> float:
        """Compute pkill probability for a fragment launched in direction (phi, cos(theta))."""
        cos_t = float(np.clip(cos_theta, -1.0, 1.0))
        sin_t = float(np.sqrt(max(0.0, 1.0 - cos_t * cos_t)))
        direction = np.array([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], dtype=float)

        hit_pt, _ = self.target_sub.ray_trace(self.explosion_point, self.explosion_point + direction * self.ray_length, first_point=True)
        if len(hit_pt) == 0:
            return 0.0

        sections = self.ray_tracer.get_path_sections(self.explosion_point, hit_pt + direction * 1e-4)
        v, m = self.initial_velocity, self.initial_mass
        for sec in sections:
            if sec.object_name == self.target_name:
                return float(self.pkill_table.get_probability(m, v)) if v > 0.0 and m > 0.0 else 0.0
            if sec.object_name == "Air":
                v = air_drag(v, self.c_d, self.air_density, self.fragment_density, self.shape_factor, m, sec.length)
            else:
                sigma = self.shape_factor * ((m / self.fragment_density) ** (2.0 / 3.0))
                v, m = self.materials[sec.object_name.lower()].exit_velocity_and_mass(v, m, sigma, sec.length, float(np.cos(np.radians(sec.entry_angle))))
            if v <= 0.0 or m <= 0.0:
                return 0.0
        return 0.0

    def _build_adaptive_map(self) -> tuple[np.ndarray, np.ndarray]:
        """Adaptively sample interior points by subdividing triangles whose vertices differ by more than threshold."""
        center = np.mean(self.polygon, axis=0)
        ring = self.polygon[::max(1, len(self.polygon) // 8)]
        pts = np.array([center] + [center + 0.5 * (v - center) for v in ring], dtype=float)
        probs = np.array([self.calculate_direction_pkill(float(p[0]), float(p[1])) for p in pts], dtype=float)
        poly_zeros = np.zeros(len(self.polygon), dtype=float)

        for _ in range(8):
            if len(pts) >= self.max_points:
                break
            all_pts = np.vstack([pts, self.polygon])
            all_probs = np.concatenate([probs, poly_zeros])
            tri = mtri.Triangulation(all_pts[:, 0], all_pts[:, 1])

            new_pts = []
            for i, j, k in tri.triangles:
                if max(all_probs[i], all_probs[j], all_probs[k]) - min(all_probs[i], all_probs[j], all_probs[k]) <= self.threshold:
                    continue
                pairs = [(i, j), (j, k), (k, i)]
                a, b = max(pairs, key=lambda e: float(np.linalg.norm(all_pts[e[0]] - all_pts[e[1]])) if abs(all_probs[e[0]] - all_probs[e[1]]) > self.threshold else 0.0)
                if float(np.linalg.norm(all_pts[a] - all_pts[b])) < self.min_dist * 2.0:
                    continue
                mid = 0.5 * (all_pts[a] + all_pts[b])
                if float(np.min(np.linalg.norm(all_pts - mid, axis=1))) >= self.min_dist:
                    if not new_pts or float(np.min(np.linalg.norm(np.array(new_pts) - mid, axis=1))) >= self.min_dist:
                        new_pts.append(mid)
                        if len(pts) + len(new_pts) >= self.max_points:
                            break
            if not new_pts:
                break
            new_arr = np.array(new_pts, dtype=float)
            new_probs = np.array([self.calculate_direction_pkill(float(p[0]), float(p[1])) for p in new_arr], dtype=float)
            pts = np.vstack([pts, new_arr])
            probs = np.concatenate([probs, new_probs])

        return pts, probs

    def show(self) -> None:
        """Display the 2D pkill heatmap as a function of phi and cos(theta), adding p=0 boundary points for plotting."""
        plot_pts = np.vstack([self.points, self.polygon])
        plot_probs = np.concatenate([self.probabilities, np.zeros(len(self.polygon), dtype=float)])
        closed_poly = np.vstack([self.polygon, self.polygon[0]])
        vmax = max(float(np.max(self.probabilities)), 0.1)

        fig, ax = plt.subplots(figsize=(8, 6.5))
        tri = mtri.Triangulation(plot_pts[:, 0], plot_pts[:, 1])
        cf = ax.tricontourf(tri, plot_probs, levels=np.linspace(0.0, vmax, 51), cmap="inferno")
        ax.plot(closed_poly[:, 0], closed_poly[:, 1], color="cyan", linewidth=1.5, label="Target Polygon (p=0)")
        ax.scatter(self.points[:, 0], self.points[:, 1], c="white", edgecolors="black", s=16, linewidths=0.5, label=f"Sampled Points (N={len(self.points)})")
        ax.set_xlabel(r"$\phi$ (rad)")
        ax.set_ylabel(r"$\cos(\theta)$")
        ax.set_title(r"2D $P_{\mathrm{kill}}(\phi, \cos\theta)$ Heatmap")
        ax.legend(loc="upper right")
        fig.colorbar(cf, ax=ax, label="Probability")
        plt.tight_layout()
        plt.show()

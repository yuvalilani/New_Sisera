from dataclasses import dataclass
import numpy as np
import pyvista as pv


@dataclass
class PathSection:
    length: float
    object_name: str
    entry_angle: float


class RayTracer:
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Load the OBJ file and convert all polygons into triangles
        self.mesh = pv.read(file_path).triangulate()
        # Compute outward-facing normal vectors for each triangle face
        self.mesh = self.mesh.compute_normals(cell_normals=True, point_normals=False, auto_orient_normals=False)
        self.normals = np.asarray(self.mesh.cell_data["Normals"], dtype=float)
        # Map each triangle face index to its material or object name from the OBJ file
        self.cell_names = self._load_cell_names()
        # Pre-extract sub-meshes per object/material so coincident interfaces between different objects are never dropped
        self.submeshes = self._build_submeshes()
        # Compute diagonal length of bounding box for ray casts that determine enclosing materials
        b = self.mesh.bounds
        self.max_span = float(np.linalg.norm([b[1] - b[0], b[3] - b[2], b[5] - b[4]])) * 2.0 + 10.0

    def _load_cell_names(self) -> list[str]:
        """Parse the OBJ file to assign an object/material name to every triangle face."""
        names = []
        current_name = "Object"

        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # Update current name whenever a new object ('o'), group ('g'), or material ('usemtl') is declared
                if parts[0] in ("o", "g", "usemtl") and len(parts) > 1:
                    current_name = " ".join(parts[1:])
                # Each face ('f') with N vertices gets split into (N - 2) triangles during triangulation
                elif parts[0] == "f":
                    n_triangles = max(1, len(parts) - 3)
                    names.extend([current_name] * n_triangles)

        return names

    def _build_submeshes(self) -> list[tuple[str, pv.PolyData, np.ndarray]]:
        """Group triangle cells by object/material name into separate PolyData sub-meshes."""
        submeshes = []
        unique_names = list(dict.fromkeys(self.cell_names))
        cell_names_arr = np.asarray(self.cell_names, dtype=object)

        for name in unique_names:
            mask = cell_names_arr == name
            sub = self.mesh.extract_cells(mask).extract_surface(algorithm=None)
            sub_normals = np.asarray(sub.cell_data["Normals"], dtype=float)
            submeshes.append((name, sub, sub_normals))

        return submeshes

    def _is_point_inside(self, sub: pv.PolyData, sub_normals: np.ndarray, point: np.ndarray, tol: float) -> bool:
        """Check if a 3D point is strictly inside a closed submesh by checking the first surface hit along a probe ray."""
        probe_dir = np.array([0.37139067, 0.55708601, 0.74278135], dtype=float)
        probe_dir /= float(np.linalg.norm(probe_dir))
        pts, cids = sub.ray_trace(point, point + probe_dir * self.max_span, first_point=False)
        if len(pts) == 0:
            return False

        dists = np.dot(pts - point, probe_dir)
        for pt_dist, cid in sorted(zip(dists, cids), key=lambda x: x[0]):
            if pt_dist <= tol:
                continue
            dot_val = float(np.dot(probe_dir, sub_normals[int(cid)]))
            if abs(dot_val) < tol:
                continue
            # If the nearest surface hit along the probe ray is an exit (dot_val > 0), the point started inside the submesh
            return dot_val > 0.0

        return False

    def _collect_hits(self, p0: np.ndarray, direction: np.ndarray, max_dist: float, tol: float) -> list[tuple[float, np.ndarray, str, bool]]:
        """Return sorted, deduplicated ray-mesh intersections along p0 + t * direction for t in [0, max_dist]."""
        p1 = p0 + direction * max_dist
        raw_hits = []

        for name, sub, sub_normals in self.submeshes:
            points, cell_ids = sub.ray_trace(p0, p1, first_point=False)
            if len(points) == 0:
                continue
            dists = np.dot(points - p0, direction)

            # Deduplicate hits on shared edges within this submesh first
            sub_hits = []
            for pt_dist, cid in sorted(zip(dists, cell_ids), key=lambda x: x[0]):
                d = float(np.clip(pt_dist, 0.0, max_dist))
                normal = sub_normals[int(cid)]
                dot_val = float(np.dot(direction, normal))
                # Ignore faces parallel to the ray
                if abs(dot_val) < tol:
                    continue
                is_entry = dot_val < 0.0
                if sub_hits and abs(d - sub_hits[-1][0]) < tol and is_entry == sub_hits[-1][3]:
                    continue
                sub_hits.append((d, normal, name, is_entry))

            # Filter out edge-skimming false hits by verifying the midpoint between consecutive hits is actually inside the submesh
            for i, (d, normal, obj_name, is_entry) in enumerate(sub_hits):
                next_d = sub_hits[i + 1][0] if i + 1 < len(sub_hits) else max_dist
                prev_d = sub_hits[i - 1][0] if i > 0 else 0.0
                if is_entry and next_d - d > tol:
                    mid_pt = p0 + direction * (0.5 * (d + next_d))
                    if not self._is_point_inside(sub, sub_normals, mid_pt, tol):
                        continue
                elif not is_entry and d - prev_d > tol:
                    mid_pt = p0 + direction * (0.5 * (prev_d + d))
                    if not self._is_point_inside(sub, sub_normals, mid_pt, tol):
                        continue
                raw_hits.append((d, normal, obj_name, is_entry))

        if not raw_hits:
            return []

        # Sort by distance first; at identical distance, process exits (False) before entries (True) for flush layers
        raw_hits.sort(key=lambda h: (round(h[0] / tol), h[3]))
        return raw_hits

    def _get_initial_stack(self, p0: np.ndarray, tol: float) -> list[tuple[str, float]]:
        """Find all objects that already enclose p0, ordered from outermost to innermost."""
        enclosing = []
        for name, sub, sub_normals in self.submeshes:
            if self._is_point_inside(sub, sub_normals, p0, tol):
                b = sub.bounds
                volume = (b[1] - b[0]) * (b[3] - b[2]) * (b[5] - b[4])
                enclosing.append((volume, name))

        # Sort from largest bounding volume (outermost) to smallest (innermost on top of stack)
        enclosing.sort(key=lambda x: x[0], reverse=True)
        return [(name, 0.0) for _, name in enclosing]

    def get_path_sections(self, start_point: list[float] | np.ndarray, end_point: list[float] | np.ndarray, free_space_name: str = "Air", tol: float = 1e-6) -> list[PathSection]:
        """Trace a straight ray from start_point to end_point and return the sections it passes through."""
        p0 = np.asarray(start_point, dtype=float)
        p1 = np.asarray(end_point, dtype=float)
        vec = p1 - p0
        total_length = float(np.linalg.norm(vec))

        if total_length < tol:
            return []

        # Unit vector pointing from start_point to end_point
        direction = vec / total_length

        # Determine if start_point is already inside one or more objects
        active_stack = self._get_initial_stack(p0, tol)

        # Collect all surface intersections between start_point and end_point
        hits = self._collect_hits(p0, direction, total_length, tol)

        sections = []
        current_d = 0.0

        # Build path sections from one intersection point to the next
        for d, normal, name, is_entry in hits:
            seg_len = d - current_d
            if seg_len > tol:
                obj_name, angle = active_stack[-1] if active_stack else (free_space_name, 0.0)
                sections.append(PathSection(seg_len, obj_name, angle))
                current_d = d

            if is_entry:
                # Ray is entering an object: push onto stack with entry angle in [0, 90] degrees
                cos_theta = min(1.0, abs(float(np.dot(direction, normal))))
                angle = float(np.degrees(np.arccos(cos_theta)))
                active_stack.append((name, angle))
            else:
                # Ray is exiting an object: remove the matching object from the active stack
                for i in range(len(active_stack) - 1, -1, -1):
                    if active_stack[i][0] == name:
                        active_stack.pop(i)
                        break

        # Add the final section from the last surface intersection to end_point
        remaining = total_length - current_d
        if remaining > tol:
            obj_name, angle = active_stack[-1] if active_stack else (free_space_name, 0.0)
            sections.append(PathSection(remaining, obj_name, angle))

        return sections

    def show(self) -> None:
        """Render the 3D model in an interactive PyVista window with distinct colors per material, a legend, and dimension axes."""
        palette = ["#89b4fa", "#f38ba8", "#a6e3a1", "#f9e2af", "#cba6f7", "#fab387", "#94e2d5", "#f5c2e7", "#74c7ec", "#b4befe"]
        plotter = pv.Plotter(window_size=[1200, 800], title=self.file_path.split(".")[0])
        plotter.set_background("#1e1e2e", top="#313244")

        # Render each material/object submesh with its own color and label for the legend
        for idx, (name, sub, _) in enumerate(self.submeshes):
            color = palette[idx % len(palette)]
            smoothed_sub = sub.compute_normals(cell_normals=False, point_normals=True, split_vertices=True, feature_angle=30.0)
            plotter.add_mesh(smoothed_sub, color=color, label=name, smooth_shading=True, show_edges=False, specular=0.25, specular_power=20, ambient=0.2)

        plotter.add_legend(bcolor=(0.12, 0.12, 0.18, 0.8), border=True, size=(0.18, 0.25))
        plotter.show_bounds(grid="back", location="outer", ticks="both", color="white", xtitle="X", ytitle="Y", ztitle="Z")
        plotter.add_axes()
        plotter.enable_anti_aliasing("msaa")
        plotter.show()

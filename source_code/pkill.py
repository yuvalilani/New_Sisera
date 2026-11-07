import numpy as np
import matplotlib.pyplot as plt


class PKillTable:
    def __init__(self, file_path: str):
        self.file_path = file_path
        # Load the CSV grid where row 0 is velocities (m/s), column 0 is masses (kg), and interior cells are probabilities
        with open(file_path, "r", encoding="utf-8") as f:
            rows = [line.strip().split(",") for line in f if line.strip()]

        self.header = rows[0][0]
        self.velocities = np.array([float(v) for v in rows[0][1:]], dtype=float)
        self.masses = np.array([float(r[0]) for r in rows[1:]], dtype=float)
        self.probabilities = np.array([[float(val) for val in r[1:]] for r in rows[1:]], dtype=float)

    def get_probability(self, mass: float | np.ndarray, velocity: float | np.ndarray) -> float | np.ndarray:
        """Compute the kill/injury probability for given shard mass (kg) and velocity (m/s) via bilinear interpolation."""
        m = np.clip(np.asarray(mass, dtype=float), self.masses[0], self.masses[-1])
        v = np.clip(np.asarray(velocity, dtype=float), self.velocities[0], self.velocities[-1])

        # Find bounding grid indices along the mass and velocity axes
        i1 = np.clip(np.searchsorted(self.masses, m, side="right"), 1, len(self.masses) - 1)
        i0 = i1 - 1
        j1 = np.clip(np.searchsorted(self.velocities, v, side="right"), 1, len(self.velocities) - 1)
        j0 = j1 - 1

        # Compute interpolation weights in [0, 1]
        dm = self.masses[i1] - self.masses[i0]
        dv = self.velocities[j1] - self.velocities[j0]
        wm = np.where(dm > 0.0, (m - self.masses[i0]) / dm, 0.0)
        wv = np.where(dv > 0.0, (v - self.velocities[j0]) / dv, 0.0)

        # Bilinear interpolation across the four surrounding grid corners
        p00 = self.probabilities[i0, j0]
        p01 = self.probabilities[i0, j1]
        p10 = self.probabilities[i1, j0]
        p11 = self.probabilities[i1, j1]

        result = (1.0 - wm) * ((1.0 - wv) * p00 + wv * p01) + wm * ((1.0 - wv) * p10 + wv * p11)
        return float(result) if np.ndim(mass) == 0 and np.ndim(velocity) == 0 else result

    def __call__(self, mass: float | np.ndarray, velocity: float | np.ndarray) -> float | np.ndarray:
        """Allow calling the table instance directly as pkill_table(mass, velocity)."""
        return self.get_probability(mass, velocity)

    def show(self) -> None:
        """Display a 2D filled contour and contour line plot of the kill probability table in steps of 0.1."""
        levels = np.linspace(0.1, 1.0, 10)
        levels_fill = np.linspace(0.0, 1.0, 51)
        fig, ax = plt.subplots(figsize=(10, 6))
        cf = ax.contourf(self.velocities, self.masses * 1e3, self.probabilities, levels=levels_fill, cmap="inferno")
        cs = ax.contour(self.velocities, self.masses * 1e3, self.probabilities, levels=levels, colors="white", linewidths=0.8)
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f")
        fig.colorbar(cf, ax=ax, ticks=levels, label="Probability")
        ax.set_xlabel("Velocity (m/s)")
        ax.set_ylabel("Mass (g)")
        ax.set_title(self.file_path)
        plt.tight_layout()
        plt.show()

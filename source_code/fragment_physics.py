from dataclasses import dataclass
import numpy as np


@dataclass
class THORParameters:
    c: float
    alpha: float
    beta: float
    gamma: float
    lam: float

    def apply_formula(self, x: float, v_in: float, m_in: float, sigma: float, d: float, angle_cosine: float) -> float:
        """Apply the empirical THOR reduction formula (all inputs in Imperial units: ft/s, grains, in^2, inches)."""
        return max(x - (10.0 ** self.c) * ((sigma * d) ** self.alpha) * (m_in ** self.beta) * (v_in ** self.lam) / (angle_cosine ** self.gamma), 0.0)


class THORMaterial:
    def __init__(self, file_path: str | None = None, velocity_params: THORParameters | None = None, mass_params: THORParameters | None = None):
        if file_path is not None:
            self.velocity_params, self.mass_params = self._load_from_csv(file_path)
        elif velocity_params is not None and mass_params is not None:
            self.velocity_params = velocity_params
            self.mass_params = mass_params
        else:
            raise ValueError("Provide either file_path or both velocity_params and mass_params.")

    @staticmethod
    def _load_from_csv(file_path: str) -> tuple[THORParameters, THORParameters]:
        """Load velocity and mass THORParameters from a CSV file in physical_data/THOR_materials/."""
        params = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.strip().split(",")]
                if not parts or parts[0] in ("", "parameter"):
                    continue
                vals = [float(v) for v in parts[1:]]
                params[parts[0].lower()] = THORParameters(*vals)
        return params["velocity"], params["mass"]

    def exit_velocity_and_mass(self, v_in: float, m_in: float, sigma: float, d: float, angle_cosine: float) -> tuple[float, float]:
        """
        Compute exit velocity (m/s) and residual mass (kg) of a fragment perforating a material layer using the THOR formula.
        :param v_in: Entry velocity in m/s
        :param m_in: Mass of fragment in kg
        :param sigma: Cross-sectional presented area of fragment in m^2
        :param d: Thickness of the material layer (or normal thickness) in m
        :param angle_cosine: Cosine of the entry obliquity angle relative to the surface normal
        :return: (v_out in m/s, m_out in kg)
        """
        if v_in <= 0.0 or m_in <= 0.0:
            return 0.0, 0.0
        if angle_cosine <= 0.0:
            return 0.0, 0.0

        # Convert SI units (m/s, kg, m^2, m) to Project THOR Imperial units (ft/s, grains, in^2, inches)
        v_imp = v_in * 3.28084
        m_imp = m_in * 15432.358
        sigma_imp = sigma * 1550.0031
        d_imp = d * 39.370079

        v_out_imp = self.velocity_params.apply_formula(v_imp, v_imp, m_imp, sigma_imp, d_imp, angle_cosine)
        # If the fragment is stopped inside the layer (v_out == 0), it does not exit
        if v_out_imp <= 0.0:
            return 0.0, 0.0

        m_out_imp = self.mass_params.apply_formula(m_imp, v_imp, m_imp, sigma_imp, d_imp, angle_cosine)

        # Convert back from Imperial units to SI units (m/s, kg)
        v_out = v_out_imp / 3.28084
        m_out = m_out_imp / 15432.358

        return v_out, m_out


def air_drag(v_in: float, c_d: float, rho: float, rho_f: float, k_s: float, m: float, x: float) -> float:
    """
    Compute fragment velocity after traveling distance x in air, where presented area A = k_s * (m / rho_f)^(2/3).
    :param v_in: Initial velocity in m/s
    :param c_d: Drag coefficient (unitless)
    :param rho: Air density in kg/m^3
    :param rho_f: Fragment material density in kg/m^3
    :param k_s: Fragment shape factor (unitless, where A = k_s * V_frag^(2/3) = k_s * (m / rho_f)^(2/3))
    :param m: Fragment mass in kg
    :param x: Distance traveled in air in m
    :return: Final velocity in m/s
    """
    if m <= 0.0 or v_in <= 0.0:
        return 0.0
    # From m * dv/dx = -0.5 * rho * c_d * A * v, with A = k_s * (m / rho_f)^(2/3):
    # v(x) = v_in * exp(-0.5 * c_d * rho * k_s / (rho_f^(2/3) * m^(1/3)) * x)
    v_out = v_in * float(np.exp(-0.5 * c_d * rho * k_s / ((rho_f ** 2 * m) ** (1.0 / 3.0)) * x))
    return max(v_out, 0.0)

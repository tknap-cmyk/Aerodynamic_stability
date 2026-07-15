import numpy as np
from abc import ABC, abstractmethod

class BaseAeroEngine(ABC):
    def __init__(self, geometry):
        self.geom = geometry

    @abstractmethod
    def get_instant_state(self, alpha_rad, Q_dyn, Cd):
        pass


class BarrowmanSolver:
    def __init__(self, geometry):
        self.geom = geometry

        self.CN_low = 0.0
        self.CP_low = 0.0
        self.A_total = 0.0
        self.CP_high = 0.0

        self._calculate_barrowman()
        self._calculate_crossflow()

    def _calculate_barrowman(self):
        total_CN = 0.0
        weighted_CP = 0.0

        max_r_idx = np.argmax(self.geom.r)

        for i in range(len(self.geom.x) - 1):
            x1, r1 = self.geom.x[i], self.geom.r[i]
            x2, r2 = self.geom.x[i + 1], self.geom.r[i + 1]
            L = x2 - x1
            if L <= 0:
                continue

            # Standard Barrowman Normal Force for interval
            CN_i = 2 * ((r2 / self.geom.R_ref) ** 2 - (r1 / self.geom.R_ref) ** 2)

            if i >= max_r_idx and r2 < r1:
                CN_i = 0.0

            # Local CP for the interval
            if r1 + r2 == 0:
                X_cp_i = x1 + (L / 2)
            else:
                X_cp_i = x1 + (L / 3) * (r1 + 2 * r2) / (r1 + r2)

            total_CN += CN_i
            weighted_CP += CN_i * X_cp_i

        self.CN_low = total_CN
        self.CP_low = weighted_CP / total_CN if total_CN != 0 else self.geom.x[-1] / 2

    def _calculate_crossflow(self):
        total_Area = 0.0
        weighted_Area_X = 0.0

        for i in range(len(self.geom.x) - 1):
            x1, r1 = self.geom.x[i], self.geom.r[i]
            x2, r2 = self.geom.x[i + 1], self.geom.r[i + 1]
            L = x2 - x1

            A_i = 2 * ((r1 + r2) / 2) * L

            if r1 + r2 == 0:
                X_area_i = x1 + (L / 2)
            else:
                X_area_i = x1 + (L / 3) * (r1 + 2 * r2) / (r1 + r2)

            total_Area += A_i
            weighted_Area_X += A_i * X_area_i

        self.A_total = total_Area
        self.CP_high = weighted_Area_X / total_Area if total_Area != 0 else self.geom.x[-1] / 2

    def get_instant_state(self, alpha_rad, Q_dyn, Cd): #chat com
        # 1. Barrowman Aerodynamic Lift (low angles of attack)
        F_lift = Q_dyn * self.geom.A_ref * self.CN_low * np.sin(alpha_rad)

        # 2. Newtonian Crossflow Drag (high angles of attack)
        F_cross = Q_dyn * self.A_total * Cd * (np.sin(alpha_rad) * np.abs(np.sin(alpha_rad)))

        F_normal = F_lift + F_cross

        # 3. Calculate Restoring Torque
        Torque_lift = -F_lift * (self.CP_low - self.geom.x_com)
        Torque_cross = -F_cross * (self.CP_high - self.geom.x_com)

        Torque = Torque_lift + Torque_cross

        # 4. Calculate Effective Center of Pressure (For Visualization)
        if F_normal != 0:
            CP_actual = self.geom.x_com - (Torque / F_normal)
        else:
            CP_actual = self.CP_low

        return CP_actual, F_normal, Torque
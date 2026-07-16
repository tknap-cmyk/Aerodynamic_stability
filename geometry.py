import numpy as np
import csv
from abc import ABC, abstractmethod

class BaseGeometryParser(ABC):
    @abstractmethod
    def parse(self, filepath):
        pass

class CSVParser(BaseGeometryParser):
    def parse(self, filepath):
        x_raw, r_raw = [], []
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    try:
                        x_raw.append(float(row[0]))
                        r_raw.append(float(row[1]))
                    except ValueError:
                        continue

        x_raw = np.array(x_raw)
        r_raw = np.array(r_raw)

        # Airfoil Format Detection
        if x_raw[0] > x_raw[len(x_raw) // 2]:
            nose_idx = np.argmin(x_raw)

            # Extract just the top surface (from nose to tail)
            x_top = x_raw[:nose_idx + 1][::-1]  # Reverse for 0 -> Max
            r_top = r_raw[:nose_idx + 1][::-1]

            x_points = x_top
            r_points = r_top
        else:
            x_points = x_raw
            r_points = r_raw

        x_points = x_points - x_points[0]
        r_points = np.abs(r_points)

        return x_points, r_points

#test dodawania nastepnych formatow
class TXTParser(BaseGeometryParser):
    def parse(self, filepath):
        x_points, r_points = [], []

        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 2:
                    x_points.append(float(parts[0]))
                    r_points.append(float(parts[1]))

        return np.array(x_points), np.array(r_points)

class GeometryHandler:
    def __init__(self, x_points, r_points, x_com="auto"):
        self.x = np.array(x_points, dtype=float)
        self.r = np.array(r_points, dtype=float)

        if x_com == "auto" or x_com is None:
            self.x_com = self._calculate_volumetric_com()
        else:
            self.x_com = float(x_com)

        self.R_ref = np.max(self.r)
        self.A_ref = np.pi * (self.R_ref ** 2)

        # 2D shape array for Matplotlib
        self._build_2d_shape()

    @classmethod
    def from_file(cls, filepath, parser: BaseGeometryParser, x_com="auto"):
        x_points, r_points = parser.parse(filepath)
        return cls(x_points, r_points, x_com=x_com)

    def _calculate_volumetric_com(self):
        """COM assuming uniform density (Volumetric Centroid)."""
        total_volume = 0.0
        weighted_x = 0.0

        for i in range(len(self.x) - 1):
            x1, r1 = self.x[i], self.r[i]
            x2, r2 = self.x[i + 1], self.r[i + 1]
            h = x2 - x1
            if h <= 0:
                continue

            v_i = (np.pi * h / 3.0) * (r1 ** 2 + r1 * r2 + r2 ** 2)

            if r1 == 0 and r2 == 0:
                local_centroid = h / 2.0
            else:
                local_centroid = (h / 4.0) * (r1 ** 2 + 2 * r1 * r2 + 3 * r2 ** 2) / (r1 ** 2 + r1 * r2 + r2 ** 2)

            x_com_i = x1 + local_centroid

            total_volume += v_i
            weighted_x += v_i * x_com_i

            #zero div logic
        return weighted_x / total_volume if total_volume > 0 else self.x[-1] / 2.0

    def _build_2d_shape(self):
        """Creates the coordinates needed to draw the polygon in Matplotlib."""
        right_x = self.r
        right_y = self.x - self.x_com

        left_x = -self.r[::-1]
        left_y = self.x[::-1] - self.x_com

        self.shape_x = np.concatenate((right_x, left_x))
        self.shape_y = np.concatenate((right_y, left_y))
import math
import numpy as np
from .data import XYWH
from .utils import norm_phase_to_half_circle, sub_phase_to_half_circle

class PhaseCalculator:
    """
    Handles phase calculation related operations.

    Attributes:
        antpos (dict[int, tuple[float, float, float]]): Antenna positions
        p2m (float): Pixel to meter conversion factor
    """

    def __init__(self, antpos: dict[int, tuple[float, float, float]], p2m: float):
        self.antpos = antpos
        self.p2m = p2m

    def calculate_predicted_phase(
        self, pos1: XYWH, pos2: XYWH, antid: int, freq: float
    ) -> float:
        """Calculate predicted phase difference"""
        x1, y1 = pos1.x + pos1.w / 2, pos1.y + pos1.h / 2
        x2, y2 = pos2.x + pos2.w / 2, pos2.y + pos2.h / 2

        dis1 = (
            math.sqrt(
                (x1 - self.antpos[antid][0]) ** 2
                + (y1 - self.antpos[antid][1]) ** 2
                + self.antpos[antid][2] ** 2
            )
            * self.p2m
        )
        dis2 = (
            math.sqrt(
                (x2 - self.antpos[antid][0]) ** 2
                + (y2 - self.antpos[antid][1]) ** 2
                + self.antpos[antid][2] ** 2
            )
            * self.p2m
        )

        return 4 * np.pi * (dis2 - dis1) / (3e8 / freq)

    def calculate_actual_phase(self, phase1: float, phase2: float) -> float:
        """Calculate actual phase difference"""
        norm_phase1 = norm_phase_to_half_circle(phase1)
        norm_phase2 = norm_phase_to_half_circle(phase2)
        return sub_phase_to_half_circle(norm_phase1, norm_phase2)
import numpy as np
from typing import TypedDict
from configs import simulation as SimulationConfig

M = SimulationConfig.MACH
N = SimulationConfig.AGENTS

_rng = np.random.default_rng()


class SimulationParams(TypedDict):
    positions: list[list[float]]
    headings: list[float]
    velocity_mins: list[float]
    velocity_maxs: list[float]
    azimuth_rate_mins: list[float]
    azimuth_rate_maxs: list[float]
    attack_angle_mins: list[float]
    attack_angle_maxs: list[float]
    thrust_ratio: list[float]
    attack_angle_ratio: list[float]
    roll_angle_ratio: list[float]


BASE: SimulationParams = {
    "positions": _rng.uniform([1000, 1000, 4000], [9000, 9000, 8000], size=(N, 3)).tolist(),
    "headings": _rng.uniform(0, 2 * np.pi, size=N).tolist(),
    "velocity_mins": [0.1 * M] * N,
    "velocity_maxs": [0.3 * M] * N,
    "azimuth_rate_mins": [-1.3] * N,
    "azimuth_rate_maxs": [1.3] * N,
    "attack_angle_mins": [0.09] * N,
    "attack_angle_maxs": [0.52] * N,
    "thrust_ratio": [10.0] * N,
    "attack_angle_ratio": [1.5] * N,
    "roll_angle_ratio": [1.0] * N,
}

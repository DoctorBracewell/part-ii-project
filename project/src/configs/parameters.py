from typing import TypedDict
from configs import simulation as SimulationConfig

M = SimulationConfig.MACH


class SimulationParams(TypedDict):
    positions: list[list[float]]
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
    "positions": [[5000, 4000, 6500], [5000, 6000, 6500]],
    "velocity_mins": [0.1 * M] * 2,
    "velocity_maxs": [0.3 * M] * 2,
    "azimuth_rate_mins": [-1.3] * 2,
    "azimuth_rate_maxs": [1.3] * 2,
    "attack_angle_mins": [0.09] * 2,
    "attack_angle_maxs": [0.52] * 2,
    "thrust_ratio": [10.0, 10.0],
    "attack_angle_ratio": [1.5] * 2,
    "roll_angle_ratio": [1.0] * 2,
}

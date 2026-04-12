from __future__ import annotations

import numpy as np
from typing import TypedDict, NotRequired
from configs import simulation as SimulationConfig

M = SimulationConfig.MACH


class ScenarioParams(TypedDict):
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


class Scenario(TypedDict):
    id: int
    name: str
    description: str
    N: int
    params: ScenarioParams
    initial: NotRequired[dict[str, list[float]]]


# Each scenario is passed directly to Simulation(**params).
# `initial` overrides Simulation attributes after construction (e.g. azimuth_angles).

SCENARIOS: list[Scenario] = [
    {
        "id": 1,
        "name": "symmetric_head_on",
        "description": "Two equal-capability agents approaching each other head-on",
        "N": 2,
        "params": {
            "positions": [[5000.0, 3000.0, 6500.0], [5000.0, 7000.0, 6500.0]],
            "velocity_mins": [0.1 * M] * 2,
            "velocity_maxs": [0.3 * M] * 2,
            "azimuth_rate_mins": [-1.3] * 2,
            "azimuth_rate_maxs": [1.3] * 2,
            "attack_angle_mins": [0.09] * 2,
            "attack_angle_maxs": [0.52] * 2,
            "thrust_ratio": [10.0] * 2,
            "attack_angle_ratio": [1.5] * 2,
            "roll_angle_ratio": [1.0] * 2,
        },
        "initial": {"azimuth_angles": [np.pi / 2, -np.pi / 2]},
    },
    {
        "id": 2,
        "name": "tail_chase",
        "description": "Pursuer directly behind evader, both heading in the same direction",
        "N": 2,
        "params": {
            "positions": [[5000.0, 4000.0, 6500.0], [5000.0, 6000.0, 6500.0]],
            "velocity_mins": [0.1 * M] * 2,
            "velocity_maxs": [0.35 * M, 0.3 * M],
            "azimuth_rate_mins": [-1.3] * 2,
            "azimuth_rate_maxs": [1.3] * 2,
            "attack_angle_mins": [0.09] * 2,
            "attack_angle_maxs": [0.52] * 2,
            "thrust_ratio": [12.0, 10.0],
            "attack_angle_ratio": [1.5] * 2,
            "roll_angle_ratio": [1.0] * 2,
        },
        "initial": {"azimuth_angles": [np.pi / 2, np.pi / 2]},
    },
    {
        "id": 3,
        "name": "lateral_intercept",
        "description": "Agents start with perpendicular headings on a crossing course",
        "N": 2,
        "params": {
            "positions": [[3000.0, 5000.0, 6500.0], [7000.0, 5000.0, 6500.0]],
            "velocity_mins": [0.1 * M] * 2,
            "velocity_maxs": [0.3 * M] * 2,
            "azimuth_rate_mins": [-1.3] * 2,
            "azimuth_rate_maxs": [1.3] * 2,
            "attack_angle_mins": [0.09] * 2,
            "attack_angle_maxs": [0.52] * 2,
            "thrust_ratio": [10.0] * 2,
            "attack_angle_ratio": [1.5] * 2,
            "roll_angle_ratio": [1.0] * 2,
        },
        "initial": {"azimuth_angles": [0.0, np.pi]},
    },
    {
        "id": 4,
        "name": "facing_away",
        "description": "Agents face directly away from each other, forcing maximum turn rates",
        "N": 2,
        "params": {
            "positions": [[5000.0, 4000.0, 6500.0], [5000.0, 6000.0, 6500.0]],
            "velocity_mins": [0.1 * M] * 2,
            "velocity_maxs": [0.3 * M] * 2,
            "azimuth_rate_mins": [-1.5] * 2,
            "azimuth_rate_maxs": [1.5] * 2,
            "attack_angle_mins": [0.09] * 2,
            "attack_angle_maxs": [0.52] * 2,
            "thrust_ratio": [10.0] * 2,
            "attack_angle_ratio": [1.5] * 2,
            "roll_angle_ratio": [1.5] * 2,
        },
        "initial": {"azimuth_angles": [-np.pi / 2, np.pi / 2]},
    },
    {
        "id": 5,
        "name": "capability_asymmetry",
        "description": "Pursuer has significantly higher speed envelope than evader",
        "N": 2,
        "params": {
            "positions": [[5000.0, 3000.0, 6500.0], [5000.0, 7000.0, 6500.0]],
            "velocity_mins": [0.1 * M] * 2,
            "velocity_maxs": [0.5 * M, 0.2 * M],
            "azimuth_rate_mins": [-1.3] * 2,
            "azimuth_rate_maxs": [1.3] * 2,
            "attack_angle_mins": [0.09] * 2,
            "attack_angle_maxs": [0.52] * 2,
            "thrust_ratio": [15.0, 10.0],
            "attack_angle_ratio": [1.5] * 2,
            "roll_angle_ratio": [1.0] * 2,
        },
        "initial": {"azimuth_angles": [np.pi / 2, -np.pi / 2]},
    },
    {
        "id": 6,
        "name": "three_way",
        "description": "Three-agent engagement matching the main simulation configuration",
        "N": 3,
        "params": {
            "positions": [[5000.0, 4000.0, 6500.0], [5000.0, 6000.0, 6500.0], [5000.0, 5000.0, 6500.0]],
            "velocity_mins": [0.1 * M] * 3,
            "velocity_maxs": [0.3 * M] * 3,
            "azimuth_rate_mins": [-1.3] * 3,
            "azimuth_rate_maxs": [1.3] * 3,
            "attack_angle_mins": [0.09] * 3,
            "attack_angle_maxs": [0.52] * 3,
            "thrust_ratio": [10.0] * 3,
            "attack_angle_ratio": [1.5] * 3,
            "roll_angle_ratio": [1.0] * 3,
        },
    },
]

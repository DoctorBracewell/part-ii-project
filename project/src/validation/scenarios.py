from __future__ import annotations

import numpy as np
from typing import TypedDict

from configs.parameters import M


class Scenario(TypedDict):
    id: int
    name: str
    positions: list[list[float]]
    velocities: list[float]
    azimuth_angles: list[float]


def _pos(sep: float, alt: float) -> list[list[float]]:
    return [[5000.0, 5000.0 - sep / 2, alt], [5000.0, 5000.0 + sep / 2, alt]]


HEAD_ON = [np.pi / 2, -np.pi / 2]
FACING_AWAY = [-np.pi / 2, np.pi / 2]
PERPENDICULAR = [np.pi / 2, 0.0]


SCENARIOS: list[Scenario] = [
    {
        "id": 1,
        "name": "head_on_close",
        "positions": _pos(500, 5000.0),
        "velocities": [0.2 * M] * 2,
        "azimuth_angles": HEAD_ON,
    },
    {
        "id": 2,
        "name": "head_on_distant",
        "positions": _pos(20000, 5000),
        "velocities": [0.2 * M] * 2,
        "azimuth_angles": HEAD_ON,
    },
    {
        "id": 3,
        "name": "facing_away",
        "positions": _pos(2000, 5000.0),
        "velocities": [0.2 * M] * 2,
        "azimuth_angles": FACING_AWAY,
    },
    {
        "id": 4,
        "name": "perpendicular",
        "positions": _pos(2000, 5000.0),
        "velocities": [0.2 * M] * 2,
        "azimuth_angles": PERPENDICULAR,
    },
    {
        "id": 5,
        "name": "pursuer_faster",
        "positions": _pos(2000, 5000.0),
        "velocities": [0.3 * M, 0.15 * M],
        "azimuth_angles": HEAD_ON,
    },
    {
        "id": 6,
        "name": "pursuer_slower",
        "positions": _pos(2000, 5000.0),
        "velocities": [0.15 * M, 0.3 * M],
        "azimuth_angles": HEAD_ON,
    },
    {
        "id": 7,
        "name": "low_altitude",
        "positions": _pos(2000, 15.0),
        "velocities": [0.2 * M] * 2,
        "azimuth_angles": HEAD_ON,
    },
]

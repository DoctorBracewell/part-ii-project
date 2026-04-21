from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Callable, TypedDict
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from configs import simulation as SimulationConfig
from configs.parameters import SimulationParams, BASE as BaseParams
from validation.scenarios import Scenario

if TYPE_CHECKING:
    from simulation.simulation import Simulation

TOLERANCE = 1e-6


class RecordData(TypedDict):
    initial_positions: npt.NDArray[np.float64]
    positions: list[npt.NDArray[np.float64]]
    velocities: list[npt.NDArray[np.float64]]
    azimuth_angles: list[npt.NDArray[np.float64]]
    active: list[npt.NDArray[np.bool_]]


class KinematicResult(TypedDict):
    passed: bool
    speed_violations: list[tuple[int, int]]
    altitude_violations: list[tuple[int, int]]
    turn_rate_violations: list[tuple[int, int]]


class BehaviouralResult(TypedDict):
    passed: bool
    pair: tuple[int, int]
    d: npt.NDArray[np.float64]
    d0: float
    d_min: float
    monotonically_increasing: bool


def make_recorder(
    N: int, initial_positions: npt.NDArray[np.float64]
) -> tuple[Callable[[Simulation], None], RecordData]:
    """
    Returns a (callback, data) pair.
    Call callback(sim) after each step. data accumulates per-timestep state.
    initial_positions is recorded separately so d_0 reflects the true starting separation.
    """
    data: RecordData = {
        "initial_positions": initial_positions.copy(),
        "positions": [],
        "velocities": [],
        "azimuth_angles": [],
        "active": [],
    }

    def record(sim: Simulation) -> None:
        data["positions"].append(sim.positions.copy())
        data["velocities"].append(sim.velocities.copy())
        data["azimuth_angles"].append(sim.azimuth_angles.copy())
        data["active"].append(sim.active.copy())

    return record, data


def kinematic_valid(data: RecordData, params: SimulationParams) -> KinematicResult:
    """Check speed, altitude, and turn rate remain within bounds for all active agents."""
    positions: npt.NDArray[np.float64] = np.array(data["positions"])  # (T, N, 3)
    velocities: npt.NDArray[np.float64] = np.array(data["velocities"])  # (T, N, 3)
    speeds: npt.NDArray[np.float64] = np.linalg.norm(velocities, axis=2)  # (T, N)
    azimuths: npt.NDArray[np.float64] = np.array(data["azimuth_angles"])  # (T, N)
    active: npt.NDArray[np.bool_] = np.array(data["active"])  # (T, N)
    T, N = speeds.shape
    dt: float = 1.0 / SimulationConfig.STEPS_PER_SECOND

    v_mins: npt.NDArray[np.float64] = np.array(params["velocity_mins"])
    v_maxs: npt.NDArray[np.float64] = np.array(params["velocity_maxs"])
    az_mins: npt.NDArray[np.float64] = np.array(params["azimuth_rate_mins"])
    az_maxs: npt.NDArray[np.float64] = np.array(params["azimuth_rate_maxs"])

    speed_violations: list[tuple[int, int]] = []
    altitude_violations: list[tuple[int, int]] = []
    turn_rate_violations: list[tuple[int, int]] = []

    for t in range(T):
        for a in range(N):
            if not active[t, a]:
                continue

            if speeds[t, a] < v_mins[a] - TOLERANCE or speeds[t, a] > v_maxs[a] + TOLERANCE:
                speed_violations.append((t, a))

            z: float = float(positions[t, a, 2])
            if z < SimulationConfig.HARD_DECK - TOLERANCE or z > SimulationConfig.HEIGHT + TOLERANCE:
                altitude_violations.append((t, a))

            if t > 0 and active[t - 1, a]:
                turn_rate: float = float(azimuths[t, a] - azimuths[t - 1, a]) / dt
                if turn_rate < az_mins[a] - TOLERANCE or turn_rate > az_maxs[a] + TOLERANCE:
                    turn_rate_violations.append((t, a))

    return KinematicResult(
        passed=not (speed_violations or altitude_violations or turn_rate_violations),
        speed_violations=speed_violations,
        altitude_violations=altitude_violations,
        turn_rate_violations=turn_rate_violations,
    )


def behavioural_valid(data: RecordData, pair: tuple[int, int]) -> BehaviouralResult:
    """
    For the given agent pair, check:
      - d(t) is not monotonically increasing throughout the run
      - d_min < d_0 (agents got closer than their initial separation at some point)
    """
    positions: npt.NDArray[np.float64] = np.array(data["positions"])  # (T, N, 3)
    a, b = pair

    d0: float = float(
        np.linalg.norm(data["initial_positions"][a] - data["initial_positions"][b])
    )
    d: npt.NDArray[np.float64] = np.linalg.norm(
        positions[:, a] - positions[:, b], axis=1
    )
    d_min: float = float(np.min(d))
    monotonic: bool = bool(np.all(np.diff(d) >= 0))

    return BehaviouralResult(
        passed=(not monotonic) and (d_min < d0),
        pair=pair,
        d=d,
        d0=d0,
        d_min=d_min,
        monotonically_increasing=monotonic,
    )


def plot_kinematic(scenario: Scenario, data: RecordData, output_dir: str) -> KinematicResult:
    result = kinematic_valid(data, BaseParams)

    positions: npt.NDArray[np.float64] = np.array(data["positions"])   # (T, N, 3)
    velocities: npt.NDArray[np.float64] = np.array(data["velocities"]) # (T, N, 3)
    speeds: npt.NDArray[np.float64] = np.linalg.norm(velocities, axis=2)
    azimuths: npt.NDArray[np.float64] = np.array(data["azimuth_angles"])
    dt = 1.0 / SimulationConfig.STEPS_PER_SECOND
    turn_rates = np.diff(azimuths, axis=0) / dt  # (T-1, N)
    N = speeds.shape[1]

    linestyles = ["-", "--", ":"]
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    for a in range(N):
        axes[0].plot(speeds[:, a], linestyle=linestyles[a % len(linestyles)], label=f"agent {a}")
    axes[0].axhline(BaseParams["velocity_mins"][0], color="r", linestyle="--", linewidth=0.8)
    axes[0].axhline(BaseParams["velocity_maxs"][0], color="r", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].legend(fontsize=8)

    for a in range(N):
        axes[1].plot(positions[:, a, 2], linestyle=linestyles[a % len(linestyles)], label=f"agent {a}")
    axes[1].axhline(SimulationConfig.HARD_DECK, color="r", linestyle="--", linewidth=0.8)
    axes[1].axhline(SimulationConfig.HEIGHT, color="r", linestyle="--", linewidth=0.8)
    axes[1].set_ylabel("Altitude (m)")
    axes[1].legend(fontsize=8)

    for a in range(N):
        axes[2].plot(turn_rates[:, a], linestyle=linestyles[a % len(linestyles)], label=f"agent {a}")
    axes[2].axhline(BaseParams["azimuth_rate_mins"][0], color="r", linestyle="--", linewidth=0.8)
    axes[2].axhline(BaseParams["azimuth_rate_maxs"][0], color="r", linestyle="--", linewidth=0.8)
    axes[2].set_ylabel("Turn rate (rad/s)")
    axes[2].set_xlabel("Timestep")
    axes[2].legend(fontsize=8)

    axes[0].set_title(f"{scenario['id']:02d}_{scenario['name']}  kinematic  {'PASS' if result['passed'] else 'FAIL'}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{scenario['id']:02d}_{scenario['name']}_kinematic.png"), dpi=150)
    plt.close(fig)

    return result


def plot_behavioural(scenario: Scenario, data: RecordData, output_dir: str) -> list[BehaviouralResult]:
    N = len(scenario["positions"])
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    results = [behavioural_valid(data, pair) for pair in pairs]

    overall = "PASS" if all(b["passed"] for b in results) else "FAIL"
    fig, ax = plt.subplots(figsize=(8, 4))
    for beh in results:
        ax.plot(beh["d"], label=f"agents {beh['pair'][0]}-{beh['pair'][1]}")
        ax.axhline(beh["d0"], color="steelblue", linestyle="--", linewidth=0.8, alpha=0.7, label=f"d0 = {beh['d0']:.0f} m")
        ax.axhline(beh["d_min"], color="steelblue", linestyle=":", linewidth=0.8, alpha=0.7, label=f"d_min = {beh['d_min']:.0f} m")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Separation (m)")
    ax.set_title(f"{scenario['id']:02d}_{scenario['name']}  behavioural  {overall}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"{scenario['id']:02d}_{scenario['name']}_behavioural.png"), dpi=150)
    plt.close(fig)

    return results


def load_data(path: str) -> RecordData:
    with open(path) as f:
        raw = json.load(f)
    return RecordData(
        initial_positions=np.array(raw["initial_positions"]),
        positions=[np.array(p) for p in raw["positions"]],
        velocities=[np.array(v) for v in raw["velocities"]],
        azimuth_angles=[np.array(a) for a in raw["azimuth_angles"]],
        active=[np.array(a) for a in raw["active"]],
    )

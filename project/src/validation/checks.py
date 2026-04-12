from __future__ import annotations

import os
from typing import TYPE_CHECKING, Callable, TypedDict
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from configs import simulation as SimulationConfig
from validation.scenarios import Scenario, ScenarioParams

if TYPE_CHECKING:
    from simulation.simulation import Simulation


class RecordData(TypedDict):
    initial_positions: npt.NDArray[np.float64]
    positions: list[npt.NDArray[np.float64]]
    speeds: list[npt.NDArray[np.float64]]
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
        "speeds": [],
        "azimuth_angles": [],
        "active": [],
    }

    def record(sim: Simulation) -> None:
        data["positions"].append(sim.positions.copy())
        data["speeds"].append(sim.speeds.copy())
        data["azimuth_angles"].append(sim.azimuth_angles.copy())
        data["active"].append(sim.active.copy())

    return record, data


def kinematic_valid(data: RecordData, params: ScenarioParams) -> KinematicResult:
    """Check speed, altitude, and turn rate remain within bounds for all active agents."""
    positions: npt.NDArray[np.float64] = np.array(data["positions"])    # (T, N, 3)
    speeds: npt.NDArray[np.float64] = np.array(data["speeds"])          # (T, N)
    azimuths: npt.NDArray[np.float64] = np.array(data["azimuth_angles"])  # (T, N)
    active: npt.NDArray[np.bool_] = np.array(data["active"])            # (T, N)
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

            if speeds[t, a] < v_mins[a] or speeds[t, a] > v_maxs[a]:
                speed_violations.append((t, a))

            z: float = float(positions[t, a, 2])
            if z < SimulationConfig.HARD_DECK or z > SimulationConfig.HEIGHT:
                altitude_violations.append((t, a))

            if t > 0 and active[t - 1, a]:
                turn_rate: float = float(azimuths[t, a] - azimuths[t - 1, a]) / dt
                if turn_rate < az_mins[a] - 1e-9 or turn_rate > az_maxs[a] + 1e-9:
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

    d0: float = float(np.linalg.norm(
        data["initial_positions"][a] - data["initial_positions"][b]
    ))
    d: npt.NDArray[np.float64] = np.linalg.norm(positions[:, a] - positions[:, b], axis=1)
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


def summarise_and_plot(
    scenario: Scenario, data: RecordData, output_dir: str
) -> tuple[KinematicResult, list[BehaviouralResult]]:
    """Run all checks, print a summary, and save a d(t) plot."""
    N: int = scenario["N"]
    pairs: list[tuple[int, int]] = [(i, j) for i in range(N) for j in range(i + 1, N)]

    kin: KinematicResult = kinematic_valid(data, scenario["params"])
    beh_results: list[BehaviouralResult] = [behavioural_valid(data, pair) for pair in pairs]

    # --- summary ---
    k_label: str = "PASS" if kin["passed"] else (
        f"FAIL  speed={len(kin['speed_violations'])}  "
        f"alt={len(kin['altitude_violations'])}  "
        f"turn={len(kin['turn_rate_violations'])}"
    )
    print(f"    kinematic  : {k_label}")
    for beh in beh_results:
        b_label: str = "PASS" if beh["passed"] else (
            f"FAIL  d_min={beh['d_min']:.0f} m  d_0={beh['d0']:.0f} m  "
            + ("monotonic" if beh["monotonically_increasing"] else "")
            + ("" if beh["d_min"] < beh["d0"] else " d_min>=d_0")
        )
        print(f"    behavioural {beh['pair']}: {b_label}")

    # --- plot ---
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))

    colors: list[str] = ["#6366f1", "#22c55e", "#f97316"]
    for idx, beh in enumerate(beh_results):
        c: str = colors[idx % len(colors)]
        ax.plot(beh["d"], color=c, linewidth=1.5,
                label=f"agents {beh['pair'][0]}–{beh['pair'][1]}")
        ax.axhline(beh["d0"], color=c, linestyle="--", linewidth=0.8, alpha=0.6,
                   label=f"d₀ = {beh['d0']:.0f} m")
        ax.axhline(beh["d_min"], color=c, linestyle=":", linewidth=0.8, alpha=0.6,
                   label=f"d_min = {beh['d_min']:.0f} m")

    b_pass: bool = all(b["passed"] for b in beh_results)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Separation (m)")
    ax.set_title(
        f"Scenario {scenario['id']}: {scenario['name']}\n"
        f"Kinematic: {'PASS' if kin['passed'] else 'FAIL'}  |  "
        f"Behavioural: {'PASS' if b_pass else 'FAIL'}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()

    path: str = os.path.join(output_dir, f"{scenario['id']:02d}_{scenario['name']}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    plot saved : {path}")

    return kin, beh_results

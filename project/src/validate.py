import json
import os
from argparse import ArgumentParser
from threading import Thread
from typing import Any, Callable

import numpy as np
from rich.progress import Progress, TextColumn, BarColumn

from configs import output as OutputConfig
from configs import simulation as SimulationConfig
from configs.parameters import BASE as BaseParams
from simulation.simulation import Simulation
from validation.checks import RecordData, make_recorder
from validation.scenarios import SCENARIOS, Scenario


def run_scenario(
    scenario: Scenario,
    progress: Progress,
    callbacks: list[Callable[[Simulation], None]],
) -> RecordData:
    N = len(scenario["positions"])
    sim = Simulation(
        N,
        positions=scenario["positions"],
        velocity_mins=BaseParams["velocity_mins"],
        velocity_maxs=BaseParams["velocity_maxs"],
        azimuth_rate_mins=BaseParams["azimuth_rate_mins"],
        azimuth_rate_maxs=BaseParams["azimuth_rate_maxs"],
        attack_angle_mins=BaseParams["attack_angle_mins"],
        attack_angle_maxs=BaseParams["attack_angle_maxs"],
        thrust_ratio=BaseParams["thrust_ratio"],
        attack_angle_ratio=BaseParams["attack_angle_ratio"],
        roll_angle_ratio=BaseParams["roll_angle_ratio"],
    )
    sim.speeds = np.array(scenario["velocities"], dtype=np.float64)
    sim.azimuth_angles = np.array(scenario["azimuth_angles"], dtype=np.float64)

    recorder, data = make_recorder(N, sim.positions)
    task = progress.add_task(scenario["name"], total=SimulationConfig.MAX_TIMESTEPS)

    while sim.timestep < SimulationConfig.MAX_TIMESTEPS:
        sim.step()
        recorder(sim)
        for cb in callbacks:
            cb(sim)
        progress.advance(task)
        if np.sum(sim.active) <= 1:
            break

    progress.update(task, completed=SimulationConfig.MAX_TIMESTEPS)
    return data


def save_data(scenario: Scenario, data: RecordData) -> None:
    os.makedirs(OutputConfig.VALIDATION_DIRECTORY, exist_ok=True)
    serialisable: dict[str, Any] = {
        "initial_positions": data["initial_positions"].tolist(),
        "positions": [p.tolist() for p in data["positions"]],
        "velocities": [v.tolist() for v in data["velocities"]],
        "azimuth_angles": [a.tolist() for a in data["azimuth_angles"]],
        "active": [a.tolist() for a in data["active"]],
    }
    path = os.path.join(
        OutputConfig.VALIDATION_DIRECTORY,
        f"{scenario['id']:02d}_{scenario['name']}.json",
    )
    with open(path, "w") as f:
        json.dump(serialisable, f)


def run_all(vis_update: Callable[[Simulation], None] | None = None) -> None:
    callbacks: list[Callable[[Simulation], None]] = [vis_update] if vis_update else []
    with Progress(
        TextColumn("{task.description:<30}"),
        BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
    ) as progress:
        for scenario in SCENARIOS:
            data = run_scenario(scenario, progress, callbacks)
            save_data(scenario, data)


def plot_all() -> None:
    from validation.checks import load_data, plot_kinematic, plot_behavioural

    output_dir = OutputConfig.VALIDATION_DIRECTORY
    os.makedirs(output_dir, exist_ok=True)
    for scenario in SCENARIOS:
        path = os.path.join(output_dir, f"{scenario['id']:02d}_{scenario['name']}.json")
        if not os.path.exists(path):
            print(f"[skip] {scenario['name']} no data found")
            continue
        data = load_data(path)
        plot_kinematic(scenario, data, output_dir)
        plot_behavioural(scenario, data, output_dir)


if __name__ == "__main__":
    parser = ArgumentParser(prog="validate")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_parser = sub.add_parser("run", help="Simulate all scenarios and save data.")
    run_parser.add_argument(
        "-v", "--visualise", action="store_true", help="Live 3D visualisation."
    )
    run_parser.add_argument(
        "-s", "--scenario", type=int, default=None, help="Run a single scenario by ID."
    )

    sub.add_parser("plot", help="Load saved data and generate plots.")

    args = parser.parse_args()

    if args.cmd == "run":
        if args.scenario is not None:
            matching = [s for s in SCENARIOS if s["id"] == args.scenario]
            if not matching:
                print(f"No scenario with id {args.scenario}")
                raise SystemExit(1)
            SCENARIOS[:] = matching
        if args.visualise:
            from visualisation import VisualisationManager
            from PyQt5.QtWidgets import QApplication

            vis = VisualisationManager(None, 2)  # type: ignore
            Thread(target=run_all, args=(vis.update,), daemon=True).start()
            QApplication([]).exec_()
        else:
            run_all()
    elif args.cmd == "plot":
        plot_all()

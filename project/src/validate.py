import json
import os
from argparse import ArgumentParser
from threading import Thread
from typing import Any, Callable

import numpy as np
from rich.progress import Progress, TextColumn, BarColumn

from configs import output as OutputConfig
from configs import simulation as SimulationConfig
from simulation.simulation import Simulation
from validation.checks import RecordData, make_recorder
from validation.scenarios import SCENARIOS, Scenario


def run_scenario(
    scenario: Scenario,
    progress: Progress,
    callbacks: list[Callable[[Simulation], None]],
) -> RecordData:
    sim = Simulation(scenario["N"], **scenario["params"])
    for attr, val in scenario.get("initial", {}).items():
        setattr(sim, attr, np.array(val, dtype=np.float64))

    recorder, data = make_recorder(scenario["N"], sim.positions)
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
        "speeds": [s.tolist() for s in data["speeds"]],
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
        TextColumn("{task.description:<30}"), BarColumn(),
        TextColumn("{task.completed:.0f}/{task.total:.0f}"),
    ) as progress:
        for scenario in SCENARIOS:
            data = run_scenario(scenario, progress, callbacks)
            save_data(scenario, data)


if __name__ == "__main__":
    parser = ArgumentParser(prog="Validation")
    parser.add_argument("-v", "--visualise", action="store_true",
                        help="Live 3D visualisation.")
    args = parser.parse_args()

    if args.visualise:
        from simulation.visualisation import VisualisationManager
        from PyQt5.QtWidgets import QApplication
        vis = VisualisationManager(None, max(s["N"] for s in SCENARIOS))  # type: ignore
        Thread(target=run_all, args=(vis.update,), daemon=True).start()
        QApplication([]).exec_()
    else:
        run_all()

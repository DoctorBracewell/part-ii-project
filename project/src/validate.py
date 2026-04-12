import json
import os
from argparse import ArgumentParser
from threading import Thread
from typing import Any, Callable
import numpy as np
from rich.progress import Progress, TextColumn, BarColumn

from configs import simulation as SimulationConfig
from simulation.simulation import Simulation
from validation.scenarios import SCENARIOS, Scenario
from validation.checks import RecordData, make_recorder, summarise_and_plot

OUTPUT_DIR = "results/validation"


def run_scenario(
    scenario: Scenario,
    progress: Progress,
    callbacks: list[Callable[[Simulation], None]],
) -> tuple[RecordData, int]:
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
    return data, sim.timestep


def save_data(scenario: Scenario, data: RecordData, output_dir: str) -> None:
    serialisable: dict[str, Any] = {
        "initial_positions": data["initial_positions"].tolist(),
        "positions": [p.tolist() for p in data["positions"]],
        "speeds": [s.tolist() for s in data["speeds"]],
        "azimuth_angles": [a.tolist() for a in data["azimuth_angles"]],
        "active": [a.tolist() for a in data["active"]],
    }
    path = os.path.join(output_dir, f"{scenario['id']:02d}_{scenario['name']}.json")
    with open(path, "w") as f:
        json.dump(serialisable, f)


def run_all(visualise: bool) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    def sim_thread() -> None:
        with Progress(TextColumn("{task.description:<30}"), BarColumn(),
                      TextColumn("{task.completed:.0f}/{task.total:.0f}")) as progress:
            for scenario in SCENARIOS:
                from simulation.visualisation import VisualisationManager
                vis = VisualisationManager(None, scenario["N"]) if visualise else None  # type: ignore
                callbacks: list[Callable[[Simulation], None]] = [vis.update] if vis else []
                data, _ = run_scenario(scenario, progress, callbacks)
                save_data(scenario, data, OUTPUT_DIR)
                summarise_and_plot(scenario, data, OUTPUT_DIR)

    if visualise:
        Thread(target=sim_thread, daemon=True).start()
        from PyQt5.QtWidgets import QApplication
        QApplication([]).exec_()
    else:
        sim_thread()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--visualise", action="store_true",
                        help="Show live 3D visualisation for each scenario")
    args = parser.parse_args()
    run_all(args.visualise)

from simulation.simulation import SimulationManager
from visualisation import VisualisationManager
from configs import simulation as SimulationConfig
from logging import Logger
from numpy import linspace
from rich.progress import Progress, TextColumn, BarColumn
from pathlib import Path
from typing import Any
import copy
import os
import json
from threading import Thread

from configs.parameters import BASE as base

M = SimulationConfig.MACH
inputs: dict[str, tuple[float, float, int]] = {
    "velocity_maxs": (0.3 * M, 0.4 * M, 4),
    "azimuth_rate_mins": (-1.3, -3.14, 4),
    "azimuth_rate_maxs": (1.3, 3.14, 4),
    "attack_angle_maxs": (0.52, 1.57, 4),
    "thrust_ratio": (10, 15, 4),
    "roll_angle_ratio": (1, 1.5, 4),
}

parameters: dict[str, list[float]] = {
    k: [float(x) for x in linspace(*v)] for k, v in inputs.items()
}

RESULTS_FILE = Path(__file__).parent.parent / "results" / "tune_results.json"


def tune(logger: Logger):
    simulation_manager = SimulationManager(logger)
    visulisation_manager = VisualisationManager(logger, 3)

    results: list[dict[str, Any]] = json.loads(RESULTS_FILE.read_text()) if RESULTS_FILE.exists() else []
    completed = {(r["parameter"], round(float(r["value"]), 8)) for r in results}

    if results:
        logger.info(f"Resuming: {len(results)} runs already complete.")

    def save(parameter: str, value: float, captures: int, steps: int):
        results.append({"parameter": parameter, "value": value, "captures": captures, "steps": steps})
        RESULTS_FILE.write_text(json.dumps(results, indent=2))

    def run_simulation():
        # --- 1. BASE RUN ---
        if ("base", round(-1.0, 8)) not in completed:
            logger.info(f"Base parameters: {base}\n")
            progress = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("step {task.completed:.0f}/{task.total:.0f}"))
            print()
            progress.start()
            task = progress.add_task("base", total=4000)
            simulation_manager.setup(base)
            steps, captures = simulation_manager.run(lambda _: progress.update(task, advance=1), visulisation_manager.update)
            progress.stop()
            save("base", -1, captures, steps)
            logger.info(f"Captures {captures} by step {steps} (base)." if captures else f"No capture by step {steps} (base).")

        # --- 2. PARAMETER SWEEP ---
        for parameter, values in parameters.items():
            for value in values[1:]:  # skip base value
                if (parameter, round(value, 8)) in completed:
                    continue

                test_parameters = copy.deepcopy(base)
                test_parameters[parameter][-1] = value  # type: ignore

                progress = Progress(TextColumn("{task.description}"), BarColumn(), TextColumn("step {task.completed:.0f}/{task.total:.0f}"))
                print()
                progress.start()
                task = progress.add_task(f"{parameter}={value:.3f}", total=4000)
                simulation_manager.setup(test_parameters)
                steps, captures = simulation_manager.run(lambda _: progress.update(task, advance=1), visulisation_manager.update)
                progress.stop()
                save(parameter, value, captures, steps)
                logger.info(f"Captures {captures} by step {steps} with {parameter}={value:.3f}." if captures else f"No capture by step {steps} with {parameter}={value:.3f}.")

        logger.info(f"Tuning complete. Results saved to {RESULTS_FILE}")

    Thread(target=run_simulation, daemon=True).start()

    from PyQt5.QtWidgets import QApplication
    app = QApplication([])
    app.exec_()
    os._exit(0)


if __name__ == "__main__":
    import logging
    from rich.logging import RichHandler
    from rich.console import Console

    logging.basicConfig(
        level="INFO", format="%(message)s", datefmt="[%X]",
        handlers=[RichHandler(console=Console(), show_time=True, show_level=True, show_path=True)],
    )
    tune(logging.getLogger("rich"))

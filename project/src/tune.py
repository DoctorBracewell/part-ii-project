from simulation.simulation import SimulationManager, Simulation
from visualisation import VisualisationManager
from configs import simulation as SimulationConfig
from logging import Logger
from numpy import linspace
from itertools import product
from rich.progress import Progress, TextColumn, BarColumn
import copy
import os
from threading import Thread

from configs.parameters import BASE as base

M = SimulationConfig.MACH
inputs: dict[str, tuple[float, float, int]] = {
    "velocity_maxs": (0.3 * M, 0.4 * M, 2),
    "azimuth_rate_mins": (-1.3, -1.5, 2),
    "azimuth_rate_maxs": (1.3, 1.5, 2),
    "attack_angle_maxs": (0.52, 0.69, 2),
    "thrust_ratio": (10, 15, 3),
    "roll_angle_ratio": (1, 1.5, 2),
}

parameters: dict[str, list[float] | list[list[float]]] = {
    "velocity_maxs": [float(x) for x in linspace(*inputs["velocity_maxs"])],
    "azimuth_rate_mins": [float(x) for x in linspace(*inputs["azimuth_rate_mins"])],
    "azimuth_rate_maxs": [float(x) for x in linspace(*inputs["azimuth_rate_maxs"])],
    "attack_angle_maxs": [float(x) for x in linspace(*inputs["attack_angle_maxs"])],
    "thrust_ratio": [float(x) for x in linspace(*inputs["thrust_ratio"])],
    "roll_angle_ratio": [float(x) for x in linspace(*inputs["roll_angle_ratio"])],
}


def tune(logger: Logger):
    simulation_manager = SimulationManager(logger)
    visulisation_manager = VisualisationManager(logger, 3)
    results = []

    def run_simulation():
        logger.info(f"Base parameters: {base}\n")
        progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("step {task.completed:.0f}/{task.total:.0f}"),
        )
        print()
        progress.start()
        task = progress.add_task("base", total=1000)

        def update(simulation: Simulation):
            progress.update(task, advance=1)

        simulation_manager.setup(base)
        steps, captures = simulation_manager.run(update, visulisation_manager.update)

        results.append(("base", -1, captures, steps))
        progress.stop()

        if captures:
            logger.info(
                f"Captures {captures} by timestep {steps} with base parameters!"
            )
        else:
            logger.info(f"No capture by timestep {steps} with base parameters.")

    sim_thread = Thread(target=run_simulation, daemon=True)
    sim_thread.start()

    from PyQt5.QtWidgets import QApplication

    app = QApplication([])
    app.exec_()
    os._exit(0)

    for parameter, values in parameters.items():
        for value in values[1:]:  # skip the first value since it's the base
            test_parameters = copy.deepcopy(base)
            test_parameters[parameter][-1] = value  # type: ignore

            progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                TextColumn("step {task.completed:.0f}/{task.total:.0f}"),
            )
            print()
            progress.start()
            task = progress.add_task(f"{parameter}={value:.1f}", total=1000)

            def update(simulation: Simulation):
                progress.update(task, advance=1)

            simulation_manager.setup(test_parameters)
            steps, captures = simulation_manager.run(update)

            results.append((parameter, value, captures, steps))
            progress.stop()

            if captures:
                logger.info(
                    f"Captures {captures} by timestep {steps} with {parameter}={value:.1f}!"
                )
            else:
                logger.info(
                    f"No capture by timestep {steps} with {parameter}={value:.1f}."
                )

    logger.info(f"\nTuning results: {results}")


if __name__ == "__main__":
    import logging
    from rich.logging import RichHandler
    from rich.console import Console

    console = Console()
    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=console, show_time=True, show_level=True, show_path=True
            )
        ],
    )
    tune(logging.getLogger("rich"))

from simulation.simulation import SimulationManager, Simulation
from simulation.visualisation import VisualisationManager
from configs import simulation as SimulationConfig
from logging import Logger
from numpy import linspace
from itertools import product
from rich.progress import Progress, TextColumn, BarColumn
import copy
import os
from threading import Thread

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

base: dict[str, list[float] | list[list[float]]] = {
    "positions": [[5000, 4000, 6500], [5000, 6000, 6500], [5000, 5000, 6500]],
    "velocity_mins": [0.1 * M] * 3,
    "velocity_maxs": [inputs["velocity_maxs"][0]] * 3,
    "azimuth_rate_mins": [inputs["azimuth_rate_mins"][0]] * 3,
    "azimuth_rate_maxs": [inputs["azimuth_rate_maxs"][0]] * 3,
    "attack_angle_mins": [0.09] * 3,
    "attack_angle_maxs": [inputs["attack_angle_maxs"][0]] * 3,
    "thrust_ratio": [10, 10, 10],  # [inputs["thrust_ratio"][0]] * 2,
    "attack_angle_ratio": [1.5] * 3,
    "roll_angle_ratio": [inputs["roll_angle_ratio"][0]] * 3,
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
            logger.info(f"Captures {captures} by timestep {steps} with base parameters!")
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
                logger.info(f"Captures {captures} by timestep {steps} with {parameter}={value:.1f}!")
            else:
                logger.info(f"No capture by timestep {steps} with {parameter}={value:.1f}.")

    logger.info(f"\nTuning results: {results}")

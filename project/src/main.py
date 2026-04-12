import logging
import os
from argparse import ArgumentParser, Namespace
from contextlib import nullcontext
from threading import Thread
from typing import Callable

from rich.console import Console
from rich.logging import RichHandler

from configs import simulation as SimulationConfig
from display import Display
from outputs.base import OutputManager
from simulation.simulation import Simulation, SimulationManager
from configs.parameters import BASE

console = Console()
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            console=console,
            show_time=True,
            show_level=True,
            show_path=True,
            enable_link_path=True,
        )
    ],
)
logger = logging.getLogger("rich")

parser = ArgumentParser(prog="Pursuit-Evasion Simulation")
parser.add_argument(
    "-d", "--display", action="store_true", help="Live terminal dashboard."
)
parser.add_argument(
    "-o", "--outputs", action="store_true", help="Save plots and video."
)
parser.add_argument(
    "-v", "--visualisation", action="store_true", help="Live 3D visualisation."
)


def run(
    args: Namespace, vis_update: Callable[[Simulation], None] | None = None
) -> None:
    callbacks: list[Callable[[Simulation], None]] = []

    if vis_update:
        callbacks.append(vis_update)

    output_manager = OutputManager(logger) if args.outputs else None
    if output_manager:
        callbacks.append(output_manager.add_agent_data)

    with Display(console) if args.display else nullcontext() as display:
        if display:
            callbacks.append(display.update)

        simulation_manager = SimulationManager(logger)
        simulation_manager.setup(BASE)
        steps, captures = simulation_manager.run(*callbacks)

    if output_manager:
        output_manager.create_outputs()

    logger.info(
        f"Captures {captures} by timestep {steps}."
        if captures
        else f"No capture by timestep {steps}."
    )


if __name__ == "__main__":
    # setup args
    args = parser.parse_args()

    # the pyvista simulation MUST run in the main thread, so we run the simulation in a separate if it's active
    try:
        if args.visualisation:
            from simulation.visualisation import VisualisationManager
            from PyQt5.QtWidgets import QApplication

            vis = VisualisationManager(logger, SimulationConfig.AGENTS)
            Thread(target=run, args=(args, vis.update), daemon=True).start()
            QApplication([]).exec_()
            os._exit(0)
        else:
            run(args)
    except KeyboardInterrupt:
        pass

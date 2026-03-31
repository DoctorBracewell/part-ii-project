import logging
import os

from rich.logging import RichHandler
from rich.console import Console

from argparse import ArgumentParser
from contextlib import nullcontext
from threading import Thread
from typing import Callable

from display import Display
from tune import tune, base
from simulation.simulation import SimulationManager, Simulation
from configs import simulation as SimulationConfig

from outputs.base import OutputManager
from simulation.visualisation import VisualisationManager

# Logging
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

# Arguments
parser = ArgumentParser(
    prog="Pursuit-Evasion Simulation",
    description="Simulate pursuit-evasion agents in a 3D environment.",
)

parser.add_argument(
    "-d", "--display", action="store_true", help="Enable live terminal dashboard."
)
parser.add_argument(
    "-o", "--outputs", action="store_true", help="Enable output generation."
)
parser.add_argument(
    "-v", "--visualisation", action="store_true", help="Open interactive visualisation."
)
parser.add_argument("--tune", action="store_true", help="Run tuning")


def main():
    simulation_manager = SimulationManager(logger)
    visualisation_manager = VisualisationManager(logger, SimulationConfig.AGENTS)
    args = parser.parse_args()

    if args.tune:
        tune(logger)
        return

    def run_simulation():
        callbacks: list[Callable[[Simulation], None]] = []
        if args.visualisation:
            callbacks.append(visualisation_manager.update)

        output_manager = OutputManager(logger) if args.outputs else None
        if output_manager:
            callbacks.append(output_manager.add_agent_data)

        with (Display(console) if args.display else nullcontext()) as display:
            if display:
                callbacks.append(display.update)

            simulation_manager.setup(base)
            steps, captures = simulation_manager.run(*callbacks)

        if output_manager:
            output_manager.create_outputs()

        if captures:
            logger.info(f"Captures {captures} by timestep {steps}.")
        else:
            logger.info(f"No capture by timestep {steps}.")

    try:
        if args.visualisation:
            sim_thread = Thread(target=run_simulation, daemon=True)
            sim_thread.start()

            from PyQt5.QtWidgets import QApplication

            app = QApplication([])
            app.exec_()
            os._exit(0)
        else:
            run_simulation()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

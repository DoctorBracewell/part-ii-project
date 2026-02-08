import logging
import os

from rich.logging import RichHandler
from rich.console import Console
from typing import Callable

from argparse import ArgumentParser
from threading import Thread

from display import Display
from simulation.simulation import Simulation, SimulationManager

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


def main():
    output_manager = OutputManager(logger)
    simulation_manager = SimulationManager(logger)
    visualisation_manager = VisualisationManager(
        logger, simulation_manager.simulation.N
    )
    args = parser.parse_args()

    try:
        logger.info("Simulation begun with t=0")

        def run_simulation():
            callbacks: list[Callable[[Simulation], None]] = []
            if args.outputs:
                callbacks.append(output_manager.add_agent_data)
            if args.visualisation:
                callbacks.append(visualisation_manager.update)

            if args.display:
                with Display(console) as display:
                    simulation_manager.run(display.update, *callbacks)
            else:
                simulation_manager.run(*callbacks)

        sim_thread = Thread(target=run_simulation, daemon=True)
        sim_thread.start()

        from PyQt5.QtWidgets import QApplication

        app = QApplication([])
        app.exec_()
        os._exit(0)

    except KeyboardInterrupt:
        pass

    finally:
        logger.info(
            "Simulation ended with t=%d", simulation_manager.simulation.timestep
        )

        if args.outputs:
            try:
                output_manager.create_outputs()
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()

import logging

from rich.logging import RichHandler
from rich.console import Console

from argparse import ArgumentParser

from display import Display
from simulation.simulation import SimulationManager

from outputs.base import OutputManager

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
    args = parser.parse_args()

    try:
        logger.info("Simulation begun with t=0")

        if args.display:
            with Display(console) as display:
                simulation_manager.run(display.update, output_manager.add_agent_data)
        else:
            simulation_manager.run(output_manager.add_agent_data)

    except KeyboardInterrupt:
        pass

    finally:
        logger.info(
            "Simulation ended with t=%d", simulation_manager.simulation.timestep
        )

        if args.outputs:
            try:
                output_manager.create_outputs(args.visualisation)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()

import logging

from rich.logging import RichHandler
from rich.console import Console

from display import Display
from simulation.simulation import Simulation
from outputs.base import OutputManager

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


def main():
    with Display(console) as display:
        output_manager = OutputManager(logger)
        simulation = Simulation(logger)

        # Run the simulation with display and output_manager callbacks
        simulation.run(display.update, output_manager.add_agent_positions)

        output_manager.create_outputs()


if __name__ == "__main__":
    main()

import logging
import sys

from rich.logging import RichHandler
from rich.console import Console

from display import Display
from simulation.simulation import SimulationManager

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
    output_manager = OutputManager(logger)
    simulation_manager = SimulationManager(logger)

    try:
        logger.info("Simulation begun with t=0")

        if "--no-display" in sys.argv:
            simulation_manager.run(output_manager.add_agent_data)
        else:
            with Display(console) as display:
                simulation_manager.run(display.update, output_manager.add_agent_data)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info(
            "Simulation ended with t=%d", simulation_manager.simulation.timestep
        )
        if "--no-output" not in sys.argv:
            output_manager.create_outputs()


if __name__ == "__main__":
    main()

import logging

from rich.logging import RichHandler
from rich.console import Console

from display import Display
from simulation import Simulation
from plotter import Plotter

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
        simulation = Simulation(logger)
        simulation.run(display.update)

    plotter = Plotter()
    plotter.plot()
    plotter.save_to_directory()


if __name__ == "__main__":
    main()

import logging
from rich.logging import RichHandler

from simulation import Simulation
from plotter import Plotter

# Configure logging with RichHandler
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(
            show_time=True,
            show_level=True,
            show_path=True,
            enable_link_path=True,
        )
    ],
)

# Create a logger instance
logger = logging.getLogger("rich")


def main():
    logger.info("logging enabled")

    simulation = Simulation(logger)
    simulation.run()

    plotter = Plotter()
    plotter.plot()
    plotter.save_to_directory()


if __name__ == "__main__":
    main()

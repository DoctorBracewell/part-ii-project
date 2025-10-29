from logging import Logger
from typing import Callable
from time import sleep


class SimulationStatus:
    timestep: int = 0
    agents: int = 0
    hard_deck: int = 0


class Simulation:
    status: SimulationStatus

    def __init__(self, logger: Logger):
        self.logger = logger
        self.status = SimulationStatus()

    def run(self, *callbacks: Callable[[SimulationStatus], None]):
        self.logger.info("simulation started")

        for _ in range(100):
            self.logger.info("stepped time!")
            for callback in callbacks:
                callback(self.status)
                self.status.timestep += 1
                sleep(0.05)

        self.logger.info("simulation finished")

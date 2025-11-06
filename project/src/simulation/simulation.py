from logging import Logger
from typing import Callable
from time import sleep
from simulation.agents import Agent
from config import SimulationConfig
from math import sin


class SimulationStatus:
    timestep: int = 0
    agents: int = 2
    hard_deck: int = SimulationConfig.HARD_DECK
    pursuer: Agent
    evader: Agent

    def __init__(self, pursuer: Agent, evader: Agent):
        self.pursuer = pursuer
        self.evader = evader


class Simulation:
    status: SimulationStatus
    pursuer: Agent
    evader: Agent

    def __init__(self, logger: Logger):
        self.logger = logger

        self.pursuer = Agent(position=0.0)
        self.evader = Agent(position=5.0)

        self.status = SimulationStatus(self.pursuer, self.evader)

    def run(self, *callbacks: Callable[[SimulationStatus], None]):
        self.logger.info("simulation started")

        self.pursuer.velocity = -1  # metres/second
        self.evader.velocity = 0  # metres/second
        self.pursuer.acceleration = 0.1

        for i in range(10 * SimulationConfig.STEPS_PER_SECOND):
            self.pursuer.move(1 / SimulationConfig.STEPS_PER_SECOND)
            self.evader.move(1 / SimulationConfig.STEPS_PER_SECOND)

            self.logger.info(i)

            self.status.timestep += 1

            for callback in callbacks:
                callback(self.status)

        self.logger.info("simulation finished")

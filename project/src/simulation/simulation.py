from logging import Logger
from typing import Callable

from simulation.agent import Agent
from simulation.mdp import Simulation


class SimulationManager:
    logger: Logger
    simulation: Simulation

    def __init__(self, logger: Logger):
        self.logger = logger

        # Initialise agents outside of simulation to reserve memory for numba
        pursuer = Agent(0)
        evader = Agent(1)
        self.simulation = Simulation(pursuer, evader)

    def run(self, *callbacks: Callable[[Simulation], None]):
        while True:
            self.simulation.step()

            for callback in callbacks:
                callback(self.simulation)

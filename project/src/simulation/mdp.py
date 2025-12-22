import numpy as np
from numpy.typing import NDArray
from logging import Logger

from simulation.agent import Agent


# The MDP
class MDP:
    world_state: NDArray[np.float64]
    agents: list[Agent]

    def __init__(self, logger: Logger, *args: Agent):
        self.logger = logger
        self.agents = list(args)
        # self.world_state =

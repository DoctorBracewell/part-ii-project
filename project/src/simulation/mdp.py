from __future__ import annotations

# import numpy as np
# from numpy.typing import NDArray
from logging import Logger
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from simulation.agent import Agent
    from simulation.simulation import SimulationStatus, SimulationStatusState

type State = tuple[float, float, list[tuple[float, float]]]
"""(self_pos, self_vel, [(other_pos, other_vel), ...]"""


# The MDP
class MDP:
    state: State
    agent: Agent
    other_agents: list[Agent]

    def __init__(self, logger: Logger, agent: Agent, status: SimulationStatus):
        self.logger = logger

        self.agent = agent
        self.other_agents = [
            a for a in [status.pursuer, status.evader] if a is not agent
        ]

    #     self.set_state_from_status_state(status.get_state())

    # def set_state_from_status_state(self, agent: Agent, state: SimulationStatusState):

    #     self.state = (
    #         agent.position,
    #         agent.velocity,
    #         [(a.position, a.velocity) for a in state[]],
    #     )

    # def find_action(self):
    #     one_step =

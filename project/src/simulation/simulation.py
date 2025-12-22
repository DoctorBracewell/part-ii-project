from logging import Logger
from typing import Callable

from config import SimulationConfig
from simulation.agent import Agent, Action
from simulation.mdp import MDP

type SimulationStatusState = tuple[
    int, int, int, tuple[int, float, float, float], tuple[int, float, float, float]
]
"""(timestep, agent_count, hard_deck, pursuer_state, evader_state)"""


class SimulationStatus:
    timestep: int = 0
    agent_count: int = 2
    hard_deck: int = SimulationConfig.HARD_DECK
    pursuer: Agent
    evader: Agent

    def __init__(self):
        self.pursuer = Agent()
        self.evader = Agent()
        self.pursuer.randomise()
        self.evader.randomise()

    def get_state(
        self,
    ) -> SimulationStatusState:
        """
        :return: (timestep, agent_count, hard_deck, pursuer_state, evader_state)
        """
        return (
            self.timestep,
            self.agent_count,
            self.hard_deck,
            self.pursuer.get_state(),
            self.evader.get_state(),
        )

    def set_state(
        self,
        state: SimulationStatusState,
    ):
        """
        :param state: (timestep, agent_count, hard_deck, pursuer_state, evader_state)
        """
        (
            self.timestep,
            self.agent_count,
            self.hard_deck,
            pursuer_state,
            evader_state,
        ) = state
        self.pursuer.set_state(pursuer_state)
        self.evader.set_state(evader_state)


class Simulation:
    def __init__(self, logger: Logger):
        self.logger = logger

        self.status = SimulationStatus()

    def run(self, *callbacks: Callable[[SimulationStatus], None]):
        self.logger.info("simulation started")

        while True:
            self.step()

            for callback in callbacks:
                callback(self.status)

    def step(self):
        # determine and set action for each agent
        for agent in [self.status.pursuer, self.status.evader]:
            mdp = MDP(self.logger, agent, self.status)
            # action = mdp.find_action()

        # step each agent with that action
        self.status.pursuer.step()
        self.status.evader.step()

        # increase timestep
        self.status.timestep += 1

    def forward_project(self, agent: Agent, action: Action, time: float):
        steps = int(time * SimulationConfig.STEPS_PER_SECOND)
        initial_state = self.status.get_state()

        agent.update_from_action(action)

        for _ in range(steps):
            self.status.pursuer.step()
            self.status.evader.step()

        # return final state reached and reset
        final_state = self.status.get_state()
        self.status.set_state(initial_state)
        return final_state

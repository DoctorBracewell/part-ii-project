from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from configs import mdp as MDPConfig
from simulation.agent import Agent, actions

from typing import TYPE_CHECKING, Callable

from numba import float64, njit
from numba.typed import List
from numba.types import ListType
from numba.experimental import jitclass

from configs import simulation as SimulationConfig
from simulation.agent import Agent, Action, AgentState

# if TYPE_CHECKING:
#     from simulation.simulation import Simulation, SimulationState


# Reward parameters
positive_magnitudes: NDArray[np.float64] = np.array([200])
positive_discounts: NDArray[np.float64] = np.array([0.999])


@njit
def positive_positions(p: float, _: float) -> NDArray[np.float64]:
    return np.array([p])


timesteps = [0, 1, 5, 10]
negative_magnitudes: NDArray[np.float64] = np.array([300 for _ in timesteps])
negative_discounts: NDArray[np.float64] = np.array([0.99 for _ in timesteps])
negative_positions: Callable[[float, float], NDArray[np.float64]] = (
    lambda p, v: np.array([p + v * t for t in timesteps])
)
negative_radii: Callable[[float, float], NDArray[np.float64]] = lambda _, v: np.array(
    [v * t for t in timesteps]
)


@jitclass
class Simulation:
    agent_id_counter: int
    timestep: int
    agent_count: int
    hard_deck: int
    pursuer: Agent
    evader: Agent

    def __init__(self, pursuer: Agent, evader: Agent):
        self.agent_id_counter = 0
        self.timestep = 0
        self.agent_count = 2
        self.hard_deck = SimulationConfig.HARD_DECK

        self.pursuer = pursuer
        self.evader = evader
        self.pursuer.randomise()
        self.evader.randomise()

    def step(self):
        # determine and set action for each agent
        agents: ListType[Agent] = List()
        agents.append(self.pursuer)
        agents.append(self.evader)

        for agent in agents:
            mdp = MDP(self, agent)
            # action = mdp.find_action()

        # step each agent with that action
        self.pursuer.step()
        self.evader.step()

        # increase timestep
        self.timestep += 1

    def forward_project(self, agent: Agent, action: Action, steps: int):
        initial_state = self.get_state()

        agent.update_from_action(action)

        for _ in range(steps):
            self.pursuer.step()
            self.evader.step()

        # return final state reached and reset
        final_state = self.get_state()
        self.set_state(initial_state)
        return final_state

    def get_state(
        self,
    ) -> SimulationState:
        return (
            self.timestep,
            self.agent_count,
            self.hard_deck,
            self.pursuer.get_state(),
            self.evader.get_state(),
        )

    def set_state(
        self,
        state: SimulationState,
    ):
        (
            self.timestep,
            self.agent_count,
            self.hard_deck,
            pursuer_state,
            evader_state,
        ) = state
        self.pursuer.set_state(pursuer_state)
        self.evader.set_state(evader_state)


# The MDP
@jitclass(spec=[("other_agents", float64[:, :])])  # type: ignore
class MDP:
    agent: Agent
    simulation: Simulation

    # Internal State
    self_agent_id: int
    self_agent_pos: float
    self_agent_vel: float
    other_agents: NDArray[np.float64]

    def __init__(self, simulation: Simulation, agent: Agent):
        self.simulation = simulation
        self.agent = agent
        self.self_agent_id = agent.id

        self.set_state_from_simulation_state(self.simulation.get_state())

    def set_state_from_simulation_state(self, state: SimulationState):
        # Determine self and other agents
        pursuer = state[3]
        evader = state[4]
        agent = pursuer if self.self_agent_id == pursuer[0] else evader
        other_agent = pursuer if self.self_agent_id != pursuer[0] else evader

        # Update internal state
        self.self_agent_pos = agent[1]
        self.self_agent_vel = agent[2]
        # nopython-safe method of allocating 2D array
        n = 1  # number of other agents
        self.other_agents = np.empty((n, 2), dtype=np.float64)
        self.other_agents[0, 0] = other_agent[1]
        self.other_agents[0, 1] = other_agent[2]

    def find_action(self):
        best_action = None
        best_result = -float("inf")

        for action in actions:
            # forward project
            horizon = self.simulation.forward_project(
                self.agent,
                action,
                MDPConfig.FORWARD_PROJECTION_STEPS,
            )

            self.set_state_from_simulation_state(horizon)

        # rewards
        # best_positive_reward = self.positive_maximum()
        # return np.array([best_positive_reward])

        # best_negative_reward = self.negative_maximum()

    # def positive_maximum(self):
    #     best_value = -float("inf")

    #     for p, v in self.other_agents:
    #         # numpy arrays for vectorised computation
    #         positions = positive_positions(p, v)
    #         distances = np.abs(positions - self.self_agent_pos)
    #         values = positive_magnitudes * (positive_discounts**distances)
    #         max_value = np.max(values)

    #         if max_value > best_value:
    #             best_value = max_value

    #     return best_value

    # def negative_maximum(self):
    #     best_value = -float("inf")

    #     for p, v in self.other_agents:
    #         # numpy arrays for vectorised computation


type SimulationState = tuple[int, int, int, AgentState, AgentState]
"""(timestep, agent_count, hard_deck, pursuer_state, evader_state)"""

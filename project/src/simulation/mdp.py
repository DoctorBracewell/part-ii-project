from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from configs import mdp as MDPConfig
from configs import simulation as SimulationConfig

from simulation.agent import Position, Agent, Action, AgentState, actions

# Reward parameters
positive_magnitudes: NDArray[np.float64] = np.array([200])
positive_discounts: NDArray[np.float64] = np.array([0.999])
positive_positions: Callable[[float, float], NDArray[np.float64]] = (
    lambda p, _: np.array([p])
)


timesteps = [0, 1, 5, 10]
negative_magnitudes: NDArray[np.float64] = np.array([300 for _ in timesteps])
negative_discounts: NDArray[np.float64] = np.array([0.99 for _ in timesteps])
negative_positions: Callable[[float, float], NDArray[np.float64]] = (
    lambda p, v: np.array([p + v * t for t in timesteps])
)
negative_radii: Callable[[float, float], NDArray[np.float64]] = lambda _, v: np.array(
    [v * t for t in timesteps]
)


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
        for agent in [self.pursuer, self.evader]:
            mdp = MDP(self, agent)
            action = mdp.find_action()
            agent.next_action = action

        for agent in [self.pursuer, self.evader]:
            agent.update_from_next_action()

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
class MDP:
    agent: Agent
    simulation: Simulation

    # Internal State
    self_agent_id: int
    self_agent_pos: float
    self_agent_vel: float
    other_agents: list[tuple[Position, float]]

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
        self.other_agents = [(other_agent[1], other_agent[2])]

    def find_action(self):
        best_action = None
        best_reward = -float("inf")

        for action in actions:
            # forward project
            horizon = self.simulation.forward_project(
                self.agent,
                action,
                MDPConfig.FORWARD_PROJECTION_STEPS,
            )

            self.set_state_from_simulation_state(horizon)

            # rewards
            best_positive_reward = self.positive_maximum()
            best_negative_reward = self.negative_maximum()
            hard_deck_penalty = self.hard_deck_penalty()

            total_reward = (
                best_positive_reward - best_negative_reward - hard_deck_penalty
            )

            if total_reward > best_reward:
                best_reward = total_reward
                best_action = action

        return best_action

    def positive_maximum(self):
        best_value = -np.inf

        for p, v in self.other_agents:
            # numpy arrays for vectorised computation
            positions = positive_positions(p, v)
            distances = np.abs(positions - self.self_agent_pos)
            values = positive_magnitudes * (positive_discounts**distances)
            max_value = np.max(values)

            if max_value > best_value:
                best_value = max_value

        return best_value

    def negative_maximum(self):
        best_value = -np.inf

        for p, v in self.other_agents:
            # numpy arrays for vectorised computation
            positions = negative_positions(p, v)
            radii = negative_radii(p, v)
            distances = np.abs(positions - self.self_agent_pos)
            within_radius = distances <= radii
            values = (
                within_radius * negative_magnitudes * (negative_discounts**distances)
            )
            max_value = np.max(values)

            if max_value > best_value:
                best_value = max_value

        return best_value

    def hard_deck_penalty(self):
        return 0.0


type SimulationState = tuple[int, int, int, AgentState, AgentState]
"""(timestep, agent_count, hard_deck, pursuer_state, evader_state)"""

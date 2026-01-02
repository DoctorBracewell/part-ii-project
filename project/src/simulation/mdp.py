from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from configs import mdp as MDPConfig

if TYPE_CHECKING:
    from simulation.simulation import Vectors, Scalars, Vector

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

# Actions
type Action = NDArray[np.float64]
thrusts: list[float] = [50.0]
rotation_rates: list[float] = [-1, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1]
actions: NDArray[np.float64] = np.array(
    [[thrust, rotation_rate] for thrust in thrusts for rotation_rate in rotation_rates]
)


class MDP:
    def __init__(
        self,
        i: int,
        positions: Vectors,
        velocities: Vectors,
        accelerations: Vectors,
        headings: Scalars,
        thrusts: Scalars,
        rotation_rates: Scalars,
    ):
        self.i = i
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        self.headings = headings

        self.thrusts = thrusts
        self.rotation_rates = rotation_rates

    def find_action(self) -> Action:
        TIMESTEPS = MDPConfig.FORWARD_PROJECTION_STEPS

        # Forward project all agents forward
        all_projected_positions = self.forward_project_all(TIMESTEPS)

        # Forward project self agent for each actions
        num_actions = actions.shape[0]
        rewards = np.zeros(num_actions)
        for i in range(num_actions):
            # Choose action
            action = actions[i]
            # Project self agent forward with that action
            one_projected_position = self.forward_project_one(action, TIMESTEPS)
            # Calculate reward from resulting position
            rewards[i] = self.calculate_reward(
                one_projected_position, all_projected_positions
            )

        # Choose best action
        best_action = actions[np.argmax(rewards)]
        return best_action

    def forward_project_all(self, steps: int) -> Vectors:
        from simulation.simulation import step_agents

        # start with current state
        positions, velocities, accelerations, headings = (
            self.positions,
            self.velocities,
            self.accelerations,
            self.headings,
        )

        # forward project
        for _ in range(steps):
            # update
            positions, velocities, accelerations, headings = step_agents(
                positions,
                velocities,
                accelerations,
                headings,
                self.thrusts,
                self.rotation_rates,
            )

        return positions

    def forward_project_one(self, action: Action, steps: int) -> Vector:
        from simulation.simulation import step_agents

        thrust, rotation_rate = action

        # start with current state
        position, velocity, acceleration, heading = (
            self.positions[self.i],
            self.velocities[self.i],
            self.accelerations[self.i],
            self.headings[self.i],
        )

        # forward project
        for _ in range(steps):
            # update
            position, velocity, acceleration, heading = step_agents(
                position, velocity, acceleration, heading, thrust, rotation_rate
            )

        return position

    def calculate_reward(self, self_position: Vector, all_positions: Vectors) -> float:
        # best_positive_reward = self.positive_maximum()
        # best_negative_reward = self.negative_maximum()
        # hard_deck_penalty = self.hard_deck_penalty()

        # total_reward = best_positive_reward - best_negative_reward - hard_deck_penalty
        total_reward = 0.0

        return total_reward

    # def positive_maximum(self):
    #     best_value = -np.inf

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
    #     best_value = -np.inf

    #     for p, v in self.other_agents:
    #         # numpy arrays for vectorised computation
    #         positions = negative_positions(p, v)
    #         radii = negative_radii(p, v)
    #         distances = np.abs(positions - self.self_agent_pos)
    #         within_radius = distances <= radii
    #         values = (
    #             within_radius * negative_magnitudes * (negative_discounts**distances)
    #         )
    #         max_value = np.max(values)

    #         if max_value > best_value:
    #             best_value = max_value

    #     return best_value

    def hard_deck_penalty(self):
        return 0.0

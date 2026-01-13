from __future__ import annotations
from typing import Callable, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from configs import mdp as MDPConfig
from configs import simulation as SimulationConfig

if TYPE_CHECKING:
    from simulation.simulation import Vectors, Scalars, Vector

# Actions
type Action = NDArray[np.float64]
thrusts = np.arange(0.0, 7.0, 1.0, dtype=np.float64)
rotation_rates = np.arange(-1.5, 1.5, 0.1, dtype=np.float64)
actions: NDArray[np.float64] = np.array(
    [
        [thrust * 5, rotation_rate]
        for thrust in thrusts
        for rotation_rate in rotation_rates
    ]
)


class MDP:
    def __init__(
        self,
        i: int,
        positions: Vectors,
        velocities: Vectors,
        headings: Scalars,
        thrusts: Scalars,
        rotation_rates: Scalars,
        projected_positions: Vectors,
        projected_velocities: Vectors,
    ):
        self.i = i
        self.positions = positions
        self.velocities = velocities
        self.headings = headings

        self.thrusts = thrusts
        self.rotation_rates = rotation_rates

        self.projected_positions = projected_positions.copy()
        self.projected_velocities = projected_velocities.copy()

    def find_action(self) -> Action:
        from simulation.simulation import forward_project

        # Exclude self from other agents
        other_projected_positions = np.delete(self.projected_positions, self.i, axis=0)
        other_projected_velocities = np.delete(
            self.projected_velocities, self.i, axis=0
        )

        # Forward project self agent for each actions
        num_actions = actions.shape[0]
        rewards = np.zeros(num_actions)
        for i in range(num_actions):
            # Choose action
            action = actions[i]
            # Project self agent forward with that action
            self_projected_position, _, _ = forward_project(
                MDPConfig.FORWARD_PROJECTION_STEPS,
                self.positions[self.i],
                self.velocities[self.i],
                self.headings[self.i],
                action[0],
                action[1],
            )

            # Calculate reward from resulting position
            rewards[i] = self.calculate_reward(
                self_projected_position,
                other_projected_positions,
                other_projected_velocities,
            )

        # Choose best action
        best_action = actions[np.argmax(rewards)]
        return best_action

    def calculate_reward(
        self, self_position: Vector, other_positions: Vectors, other_velocities: Vectors
    ) -> float:
        best_positive_reward = positive_maximum(self_position, other_positions)
        best_negative_reward = negative_maximum(
            self_position, other_positions, other_velocities
        )

        total_reward = best_positive_reward - best_negative_reward
        return total_reward

    def hard_deck_penalty(self):
        return 0.0


def positive_maximum(self_position: Vector, other_positions: Vectors) -> float:
    magnitudes, discounts, positions = positive_parameters(other_positions)

    distances = np.sqrt(np.sum((positions - self_position) ** 2, axis=1))
    values = magnitudes * (discounts**distances)

    return np.max(values)


def positive_parameters(other_positions: Vectors) -> tuple[Scalars, Scalars, Vectors]:
    num_rewards = other_positions.shape[0]

    magnitudes = np.ones(num_rewards) * 200
    discounts = np.ones(num_rewards) * 0.99999
    positions = other_positions

    return magnitudes, discounts, positions


def negative_maximum(
    self_position: Vector, other_positions: Vectors, other_velocities: Vectors
) -> float:
    magnitudes, discounts, positions, radii_squared = negative_parameters(
        other_positions, other_velocities
    )

    distances = np.sqrt(np.sum((positions - self_position) ** 2, axis=1))
    distances_contained = distances <= radii_squared
    values = distances_contained * magnitudes * (discounts**distances)

    # if (
    #     self_position[0] < 0
    #     or self_position[0] > SimulationConfig.WIDTH
    #     or self_position[1] < 0
    #     or self_position[1] > SimulationConfig.LENGTH
    # ):
    #     values += SimulationConfig.PENALTY  # Large penalty for being out of bounds

    return np.max(values)


def negative_parameters(
    other_positions: Vectors, other_velocities: Vectors
) -> tuple[Scalars, Scalars, Vectors, Scalars]:
    timesteps = np.array([0, 1, 5, 10])
    T = len(timesteps)
    N = other_positions.shape[0]

    magnitudes = np.ones(N * T) * 300
    discounts = np.ones(N * T) * 0.99999
    positions = np.empty((N * T, 2))
    radii = np.empty(N * T)

    # fill positions with timestep projections
    idx = 0
    for i in range(N):
        for j in range(T):
            # position = p + v * t
            positions[idx, 0] = (
                other_positions[i, 0] + other_velocities[i, 0] * timesteps[j]
            )
            positions[idx, 1] = (
                other_positions[i, 1] + other_velocities[i, 1] * timesteps[j]
            )

            # radius = ||v|| * t
            radii[idx] = np.linalg.norm(other_velocities[i]) * timesteps[j]

            idx += 1

    return magnitudes, discounts, positions, radii

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from numba import njit

from configs import mdp as MDPConfig
from configs import simulation as SimulationConfig

if TYPE_CHECKING:
    from simulation.simulation import Vectors, Scalars, Vector


type Action = NDArray[np.float64]


class MDP:
    def __init__(
        self,
        i: int,
        thrust_multiplier: float,
        attack_angle_multiplier: float,
        roll_angle_multiplier: float,
        positions: Vectors,
        velocities: Scalars,
        attack_angles: Scalars,
        flight_path_angles: Scalars,
        roll_angles: Scalars,
        azimuth_angles: Scalars,
        thrusts: Scalars,
        attack_angle_rates: Scalars,
        roll_angle_rates: Scalars,
        projected_positions: Vectors,
        projected_velocities: Vectors,
        velocity_mins: Scalars,
        velocity_maxs: Scalars,
        azimuth_rate_mins: Scalars,
        azimuth_rate_maxs: Scalars,
        attack_angle_mins: Scalars,
        attack_angle_maxs: Scalars,
    ):
        self.i = i
        self.actions: NDArray[np.float64] = np.array(
            [
                [
                    thrust * thrust_multiplier,
                    attack_angle * attack_angle_multiplier,
                    roll_angle * roll_angle_multiplier,
                ]
                for thrust in MDPConfig.ACTION_THRUSTS
                for attack_angle in MDPConfig.ACTION_ATTACK_ANGLE_RATES
                for roll_angle in MDPConfig.ACTION_ROLL_ANGLE_RATES
            ]
        )

        self.positions = positions
        self.velocities = velocities
        self.attack_angles = attack_angles
        self.flight_path_angles = flight_path_angles
        self.roll_angles = roll_angles
        self.azimuth_angles = azimuth_angles

        self.thrusts = thrusts
        self.attack_angle_rates = attack_angle_rates
        self.roll_angle_rates = roll_angle_rates

        self.projected_positions = projected_positions.copy()
        self.projected_velocities = projected_velocities.copy()

        self.velocity_mins = velocity_mins
        self.velocity_maxs = velocity_maxs
        self.azimuth_rate_mins = azimuth_rate_mins
        self.azimuth_rate_maxs = azimuth_rate_maxs
        self.attack_angle_mins = attack_angle_mins
        self.attack_angle_maxs = attack_angle_maxs

    def find_action(self) -> Action:
        from simulation.simulation import forward_project

        # Exclude self from other agents
        other_projected_positions = np.delete(self.projected_positions, self.i, axis=0)
        other_projected_velocities = np.delete(
            self.projected_velocities, self.i, axis=0
        )

        # Forward project self agent for each actions
        num_actions = self.actions.shape[0]
        rewards = np.zeros(num_actions)
        for i in range(num_actions):
            # Choose action
            action = self.actions[i]

            # Project self agent forward with that action
            results = forward_project(
                MDPConfig.FORWARD_PROJECTION_STEPS,
                self.positions[self.i : self.i + 1],
                self.velocities[self.i : self.i + 1],
                self.attack_angles[self.i : self.i + 1],
                self.flight_path_angles[self.i : self.i + 1],
                self.roll_angles[self.i : self.i + 1],
                self.azimuth_angles[self.i : self.i + 1],
                action[0:1],
                action[1:2],
                action[2:3],
                self.velocity_mins[self.i : self.i + 1],
                self.velocity_maxs[self.i : self.i + 1],
                self.azimuth_rate_mins[self.i : self.i + 1],
                self.azimuth_rate_maxs[self.i : self.i + 1],
                self.attack_angle_mins[self.i : self.i + 1],
                self.attack_angle_maxs[self.i : self.i + 1],
            )
            self_projected_position = results[0][0]
            self_projected_velocity = results[1][0]

            # Calculate reward from resulting position
            rewards[i] = self.calculate_reward(
                self_projected_position,
                self_projected_velocity,
                other_projected_positions,
                other_projected_velocities,
            )

        # Choose best action
        best_action = self.actions[np.argmax(rewards)]
        return best_action

    def calculate_reward(
        self,
        self_position: Vector,
        self_velocity: Vectors,
        other_positions: Vectors,
        other_velocities: Vectors,
    ) -> float:
        best_positive_reward = positive_maximum(
            self_position, self_velocity, other_positions, other_velocities
        )
        best_negative_reward = negative_maximum(
            self_position, other_positions, other_velocities
        )

        total_reward = best_positive_reward - best_negative_reward
        total_reward -= self.hard_deck_penalty(self_position[2])
        return total_reward

    def hard_deck_penalty(self, z: float) -> float:
        hard_deck = SimulationConfig.HARD_DECK
        if z <= hard_deck:
            return SimulationConfig.PENALTY
        return 0.0


def positive_maximum(
    self_position: Vector,
    self_velocity: Vector,
    other_positions: Vectors,
    other_velocities: Vectors,
) -> float:
    r = other_positions - self_position
    d = np.linalg.norm(r, axis=1)
    r_hat = r / d[:, np.newaxis]

    # Self pointing toward enemy
    self_v_hat = self_velocity / np.linalg.norm(self_velocity)
    pursuit_alignment = np.dot(r_hat, self_v_hat)

    # We are in enemy's rear hemisphere
    enemy_v_hat = (
        other_velocities / np.linalg.norm(other_velocities, axis=1)[:, np.newaxis]
    )
    r_from_enemy = -r_hat  # vector from enemy to self
    aspect_alignment = -np.sum(r_from_enemy * enemy_v_hat, axis=1)

    score = (pursuit_alignment + aspect_alignment) / d

    return np.max(score)


def negative_maximum(
    self_position: Vector,
    other_positions: Vectors,
    other_velocities: Vectors,
) -> float:
    r = other_positions - self_position
    d = np.linalg.norm(r, axis=1)
    r_hat = r / d[:, np.newaxis]

    # Enemy pointing toward us
    enemy_v_hat = (
        other_velocities / np.linalg.norm(other_velocities, axis=1)[:, np.newaxis]
    )
    enemy_pursuit_alignment = -np.sum(
        r_hat * enemy_v_hat, axis=1
    )  # high when enemy points at self

    # Enemy is in our front hemisphere (we are in front of them)
    r_from_enemy = -r_hat  # vector from enemy to self
    aspect_alignment = -np.sum(
        r_from_enemy * enemy_v_hat, axis=1
    )  # high when we're in their front

    score = (enemy_pursuit_alignment + aspect_alignment) / d

    return np.max(score)

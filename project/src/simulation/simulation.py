from __future__ import annotations

from logging import Logger
from typing import TYPE_CHECKING, Callable
import numpy as np
import numpy.typing as npt
from collections import deque
from numba import njit, prange
from typing import TypedDict

from configs import simulation as SimulationConfig
from configs import visualisation as VisualisationConfig
from configs import mdp as MDPConfig
from simulation.mdp import MDP

if TYPE_CHECKING:
    from configs.parameters import SimulationParams

type Vector = npt.NDArray[np.float64]
type Scalar = float
type Vectors = npt.NDArray[np.float64]
type Scalars = npt.NDArray[np.float64]


@njit
def velocity_angles_scalars_to_vectors(
    velocities: Scalars, flight_path_angles: Scalars, azimuth_angles: Scalars
) -> Vectors:
    vx = velocities * np.cos(flight_path_angles) * np.cos(azimuth_angles)
    vy = velocities * np.cos(flight_path_angles) * np.sin(azimuth_angles)
    vz = velocities * np.sin(flight_path_angles)

    return np.stack((vx, vy, vz), axis=-1)


@njit
def step_agents(
    positions: Vectors,
    velocities: Scalars,
    attack_angles: Scalars,
    flight_path_angles: Scalars,
    roll_angles: Scalars,
    azimuth_angles: Scalars,
    thrusts: Scalars,
    attack_angle_rates: Scalars,
    roll_angle_rates: Scalars,
    velocity_mins: Scalars,
    velocity_maxs: Scalars,
    azimuth_rate_mins: Scalars,
    azimuth_rate_maxs: Scalars,
    attack_angle_mins: Scalars,
    attack_angle_maxs: Scalars,
) -> tuple[Vectors, Vectors, Scalars, Scalars, Scalars, Scalars, Scalars]:
    dt = 1.0 / SimulationConfig.STEPS_PER_SECOND
    g = SimulationConfig.G

    attack_angles = np.clip(
        attack_angles + attack_angle_rates * dt,
        attack_angle_mins,
        attack_angle_maxs,
    )
    roll_angles = roll_angles + roll_angle_rates * dt

    # Change 1: Calculate how many 'Gs' the plane can actually pull at this speed
    # (This stops the "infinite turn" bug at low speeds)
    v_ratio = velocities / SimulationConfig.CORNER_VELOCITY
    max_g_at_speed = np.minimum(
        SimulationConfig.MAX_GS, SimulationConfig.MAX_GS * (v_ratio**2)
    )

    # Change 2: Redefine nf as the actual G-load being pulled
    # Your 'attack_angles' now acts as a 0.0 to 1.0 multiplier for those Gs
    nf = attack_angles * max_g_at_speed

    # Change 3: Update velocity rate to include "Turn Drag" (Induced Drag)
    # This makes the agent lose speed when it turns hard
    turn_drag = SimulationConfig.K_DRAG * (nf**2)
    velocities_rates = g * (thrusts - turn_drag - np.sin(flight_path_angles))

    velocities = np.clip(
        velocities + velocities_rates * dt,
        velocity_mins,
        velocity_maxs,
    )

    flight_path_angles_rates = (g / velocities) * (
        nf * np.cos(roll_angles) - np.cos(flight_path_angles)
    )
    flight_path_angles = np.clip(
        flight_path_angles + flight_path_angles_rates * dt,
        -1.4,
        1.4,
    )

    azimuth_angles_rates = np.clip(
        (g * nf * np.sin(roll_angles))
        / (velocities * np.maximum(np.cos(flight_path_angles), 1e-3)),
        azimuth_rate_mins,
        azimuth_rate_maxs,
    )
    azimuth_angles = azimuth_angles + azimuth_angles_rates * dt

    velocities_vectors = velocity_angles_scalars_to_vectors(
        velocities, flight_path_angles, azimuth_angles
    )
    positions = positions + velocities_vectors * dt

    return (
        positions,
        velocities_vectors,
        velocities,
        attack_angles,
        flight_path_angles,
        roll_angles,
        azimuth_angles,
    )


@njit
def forward_project(
    steps: int,
    positions: Vectors,
    velocities: Scalars,
    attack_angles: Scalars,
    flight_path_angles: Scalars,
    roll_angles: Scalars,
    azimuth_angles: Scalars,
    thrusts: Scalars,
    attack_angle_rates: Scalars,
    roll_angle_rates: Scalars,
    velocity_mins: Scalars,
    velocity_maxs: Scalars,
    azimuth_rate_mins: Scalars,
    azimuth_rate_maxs: Scalars,
    attack_angle_mins: Scalars,
    attack_angle_maxs: Scalars,
) -> tuple[Vectors, Vectors, Scalars, Scalars, Scalars, Scalars, Scalars]:
    # forward project
    velocities_vectors = np.zeros_like(positions, dtype=np.float64)

    for _ in prange(steps):
        (
            positions,
            velocities_vectors,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
        ) = step_agents(
            positions,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
            thrusts,
            attack_angle_rates,
            roll_angle_rates,
            velocity_mins,
            velocity_maxs,
            azimuth_rate_mins,
            azimuth_rate_maxs,
            attack_angle_mins,
            attack_angle_maxs,
        )

    return (
        positions,
        velocities_vectors,
        velocities,
        attack_angles,
        flight_path_angles,
        roll_angles,
        azimuth_angles,
    )


class Simulation:
    def __init__(
        self,
        N: int,
        positions: list[list[float]],
        headings: list[float],
        velocity_mins: list[float],
        velocity_maxs: list[float],
        azimuth_rate_mins: list[float],
        azimuth_rate_maxs: list[float],
        attack_angle_mins: list[float],
        attack_angle_maxs: list[float],
        thrust_ratio: list[float],
        attack_angle_ratio: list[float],
        roll_angle_ratio: list[float],
    ):
        self.N = N
        self.timestep = 0

        self.thrust_ratio = thrust_ratio
        self.attack_angle_ratio = attack_angle_ratio
        self.roll_angle_ratio = roll_angle_ratio

        self.velocity_mins: Scalars = np.array(velocity_mins)
        self.velocity_maxs: Scalars = np.array(velocity_maxs)
        self.azimuth_rate_mins: Scalars = np.array(azimuth_rate_mins)
        self.azimuth_rate_maxs: Scalars = np.array(azimuth_rate_maxs)
        self.attack_angle_mins: Scalars = np.array(attack_angle_mins)
        self.attack_angle_maxs: Scalars = np.array(attack_angle_maxs)
        self.positions: Vectors = np.array(positions, dtype=np.float64)

        self.speeds = np.zeros(N, dtype=np.float64) + 0.001
        self.velocities = np.zeros((N, 3), dtype=np.float64)
        self.attack_angles: Scalars = np.zeros(N, dtype=np.float64)
        self.flight_path_angles: Scalars = np.zeros(N, dtype=np.float64)
        self.roll_angles: Scalars = np.zeros(N, dtype=np.float64)
        # self.roll_angles: Scalars = np.array([np.pi / 2, -np.pi / 2])
        self.azimuth_angles: Scalars = np.array(headings, dtype=np.float64)
        # agent inputs
        self.thrusts: Scalars = np.zeros(N, dtype=np.float64)
        self.attack_angle_rates: Scalars = np.zeros(N, dtype=np.float64)
        self.roll_angle_rates: Scalars = np.zeros(N, dtype=np.float64)
        self.chosen_actions: Vectors = np.zeros((N, 3), dtype=np.float64)

        # capturing
        self.active: npt.NDArray[np.bool_] = np.ones(N, dtype=bool)
        self.capture_buffer: Vectors = np.zeros((self.N, self.N), dtype=int)
        self.distance_check: Vectors = np.zeros((self.N, self.N), dtype=bool)
        self.nose_check: Vectors = np.zeros((self.N, self.N), dtype=bool)
        self.asymmetry_check: Vectors = np.zeros((self.N, self.N), dtype=bool)

    def step(self) -> list[tuple[int, int]]:
        new_thrusts = np.zeros(self.N, dtype=np.float64)
        new_attack_angle_rates = np.zeros(self.N, dtype=np.float64)
        new_roll_angle_rates = np.zeros(self.N, dtype=np.float64)

        (
            projected_positions,
            projected_velocities,
            _,
            _,
            _,
            _,
            _,
        ) = forward_project(
            MDPConfig.FORWARD_PROJECTION_STEPS,
            self.positions,
            self.speeds,
            self.attack_angles,
            self.flight_path_angles,
            self.roll_angles,
            self.azimuth_angles,
            new_thrusts,
            new_attack_angle_rates,
            new_roll_angle_rates,
            self.velocity_mins,
            self.velocity_maxs,
            self.azimuth_rate_mins,
            self.azimuth_rate_maxs,
            self.attack_angle_mins,
            self.attack_angle_maxs,
        )

        # determine each agent's action via MDP
        for i in range(self.N):
            if not self.active[i]:
                continue
            mdp = MDP(
                i,
                self.thrust_ratio[i],
                self.attack_angle_ratio[i],
                self.roll_angle_ratio[i],
                self.positions,
                self.speeds,
                self.attack_angles,
                self.flight_path_angles,
                self.roll_angles,
                self.azimuth_angles,
                self.thrusts,
                self.attack_angle_rates,
                self.roll_angle_rates,
                projected_positions,
                projected_velocities,
                # self.positions.copy(),
                # self.velocities.copy(),
                self.velocity_mins,
                self.velocity_maxs,
                self.azimuth_rate_mins,
                self.azimuth_rate_maxs,
                self.attack_angle_mins,
                self.attack_angle_maxs,
            )
            action = mdp.find_action()
            self.chosen_actions[i] = action
            new_thrusts[i] = action[0]
            new_attack_angle_rates[i] = action[1]
            new_roll_angle_rates[i] = action[2]

        # update all agents with their chosen action
        self.thrusts = new_thrusts
        self.attack_angle_rates = new_attack_angle_rates
        self.roll_angle_rates = new_roll_angle_rates

        # step each agent with their action and update the state
        self.positions, self.velocities, self.speeds, self.attack_angles, self.flight_path_angles, self.roll_angles, self.azimuth_angles = (  # type: ignore
            step_agents(
                self.positions,
                self.speeds,
                self.attack_angles,
                self.flight_path_angles,
                self.roll_angles,
                self.azimuth_angles,
                self.thrusts,
                self.attack_angle_rates,
                self.roll_angle_rates,
                self.velocity_mins,
                self.velocity_maxs,
                self.azimuth_rate_mins,
                self.azimuth_rate_maxs,
                self.attack_angle_mins,
                self.attack_angle_maxs,
            )
        )

        self.timestep += 1
        # return -1
        return self.capturing()

    def capturing(self) -> list[tuple[int, int]]:
        captures: list[tuple[int, int]] = []
        captured_evaders: set[int] = set()

        for pursuer in range(self.N):
            if not self.active[pursuer]:
                continue
            for evader in range(self.N):
                if (
                    pursuer == evader
                    or not self.active[evader]
                    or evader in captured_evaders
                ):
                    continue

                # Distance check
                r_pe = self.positions[evader] - self.positions[pursuer]
                d = np.linalg.norm(r_pe)
                distance_check = d < SimulationConfig.CAPTURE_RADIUS

                # Nose directions from flight path, attack and azimuth angles
                pursuer_nose = np.array(
                    [
                        np.cos(
                            self.flight_path_angles[pursuer]
                            + self.attack_angles[pursuer]
                        )
                        * np.cos(self.azimuth_angles[pursuer]),
                        np.cos(
                            self.flight_path_angles[pursuer]
                            + self.attack_angles[pursuer]
                        )
                        * np.sin(self.azimuth_angles[pursuer]),
                        np.sin(
                            self.flight_path_angles[pursuer]
                            + self.attack_angles[pursuer]
                        ),
                    ]
                )
                evader_nose = np.array(
                    [
                        np.cos(
                            self.flight_path_angles[evader] + self.attack_angles[evader]
                        )
                        * np.cos(self.azimuth_angles[evader]),
                        np.cos(
                            self.flight_path_angles[evader] + self.attack_angles[evader]
                        )
                        * np.sin(self.azimuth_angles[evader]),
                        np.sin(
                            self.flight_path_angles[evader] + self.attack_angles[evader]
                        ),
                    ]
                )

                r_pe_hat = r_pe / d
                r_ep_hat = -r_pe_hat

                # Pursuer nose pointing at evader within 50 degrees
                pursuer_alignment = np.dot(pursuer_nose, r_pe_hat)
                nose_check = pursuer_alignment >= np.cos(np.deg2rad(50))

                # Pursuer has meaningfully better alignment than evader
                evader_alignment = np.dot(evader_nose, r_ep_hat)
                asymmetry_check = pursuer_alignment > evader_alignment + 0.2

                self.distance_check[pursuer][evader] = distance_check
                self.nose_check[pursuer][evader] = nose_check
                self.asymmetry_check[pursuer][evader] = asymmetry_check

                if distance_check and nose_check and asymmetry_check:
                    self.capture_buffer[evader][pursuer] += 1
                else:
                    self.capture_buffer[evader][pursuer] = 0

                if (
                    self.capture_buffer[evader][pursuer]
                    >= SimulationConfig.CAPTURE_POINT_STEPS
                ):
                    captures.append((evader, pursuer))
                    captured_evaders.add(evader)

        for evader, _ in captures:
            self.active[evader] = False
            self.capture_buffer[evader] = 0
            self.capture_buffer[:, evader] = 0

        return captures


class SimulationManager:
    logger: Logger
    simulation: Simulation

    def __init__(self, logger: Logger):
        self.logger = logger

    def setup(self, parameters: "SimulationParams"):
        self.simulation = Simulation(SimulationConfig.AGENTS, **parameters)  # type: ignore

    def run(
        self, *callbacks: Callable[[Simulation], None]
    ) -> tuple[int, list[tuple[int, int, int]]]:
        all_captures: list[tuple[int, int, int]] = []

        while self.simulation.timestep < SimulationConfig.MAX_TIMESTEPS:
            captures = self.simulation.step()

            for captured_id, capturer_id in captures:
                all_captures.append(
                    (self.simulation.timestep, captured_id, capturer_id)
                )

            for callback in callbacks:
                callback(self.simulation)

            if np.sum(self.simulation.active) <= 1:
                break

        return (self.simulation.timestep, all_captures)

from logging import Logger
from typing import Callable
import numpy as np
import numpy.typing as npt

from configs import simulation as SimulationConfig
from configs import mdp as MDPConfig
from simulation.mdp import MDP

type Vector = npt.NDArray[np.float64]
type Scalar = float
type Vectors = npt.NDArray[np.float64]
type Scalars = npt.NDArray[np.float64]


def velocity_angles_scalars_to_vectors(
    velocities: Scalars, flight_path_angles: Scalars, azimuth_angles: Scalars
) -> Vectors:
    vx = velocities * np.cos(flight_path_angles) * np.cos(azimuth_angles)
    vy = velocities * np.cos(flight_path_angles) * np.sin(azimuth_angles)
    vz = velocities * np.sin(flight_path_angles)

    return np.stack((vx, vy, vz), axis=-1)


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
) -> tuple[Vectors, Vectors, Scalars, Scalars, Scalars, Scalars, Scalars]:
    dt = 1.0 / SimulationConfig.STEPS_PER_SECOND
    nf = thrusts * np.sin(attack_angles) + SimulationConfig.L
    g = SimulationConfig.G

    attack_angles = attack_angles + attack_angle_rates * dt
    roll_angles = roll_angles + roll_angle_rates * dt

    velocities_rates = g * (
        thrusts * np.cos(attack_angles) - np.sin(flight_path_angles)
    )
    velocities = velocities + velocities_rates * dt

    flight_path_angles_rates = (g / velocities) * (
        nf * np.cos(roll_angles) - np.cos(flight_path_angles)
    )
    flight_path_angles = flight_path_angles + flight_path_angles_rates * dt

    azimuth_angles_rates = g * (
        (nf * np.sin(roll_angles)) / (velocities * np.cos(flight_path_angles))
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
) -> tuple[Vectors, Vectors, Scalars, Scalars, Scalars, Scalars, Scalars]:
    # forward project
    velocities_vectors = np.zeros_like(positions)

    for _ in range(steps):
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
    def __init__(self, N: int):
        self.N = N
        self.timestep = 0

        # initialise agent values
        # self.positions: Vectors = np.random.rand(N, 3) * [
        #     SimulationConfig.WIDTH,
        #     SimulationConfig.LENGTH,
        #     SimulationConfig.HEIGHT,
        # ]
        self.positions: Vectors = np.array([[5000, 100, 6500], [5000, 9900, 6500]])
        self.velocities = np.zeros(N) + 0.001
        self.attack_angles: Scalars = np.zeros(N)
        self.flight_path_angles: Scalars = np.zeros(N)
        self.roll_angles: Scalars = np.zeros(N)
        self.azimuth_angles: Scalars = np.zeros(N)

        # agent inputs
        self.thrusts: Scalars = np.zeros(N)
        self.attack_angle_rates: Scalars = np.zeros(N)
        self.roll_angle_rates: Scalars = np.zeros(N)

    def step(self):
        new_thrusts = np.zeros(self.N)
        new_attack_angle_rates = np.zeros(self.N)
        new_roll_angle_rates = np.zeros(self.N)

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
            self.velocities,
            self.attack_angles,
            self.flight_path_angles,
            self.roll_angles,
            self.azimuth_angles,
            new_thrusts,
            new_attack_angle_rates,
            new_roll_angle_rates,
        )

        # determine each agent's action via MDP
        for i in range(self.N):
            mdp = MDP(
                i,
                self.positions,
                self.velocities,
                self.attack_angles,
                self.flight_path_angles,
                self.roll_angles,
                self.azimuth_angles,
                self.thrusts,
                self.attack_angle_rates,
                self.roll_angle_rates,
                projected_positions,
                projected_velocities,
            )
            action = mdp.find_action()
            new_thrusts[i] = action[0]
            new_attack_angle_rates[i] = action[1]
            new_roll_angle_rates[i] = action[2]

        # update all agents with their chosen action
        self.thrusts = new_thrusts
        self.attack_angle_rates = new_attack_angle_rates
        self.roll_angle_rates = new_roll_angle_rates

        # step each agent with their action and update the state
        self.positions, _, self.velocities, self.attack_angles, self.flight_path_angles, self.roll_angles, self.azimuth_angles = (  # type: ignore
            step_agents(
                self.positions,
                self.velocities,
                self.attack_angles,
                self.flight_path_angles,
                self.roll_angles,
                self.azimuth_angles,
                self.thrusts,
                self.attack_angle_rates,
                self.roll_angle_rates,
            )
        )

        # increase timestep
        self.timestep += 1


class SimulationManager:
    logger: Logger
    simulation: Simulation

    def __init__(self, logger: Logger):
        self.logger = logger
        self.simulation = Simulation(SimulationConfig.AGENTS)

    def run(self, *callbacks: Callable[[Simulation], None]):
        while True:
            self.logger.info(self.simulation.positions)
            self.simulation.step()
            self.logger.info(self.simulation.positions)

            for callback in callbacks:
                callback(self.simulation)

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
type VectorType = Vectors | Vector
type ScalarType = Scalars | Scalar


def step_agents(
    positions: VectorType,
    velocities: VectorType,
    accelerations: VectorType,
    headings: ScalarType,
    thrusts: ScalarType,
    rotation_rates: ScalarType,
) -> tuple[VectorType, VectorType, VectorType, ScalarType]:
    dt = 1.0 / SimulationConfig.STEPS_PER_SECOND

    headings = headings + rotation_rates * dt
    ax = thrusts * np.cos(headings)
    ay = thrusts * np.sin(headings)

    accelerations = np.stack((ax, ay), axis=-1)
    positions = positions + velocities * dt + 0.5 * accelerations * dt * dt
    velocities = velocities + accelerations * dt

    return (positions, velocities, accelerations, headings)


def forward_project(
    steps: int,
    positions: Vectors,
    velocities: Vectors,
    accelerations: Vectors,
    headings: ScalarType,
    thrusts: Scalars,
    rotation_rates: Scalars,
) -> tuple[Vectors, Vectors, Vectors, Scalars]:
    # forward project
    for _ in range(steps):
        positions, velocities, accelerations, headings = step_agents(
            positions,
            velocities,
            accelerations,
            headings,
            thrusts,
            rotation_rates,
        )

    return positions, velocities, accelerations, headings  # type: ignore


class Simulation:
    def __init__(self, N: int):
        self.N = N
        self.timestep = 0

        # initialise agent values
        self.positions: Vectors = np.random.rand(N, 2) * [
            SimulationConfig.WIDTH,
            SimulationConfig.LENGTH,
        ]
        self.velocities: Vectors = np.random.uniform(-25, 25, size=(N, 2))
        self.accelerations: Vectors = np.zeros((N, 2))
        self.headings: Scalars = np.zeros(N)

        # np.arctan2(self.velocities[:, 1], self.velocities[:, 0])

        # agent inputs
        self.thrusts: Scalars = np.zeros((N,))
        self.rotation_rates: Scalars = np.random.uniform(-1, 1, size=(N,))

    def step(self):
        new_thrusts = np.zeros(self.N)
        new_rotation_rates = np.zeros(self.N)

        projected_positions, projected_velocities, _, _ = forward_project(
            MDPConfig.FORWARD_PROJECTION_STEPS,
            self.positions,
            self.velocities,
            self.accelerations,
            self.headings,
            self.thrusts,
            self.rotation_rates,
        )

        # determine each agent's action via MDP
        for i in range(self.N):
            mdp = MDP(
                i,
                self.positions,
                self.velocities,
                self.accelerations,
                self.headings,
                self.thrusts,
                self.rotation_rates,
                projected_positions,
                projected_velocities,
            )
            action = mdp.find_action()
            new_thrusts[i] = action[0]
            new_rotation_rates[i] = action[1]

        # update all agents with their chosen action
        self.thrusts = new_thrusts
        self.rotation_rates = new_rotation_rates

        # step each agent with their action and update the state
        self.positions, self.velocities, self.accelerations, self.headings = (  # type: ignore
            step_agents(
                self.positions,
                self.velocities,
                self.accelerations,
                self.headings,
                self.thrusts,
                self.rotation_rates,
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
            self.simulation.step()

            for callback in callbacks:
                callback(self.simulation)

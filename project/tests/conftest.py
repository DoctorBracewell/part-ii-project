import numpy as np
import pytest
from simulation.simulation import Simulation
from configs import simulation as SimulationConfig


@pytest.fixture
def make_simulation():
    """Factory fixture for creating a Simulation with valid default parameters."""

    def _factory(N: int = 3) -> Simulation:
        M = SimulationConfig.MACH
        return Simulation(
            N=N,
            positions=[[5000.0, 5000.0, 6500.0]] * N,
            velocity_mins=[0.1 * M] * N,
            velocity_maxs=[0.9 * M] * N,
            azimuth_rate_mins=[-0.5] * N,
            azimuth_rate_maxs=[0.5] * N,
            attack_angle_mins=[0.09] * N,
            attack_angle_maxs=[0.35] * N,
            thrust_ratio=[10.0] * N,
            attack_angle_ratio=[1.5] * N,
            roll_angle_ratio=[1.5] * N,
        )

    return _factory

import numpy as np
from simulation.mdp import MDP
from configs import simulation as SimulationConfig


def setup_mdp(N: int) -> MDP:
    positions = np.random.rand(N, 3) * [
        SimulationConfig.WIDTH,
        SimulationConfig.LENGTH,
        SimulationConfig.HEIGHT,
    ]
    velocities = np.random.uniform(-25, 25, size=(N,))
    attack_angles = np.zeros(N)
    flight_path_angles = np.zeros(N)
    roll_angles = np.zeros(N)
    azimuth_angles = np.zeros(N)
    thrusts = np.zeros(N)
    attack_angle_rates = np.random.uniform(-1, 1, size=(N,))
    roll_angle_rates = np.random.uniform(-1, 1, size=(N,))
    projected_positions = positions.copy()
    projected_velocities = np.random.uniform(-25, 25, size=(N, 3))

    return MDP(
        i=0,
        positions=positions,
        velocities=velocities,
        attack_angles=attack_angles,
        flight_path_angles=flight_path_angles,
        roll_angles=roll_angles,
        azimuth_angles=azimuth_angles,
        thrusts=thrusts,
        attack_angle_rates=attack_angle_rates,
        roll_angle_rates=roll_angle_rates,
        projected_positions=projected_positions,
        projected_velocities=projected_velocities,
    )


def test_mdp_initialization():
    N = 5
    mdp = setup_mdp(N)
    assert mdp.i == 0
    assert mdp.positions.shape == (N, 3)
    assert mdp.velocities.shape == (N,)


def test_find_action():
    N = 5
    mdp = setup_mdp(N)
    action = mdp.find_action()
    assert isinstance(action, np.ndarray)
    assert action.shape == (3,)

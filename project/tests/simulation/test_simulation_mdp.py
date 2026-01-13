import numpy as np
from simulation.mdp import MDP
from configs import simulation as SimulationConfig


def test_mdp_initialization():
    N = 5
    positions = np.random.rand(N, 2) * [SimulationConfig.WIDTH, SimulationConfig.LENGTH]
    velocities = np.random.uniform(-25, 25, size=(N, 2))
    headings = np.zeros(N)
    thrusts = np.zeros(N)
    rotation_rates = np.random.uniform(-1, 1, size=(N,))
    projected_positions = positions.copy()
    projected_velocities = velocities.copy()

    mdp = MDP(
        i=0,
        positions=positions,
        velocities=velocities,
        headings=headings,
        thrusts=thrusts,
        rotation_rates=rotation_rates,
        projected_positions=projected_positions,
        projected_velocities=projected_velocities,
    )

    assert mdp.i == 0
    assert np.array_equal(mdp.positions, positions)
    assert np.array_equal(mdp.velocities, velocities)


def test_find_action():
    N = 5
    positions = np.random.rand(N, 2) * [SimulationConfig.WIDTH, SimulationConfig.LENGTH]
    velocities = np.random.uniform(-25, 25, size=(N, 2))
    headings = np.zeros(N)
    thrusts = np.zeros(N)
    rotation_rates = np.random.uniform(-1, 1, size=(N,))
    projected_positions = positions.copy()
    projected_velocities = velocities.copy()

    mdp = MDP(
        i=0,
        positions=positions,
        velocities=velocities,
        headings=headings,
        thrusts=thrusts,
        rotation_rates=rotation_rates,
        projected_positions=projected_positions,
        projected_velocities=projected_velocities,
    )

    action = mdp.find_action()
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)

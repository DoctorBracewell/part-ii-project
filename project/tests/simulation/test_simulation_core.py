from simulation.simulation import SimulationManager, Simulation
from unittest.mock import Mock, patch, MagicMock
import numpy as np


def test_simulation_initialization():
    simulation = Simulation(N=2)
    assert simulation.N == 2
    assert simulation.timestep == 0
    assert simulation.positions.shape == (2, 3)
    assert simulation.velocities.shape == (2,)


def test_simulation_step():
    simulation = Simulation(N=2)
    initial_positions = simulation.positions.copy()

    with patch("simulation.simulation.MDP") as mock_mdp:
        # Mock the MDP find_action to return a fixed action
        mock_mdp.return_value.find_action.return_value = (1.0, 0.1, 0.1)
        simulation.step()

    assert simulation.timestep == 1
    assert not np.array_equal(simulation.positions, initial_positions)


@patch("simulation.simulation.Simulation.step")
def test_simulation_manager_run(mock_step: MagicMock):
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)
    mock_callback = Mock()

    # To prevent an infinite loop, we can stop the simulation after a few steps.
    def side_effect(*args, **kwargs):
        if manager.simulation.timestep >= 5:
            raise StopIteration
        # Manually increment timestep to simulate progression
        manager.simulation.timestep += 1

    mock_step.side_effect = side_effect

    try:
        # The callback is passed directly to the run method
        manager.run(mock_callback)
    except StopIteration:
        pass

    # The number of calls to the callback should be 5, as it's called after each step
    assert mock_callback.call_count == 5
    mock_callback.assert_called_with(manager.simulation)

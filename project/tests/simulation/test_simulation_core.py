import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from simulation.simulation import SimulationManager, Simulation


def test_simulation_initialization(make_simulation):
    simulation = make_simulation(N=2)
    assert simulation.N == 2
    assert simulation.timestep == 0
    assert simulation.positions.shape == (2, 3)
    assert simulation.speeds.shape == (2,)


def test_simulation_step(make_simulation):
    simulation = make_simulation(N=2)
    initial_positions = simulation.positions.copy()

    with patch("simulation.simulation.MDP") as mock_mdp:
        mock_mdp.return_value.find_action.return_value = np.array([1.0, 0.1, 0.1])
        simulation.step()

    assert simulation.timestep == 1
    assert not np.array_equal(simulation.positions, initial_positions)


@patch("simulation.simulation.Simulation.step")
def test_simulation_manager_run(mock_step: MagicMock, make_simulation):
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)
    manager.simulation = make_simulation(N=2)
    mock_callback = Mock()

    def side_effect(*args, **kwargs):
        if manager.simulation.timestep >= 5:
            raise StopIteration
        manager.simulation.timestep += 1
        return []  # step() returns list of captures

    mock_step.side_effect = side_effect

    try:
        manager.run(mock_callback)
    except StopIteration:
        pass

    assert mock_callback.call_count == 5
    mock_callback.assert_called_with(manager.simulation)

from simulation.simulation import SimulationManager, Simulation
from unittest.mock import Mock, patch, MagicMock
import numpy as np


def test_simulation_initialization():
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)

    assert manager.logger == mock_logger
    assert isinstance(manager.simulation, Simulation)
    assert manager.simulation.N > 0
    assert isinstance(manager.simulation.positions, np.ndarray)
    assert isinstance(manager.simulation.velocities, np.ndarray)
    assert isinstance(manager.simulation.headings, np.ndarray)


def test_simulation_step():
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)

    initial_timestep = manager.simulation.timestep
    initial_positions = manager.simulation.positions.copy()
    initial_velocities = manager.simulation.velocities.copy()

    manager.simulation.step()

    assert manager.simulation.timestep == initial_timestep + 1
    assert not np.array_equal(manager.simulation.positions, initial_positions)
    assert not np.array_equal(manager.simulation.velocities, initial_velocities)


@patch("simulation.simulation.SimulationManager.run")
def test_simulation_run(mock_run: MagicMock):
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)
    mock_callback = Mock()

    # To prevent an infinite loop, we need to stop the simulation after a few steps.
    # We can do this by raising an exception in the mock_step function after a few calls.
    def side_effect(*args, **kwargs):
        # The callback is the first argument of the run method
        callback = args[0]
        for i in range(6):
            callback(manager.simulation)
        raise StopIteration

    mock_run.side_effect = side_effect

    try:
        manager.run(mock_callback)
    except StopIteration:
        pass

    assert mock_run.call_count == 1
    assert mock_callback.call_count == 6
    mock_callback.assert_called_with(manager.simulation)

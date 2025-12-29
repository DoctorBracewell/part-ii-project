from simulation.simulation import SimulationManager
from simulation.mdp import Simulation
from simulation.agent import Agent
from unittest.mock import Mock, patch, MagicMock


def test_simulation_initialization():
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)

    assert manager.logger == mock_logger
    assert isinstance(manager.simulation, Simulation)
    assert isinstance(manager.simulation.pursuer, Agent)
    assert isinstance(manager.simulation.evader, Agent)


def test_simulation_step():
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)

    initial_timestep = manager.simulation.timestep
    initial_pursuer_state = manager.simulation.pursuer.get_state()
    initial_evader_state = manager.simulation.evader.get_state()

    manager.simulation.step()

    assert manager.simulation.timestep == initial_timestep + 1
    assert manager.simulation.pursuer.get_state() != initial_pursuer_state
    assert manager.simulation.evader.get_state() != initial_evader_state


@patch("simulation.simulation.SimulationManager.run")
def test_simulation_run(mock_run: MagicMock):
    mock_logger = Mock()
    manager = SimulationManager(logger=mock_logger)
    mock_callback = Mock()

    # To prevent an infinite loop, we need to stop the simulation after a few steps.
    # We can do this by raising an exception in the mock_step function after a few calls.
    def side_effect(*args, **kwargs):
        # The callback is the first argument of the run method
        callback = args[0][0]
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

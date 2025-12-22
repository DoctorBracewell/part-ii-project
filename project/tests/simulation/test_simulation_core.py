from simulation.simulation import Simulation, SimulationStatus
from simulation.agent import Agent
from unittest.mock import Mock, patch, MagicMock


def test_simulation_initialization():
    mock_logger = Mock()
    simulation = Simulation(logger=mock_logger)

    assert simulation.logger == mock_logger
    assert isinstance(simulation.status, SimulationStatus)
    assert isinstance(simulation.status.pursuer, Agent)
    assert isinstance(simulation.status.evader, Agent)
    assert simulation.mdp is not None


def test_simulation_step():
    mock_logger = Mock()
    simulation = Simulation(logger=mock_logger)

    initial_timestep = simulation.status.timestep
    initial_pursuer_state = simulation.status.pursuer.get_state()
    initial_evader_state = simulation.status.evader.get_state()

    simulation.step()

    assert simulation.status.timestep == initial_timestep + 1
    assert simulation.status.pursuer.get_state() != initial_pursuer_state
    assert simulation.status.evader.get_state() != initial_evader_state


@patch("simulation.simulation.Simulation.step")
def test_simulation_run(mock_step: MagicMock):
    mock_logger = Mock()
    simulation = Simulation(logger=mock_logger)
    mock_callback = Mock()

    # To prevent an infinite loop, we need to stop the simulation after a few steps.
    # We can do this by raising an exception in the mock_step function after a few calls.
    def side_effect():
        if mock_step.call_count > 5:
            raise StopIteration

    mock_step.side_effect = side_effect

    try:
        simulation.run(mock_callback)
    except StopIteration:
        pass

    assert mock_step.call_count > 1
    assert mock_callback.call_count > 1
    mock_callback.assert_called_with(simulation.status)

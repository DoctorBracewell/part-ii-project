import pytest
from simulation.agents import Agent
from simulation.simulation import Simulation, SimulationStatus
from config import SimulationConfig
from unittest.mock import Mock


def test_simulation_initialization():
    mock_logger = Mock()
    simulation = Simulation(logger=mock_logger)

    assert simulation.logger == mock_logger
    assert isinstance(simulation.pursuer, Agent)
    assert isinstance(simulation.evader, Agent)
    assert isinstance(simulation.status, SimulationStatus)

    assert simulation.pursuer.position == 0.0
    assert simulation.evader.position == 5.0
    assert simulation.status.pursuer == simulation.pursuer
    assert simulation.status.evader == simulation.evader


def test_simulation_run_timestep_increment():
    mock_logger = Mock()
    simulation = Simulation(logger=mock_logger)

    initial_timestep = simulation.status.timestep
    simulation.run()

    expected_timesteps = 10 * SimulationConfig.STEPS_PER_SECOND
    assert simulation.status.timestep == initial_timestep + expected_timesteps


def test_simulation_run_agent_movement():
    mock_logger = Mock()
    simulation = Simulation(logger=mock_logger)

    initial_pursuer_pos = simulation.pursuer.position
    initial_evader_pos = simulation.evader.position

    simulation.run()

    # Pursuer: initial_velocity = -1, acceleration = 0.1
    # Evader: initial_velocity = 0, acceleration = 0
    # Total time = 10 seconds (10 * STEPS_PER_SECOND * (1/STEPS_PER_SECOND))
    total_time = 10.0

    # Expected pursuer movement:
    # position = initial_pos + (initial_vel * time) + (0.5 * acc * time^2)
    expected_pursuer_pos = (
        initial_pursuer_pos + (-1 * total_time) + (0.5 * 0.1 * total_time**2)
    )
    # velocity = initial_vel + (acc * time)
    expected_pursuer_vel = -1 + (0.1 * total_time)

    # Expected evader movement:
    # position = initial_pos + (initial_vel * time) + (0.5 * acc * time^2)
    expected_evader_pos = (
        initial_evader_pos + (0 * total_time) + (0.5 * 0 * total_time**2)
    )
    # velocity = initial_vel + (acc * time)
    expected_evader_vel = 0 + (0 * total_time)

    assert simulation.pursuer.position == pytest.approx(expected_pursuer_pos)
    assert simulation.pursuer.velocity == pytest.approx(expected_pursuer_vel)
    assert simulation.evader.position == pytest.approx(expected_evader_pos)
    assert simulation.evader.velocity == pytest.approx(expected_evader_vel)


def test_simulation_run_callback_execution():
    mock_logger = Mock()
    simulation = Simulation(logger=mock_logger)
    mock_callback = Mock()

    simulation.run(mock_callback)

    expected_calls = 10 * SimulationConfig.STEPS_PER_SECOND
    assert mock_callback.call_count == expected_calls
    # Verify that the callback was called with a SimulationStatus object
    mock_callback.assert_called_with(simulation.status)

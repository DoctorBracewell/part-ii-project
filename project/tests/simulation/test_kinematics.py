import numpy as np
import pytest
from simulation.simulation import step_agents
from configs import simulation as SimulationConfig


def make_bounds(N: int):
    """Return permissive no-op bounds for N agents so physics tests are not distorted."""
    M = float(SimulationConfig.MACH)
    return dict(
        velocity_mins=np.full(N, 1e-6, dtype=np.float64),
        velocity_maxs=np.full(N, 10.0 * M, dtype=np.float64),
        azimuth_rate_mins=np.full(N, -1e6, dtype=np.float64),
        azimuth_rate_maxs=np.full(N, 1e6, dtype=np.float64),
        attack_angle_mins=np.full(N, -np.pi / 2 + 1e-3, dtype=np.float64),
        attack_angle_maxs=np.full(N, np.pi / 2 - 1e-3, dtype=np.float64),
    )


def test_no_gravity_straight_flight():
    """With no gravity and no thrust, agents fly in straight lines at constant velocity."""
    SimulationConfig.G = 0.0
    SimulationConfig.L = 0.0
    step_agents.recompile()

    N = 2
    bounds = make_bounds(N)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocities = np.array([100.0, 200.0])
    attack_angles = np.zeros(N)
    flight_path_angles = np.zeros(N)
    roll_angles = np.zeros(N)
    azimuth_angles = np.array([0.0, np.pi / 2])  # agent 0 along x, agent 1 along y
    thrusts = np.zeros(N)
    attack_angle_rates = np.zeros(N)
    roll_angle_rates = np.zeros(N)

    for _ in range(SimulationConfig.STEPS_PER_SECOND):
        (
            positions,
            _,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
        ) = step_agents(
            positions,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
            thrusts,
            attack_angle_rates,
            roll_angle_rates,
            **bounds,
        )

    assert np.allclose(positions[0], [100.0, 0.0, 0.0])
    assert np.allclose(velocities[0], 100.0)
    assert np.allclose(positions[1], [0.0, 200.0, 0.0])
    assert np.allclose(velocities[1], 200.0)


def test_level_flight_with_thrust():
    """With thrust balancing gravity, the agent maintains altitude."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 1.0
    step_agents.recompile()

    N = 1
    bounds = make_bounds(N)
    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([0.0])
    attack_angles = np.zeros(N)
    flight_path_angles = np.zeros(N)
    roll_angles = np.zeros(N)
    azimuth_angles = np.zeros(N)
    thrusts = np.zeros(N)
    attack_angle_rates = np.zeros(N)
    roll_angle_rates = np.zeros(N)

    for _ in range(SimulationConfig.STEPS_PER_SECOND):
        (
            positions,
            _,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
        ) = step_agents(
            positions,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
            thrusts,
            attack_angle_rates,
            roll_angle_rates,
            **bounds,
        )

    assert np.allclose(positions[0, 2], 1000.0, atol=1e-1)


def test_level_turn():
    """A coordinated turn changes azimuth while preserving altitude and speed."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 0.0

    v = 100.0
    roll_angle = np.deg2rad(30)
    nf = 1 / np.cos(roll_angle)
    SimulationConfig.L = nf
    step_agents.recompile()

    N = 1
    bounds = make_bounds(N)
    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([v])
    attack_angles = np.zeros(N)
    flight_path_angles = np.zeros(N)
    roll_angles = np.full(N, roll_angle)
    azimuth_angles = np.zeros(N)
    thrusts = np.zeros(N)
    attack_angle_rates = np.zeros(N)
    roll_angle_rates = np.zeros(N)

    for _ in range(SimulationConfig.STEPS_PER_SECOND):
        (
            positions,
            _,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
        ) = step_agents(
            positions,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
            thrusts,
            attack_angle_rates,
            roll_angle_rates,
            **bounds,
        )

    assert np.allclose(positions[0, 2], 1000.0, atol=1e-1)
    assert np.allclose(velocities[0], v)

    expected_azimuth_rate = SimulationConfig.G * np.tan(roll_angle) / v
    expected_azimuth_change = expected_azimuth_rate * 1.0  # 1 second
    assert np.allclose(azimuth_angles[0], expected_azimuth_change, atol=1e-2)


def test_climb():
    """A steady climb increases altitude while maintaining speed and flight path angle."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 0.0

    v = 100.0
    flight_path_angle = np.deg2rad(10)

    thrust = np.sin(flight_path_angle)
    SimulationConfig.L = np.cos(flight_path_angle)
    step_agents.recompile()

    N = 1
    bounds = make_bounds(N)
    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([v])
    attack_angles = np.zeros(N)
    flight_path_angles = np.full(N, flight_path_angle)
    roll_angles = np.zeros(N)
    azimuth_angles = np.zeros(N)
    thrusts = np.full(N, thrust)
    attack_angle_rates = np.zeros(N)
    roll_angle_rates = np.zeros(N)

    for _ in range(SimulationConfig.STEPS_PER_SECOND):
        (
            positions,
            _,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
        ) = step_agents(
            positions,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
            thrusts,
            attack_angle_rates,
            roll_angle_rates,
            **bounds,
        )

    assert np.allclose(velocities[0], v)
    assert np.allclose(flight_path_angles[0], flight_path_angle)

    expected_altitude_change = v * np.sin(flight_path_angle) * 1.0
    assert np.allclose(positions[0, 2] - 1000.0, expected_altitude_change, atol=1)


def test_dive():
    """A steady dive decreases altitude while maintaining speed and flight path angle."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 0.0

    v = 100.0
    flight_path_angle = np.deg2rad(-10)

    thrust = np.sin(flight_path_angle)
    SimulationConfig.L = np.cos(flight_path_angle)
    step_agents.recompile()

    N = 1
    bounds = make_bounds(N)
    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([v])
    attack_angles = np.zeros(N)
    flight_path_angles = np.full(N, flight_path_angle)
    roll_angles = np.zeros(N)
    azimuth_angles = np.zeros(N)
    thrusts = np.full(N, thrust)
    attack_angle_rates = np.zeros(N)
    roll_angle_rates = np.zeros(N)

    for _ in range(SimulationConfig.STEPS_PER_SECOND):
        (
            positions,
            _,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
        ) = step_agents(
            positions,
            velocities,
            attack_angles,
            flight_path_angles,
            roll_angles,
            azimuth_angles,
            thrusts,
            attack_angle_rates,
            roll_angle_rates,
            **bounds,
        )

    assert np.allclose(velocities[0], v)
    assert np.allclose(flight_path_angles[0], flight_path_angle)

    expected_altitude_change = v * np.sin(flight_path_angle) * 1.0
    assert np.allclose(positions[0, 2] - 1000.0, expected_altitude_change, atol=1)

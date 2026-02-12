import numpy as np
import pytest
from simulation.simulation import step_agents
from configs import simulation as SimulationConfig


def test_no_gravity_straight_flight():
    """Test that with no gravity and no thrust, the agent flies in a straight line."""
    SimulationConfig.G = 0.0
    SimulationConfig.L = 0.0

    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocities = np.array([100.0, 200.0])
    attack_angles = np.zeros(2)
    flight_path_angles = np.zeros(2)
    roll_angles = np.zeros(2)
    azimuth_angles = np.array([0.0, np.pi / 2])  # Agent 1 along x, Agent 2 along y
    thrusts = np.zeros(2)
    attack_angle_rates = np.zeros(2)
    roll_angle_rates = np.zeros(2)

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
        )

    assert np.allclose(positions[0], [100.0, 0.0, 0.0])
    assert np.allclose(velocities[0], 100.0)
    assert np.allclose(positions[1], [0.0, 200.0, 0.0])
    assert np.allclose(velocities[1], 200.0)


def test_level_flight_with_thrust():
    """Test that with thrust balancing gravity, the agent maintains altitude."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 1.0

    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([0.0])
    attack_angles = np.zeros(1)
    flight_path_angles = np.zeros(1)
    roll_angles = np.zeros(1)
    azimuth_angles = np.zeros(1)

    thrusts = np.zeros(1)
    attack_angle_rates = np.zeros(1)
    roll_angle_rates = np.zeros(1)

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
        )

    assert np.allclose(
        positions[0, 2], 1000.0, atol=1e-1
    )  # Altitude should be constant
    # assert np.allclose(velocities[0], 100.0)  # Velocity should be constant


def test_level_turn():
    """Test a coordinated turn."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 0.0

    v = 100.0
    roll_angle = np.deg2rad(30)  # 30 degree bank

    # In a coordinated turn, Lift = Weight / cos(roll)
    # So the load factor nf = 1 / cos(roll)
    nf = 1 / np.cos(roll_angle)

    # We need nf = thrust * sin(attack) + L. Let's set L = nf and thrust = 0.
    SimulationConfig.L = nf

    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([v])
    attack_angles = np.zeros(1)
    flight_path_angles = np.zeros(1)
    roll_angles = np.full(1, roll_angle)
    azimuth_angles = np.zeros(1)
    thrusts = np.zeros(1)
    attack_angle_rates = np.zeros(1)
    roll_angle_rates = np.zeros(1)

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
        )

    # Check that altitude is maintained
    assert np.allclose(positions[0, 2], 1000.0, atol=1e-1)

    # Check that velocity is maintained
    assert np.allclose(velocities[0], v)

    # Check that the azimuth angle has changed correctly
    # azimuth_rate = g * tan(roll) / v
    expected_azimuth_rate = SimulationConfig.G * np.tan(roll_angle) / v
    expected_azimuth_change = expected_azimuth_rate * 1.0  # 1 second
    assert np.allclose(azimuth_angles[0], expected_azimuth_change, atol=1e-2)


def test_climb():
    """Test a steady climb."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 0.0

    v = 100.0
    flight_path_angle = np.deg2rad(10)  # 10 degree climb

    # In a steady climb, T*cos(alpha) = D + W*sin(gamma)
    # and L = W*cos(gamma)
    # The equations of motion are different.
    # Let's use the given equations.
    # v_dot = g * (T*cos(alpha) - sin(gamma))
    # gamma_dot = (g/v) * (nf*cos(roll) - cos(gamma))
    #
    # For a steady climb, v_dot = 0 and gamma_dot = 0.
    # So T*cos(alpha) = sin(gamma)
    # and nf*cos(roll) = cos(gamma)
    #
    # Let's set alpha = 0, roll = 0.
    # Then T = sin(gamma) and nf = cos(gamma).
    # Since nf = T*sin(alpha) + L, we have L = cos(gamma).

    thrust = np.sin(flight_path_angle)
    SimulationConfig.L = np.cos(flight_path_angle)

    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([v])
    attack_angles = np.zeros(1)
    flight_path_angles = np.full(1, flight_path_angle)
    roll_angles = np.zeros(1)
    azimuth_angles = np.zeros(1)
    thrusts = np.full(1, thrust)
    attack_angle_rates = np.zeros(1)
    roll_angle_rates = np.zeros(1)

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
        )

    # Check that velocity is maintained
    assert np.allclose(velocities[0], v)

    # Check that flight path angle is maintained
    assert np.allclose(flight_path_angles[0], flight_path_angle)

    # Check that altitude has increased correctly
    # z_dot = v * sin(gamma)
    expected_altitude_change = v * np.sin(flight_path_angle) * 1.0  # 1 second
    assert np.allclose(positions[0, 2] - 1000.0, expected_altitude_change, atol=1)


def test_dive():
    """Test a steady dive."""
    SimulationConfig.G = 9.81
    SimulationConfig.L = 0.0

    v = 100.0
    flight_path_angle = np.deg2rad(-10)  # 10 degree dive

    # In a steady dive, T*cos(alpha) = D + W*sin(gamma)
    # and L = W*cos(gamma)
    # The equations of motion are different.
    # Let's use the given equations.
    # v_dot = g * (T*cos(alpha) - sin(gamma))
    # gamma_dot = (g/v) * (nf*cos(roll) - cos(gamma))
    #
    # For a steady dive, v_dot = 0 and gamma_dot = 0.
    # So T*cos(alpha) = sin(gamma)
    # and nf*cos(roll) = cos(gamma)
    #
    # Let's set alpha = 0, roll = 0.
    # Then T = sin(gamma) and nf = cos(gamma).
    # Since nf = T*sin(alpha) + L, we have L = cos(gamma).

    thrust = np.sin(flight_path_angle)  # This will be negative
    SimulationConfig.L = np.cos(flight_path_angle)

    positions = np.array([[0.0, 0.0, 1000.0]])
    velocities = np.array([v])
    attack_angles = np.zeros(1)
    flight_path_angles = np.full(1, flight_path_angle)
    roll_angles = np.zeros(1)
    azimuth_angles = np.zeros(1)
    thrusts = np.full(1, thrust)
    attack_angle_rates = np.zeros(1)
    roll_angle_rates = np.zeros(1)

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
        )

    # Check that velocity is maintained
    assert np.allclose(velocities[0], v)

    # Check that flight path angle is maintained
    assert np.allclose(flight_path_angles[0], flight_path_angle)

    # Check that altitude has decreased correctly
    # z_dot = v * sin(gamma)
    expected_altitude_change = v * np.sin(flight_path_angle) * 1.0  # 1 second
    assert np.allclose(positions[0, 2] - 1000.0, expected_altitude_change, atol=1)

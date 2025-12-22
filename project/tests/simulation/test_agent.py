import pytest
import random
from simulation.agent import Agent, actions
from config import SimulationConfig


def test_agent_initialization():
    agent = Agent(position=10.0, velocity=1.0, acceleration=2.0)
    assert agent.position == 10.0
    assert agent.velocity == 1.0
    assert agent.acceleration == 2.0


def test_agent_update_from_action():
    agent = Agent()
    initial_acceleration = agent.acceleration
    agent.update_from_action(actions[1])  # Assuming actions[1] has a thrust value
    assert (
        agent.acceleration == actions[1][0]
    )  # The acceleration should be updated to the thrust value in the action tuple
    assert agent.acceleration != initial_acceleration


def test_agent_step():
    agent = Agent(position=0.0, velocity=1.0, acceleration=2.0)
    time_step = 1 / SimulationConfig.STEPS_PER_SECOND

    initial_position = agent.position
    initial_velocity = agent.velocity

    agent.step()

    expected_position = (
        initial_position
        + initial_velocity * time_step
        + 0.5 * agent.acceleration * time_step**2
    )
    expected_velocity = initial_velocity + agent.acceleration * time_step

    assert agent.position == pytest.approx(expected_position)
    assert agent.velocity == pytest.approx(expected_velocity)


def test_agent_randomise():
    agent = Agent()
    initial_position = agent.position
    initial_velocity = agent.velocity

    random.seed(42)  # Seed for reproducibility
    agent.randomise()

    assert agent.position != initial_position
    assert agent.velocity != initial_velocity
    assert agent.position >= 0
    assert agent.position <= SimulationConfig.WIDTH
    assert (
        agent.velocity >= -25
    )  # Assuming these are the bounds from agent.py randomise method
    assert agent.velocity <= 25
    assert agent.acceleration in [action[0] for action in actions]


def test_agent_get_set_state():
    agent = Agent(position=10.0, velocity=5.0, acceleration=2.0)
    state = agent.get_state()
    assert state == (10.0, 5.0, 2.0)

    new_state = (20.0, 10.0, 3.0)
    agent.set_state(new_state)
    assert agent.position == 20.0
    assert agent.velocity == 10.0
    assert agent.acceleration == 3.0

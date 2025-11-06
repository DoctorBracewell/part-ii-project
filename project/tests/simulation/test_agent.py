from simulation.agents import Agent


def test_agent_initialization():
    agent = Agent(position=10.0)
    assert agent.position == 10.0
    assert agent.velocity == 0.0
    assert agent.acceleration == 0.0


def test_agent_move_no_velocity_no_acceleration():
    agent = Agent(position=0.0)
    agent.move(time=1.0)
    assert agent.position == 0.0
    assert agent.velocity == 0.0


def test_agent_move_with_velocity_no_acceleration():
    agent = Agent(position=0.0)
    agent.velocity = 5.0
    agent.move(time=2.0)
    assert agent.position == 10.0  # 0 + 5 * 2
    assert agent.velocity == 5.0


def test_agent_move_with_acceleration_no_initial_velocity():
    agent = Agent(position=0.0)
    agent.acceleration = 2.0
    agent.move(time=3.0)
    assert agent.position == 0.5 * 2.0 * (3.0**2)  # 0 + 0 * 3 + 0.5 * 2 * 3^2 = 9.0
    assert agent.velocity == 2.0 * 3.0  # 0 + 2 * 3 = 6.0


def test_agent_move_with_velocity_and_acceleration():
    agent = Agent(position=10.0)
    agent.velocity = 2.0
    agent.acceleration = 1.0
    agent.move(time=4.0)
    # position = 10 + (2 * 4) + (0.5 * 1 * 4^2) = 10 + 8 + 8 = 26.0
    # velocity = 2 + (1 * 4) = 6.0
    assert agent.position == 26.0
    assert agent.velocity == 6.0

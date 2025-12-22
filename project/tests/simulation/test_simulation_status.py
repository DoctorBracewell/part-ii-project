from simulation.simulation import SimulationStatus
from config import SimulationConfig


def test_simulation_status_initialization():
    status = SimulationStatus()

    assert status.pursuer is not None
    assert status.evader is not None
    assert status.timestep == 0
    assert status.agent_count == 2
    assert status.hard_deck == SimulationConfig.HARD_DECK


def test_simulation_status_get_set_state():
    status = SimulationStatus()
    # Set some initial state for the agents within status for predictable testing
    status.pursuer.position = 10.0
    status.pursuer.velocity = 1.0
    status.pursuer.acceleration = 0.5
    status.evader.position = 20.0
    status.evader.velocity = 2.0
    status.evader.acceleration = 0.0
    status.timestep = 5

    initial_state = status.get_state()
    assert initial_state[0] == 5  # timestep
    assert initial_state[1] == 2  # agent_count
    assert initial_state[2] == SimulationConfig.HARD_DECK  # hard_deck
    assert initial_state[3] == (10.0, 1.0, 0.5)  # pursuer state
    assert initial_state[4] == (20.0, 2.0, 0.0)  # evader state

    new_pursuer_state = (15.0, 1.5, 0.6)
    new_evader_state = (25.0, 2.5, 0.1)
    new_state = (10, 2, SimulationConfig.HARD_DECK, new_pursuer_state, new_evader_state)
    status.set_state(new_state)

    assert status.timestep == 10
    assert status.pursuer.position == 15.0
    assert status.pursuer.velocity == 1.5
    assert status.pursuer.acceleration == 0.6
    assert status.evader.position == 25.0
    assert status.evader.velocity == 2.5
    assert status.evader.acceleration == 0.1

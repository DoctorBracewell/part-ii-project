from simulation.mdp import Simulation
from simulation.agent import Agent
from configs import simulation as SimulationConfig


def test_simulation_initialization():
    pursuer = Agent(0)
    evader = Agent(1)
    simulation = Simulation(pursuer, evader)

    assert simulation.pursuer is not None
    assert simulation.evader is not None
    assert simulation.timestep == 0
    assert simulation.agent_count == 2
    assert simulation.hard_deck == SimulationConfig.HARD_DECK


def test_simulation_get_set_state():
    pursuer = Agent(0)
    evader = Agent(1)
    simulation = Simulation(pursuer, evader)

    # Set some initial state for the agents within status for predictable testing
    simulation.pursuer.position = 10.0
    simulation.pursuer.velocity = 1.0
    simulation.pursuer.acceleration = 0.5
    simulation.evader.position = 20.0
    simulation.evader.velocity = 2.0
    simulation.evader.acceleration = 0.0
    simulation.timestep = 5

    initial_state = simulation.get_state()
    assert initial_state[0] == 5  # timestep
    assert initial_state[1] == 2  # agent_count
    assert initial_state[2] == SimulationConfig.HARD_DECK  # hard_deck
    assert initial_state[3] == (0, 10.0, 1.0, 0.5)  # pursuer state
    assert initial_state[4] == (1, 20.0, 2.0, 0.0)  # evader state

    new_pursuer_state = (0, 15.0, 1.5, 0.6)
    new_evader_state = (1, 25.0, 2.5, 0.1)
    new_state = (10, 2, SimulationConfig.HARD_DECK, new_pursuer_state, new_evader_state)
    simulation.set_state(new_state)

    assert simulation.timestep == 10
    assert simulation.pursuer.position == 15.0
    assert simulation.pursuer.velocity == 1.5
    assert simulation.pursuer.acceleration == 0.6
    assert simulation.evader.position == 25.0
    assert simulation.evader.velocity == 2.5
    assert simulation.evader.acceleration == 0.1

from simulation.agent import Agent
from simulation.simulation import SimulationStatus
from config import SimulationConfig


def test_simulation_status_initialization():
    pursuer_agent = Agent(position=0.0)
    evader_agent = Agent(position=5.0)
    status = SimulationStatus(pursuer=pursuer_agent, evader=evader_agent)

    assert status.pursuer == pursuer_agent
    assert status.evader == evader_agent
    assert status.timestep == 0
    assert status.agents == 2
    assert status.hard_deck == SimulationConfig.HARD_DECK

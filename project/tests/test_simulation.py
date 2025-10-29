from simulation import SimulationStatus


def test_simulation_status_initial_values():
    status = SimulationStatus()
    assert status.timestep == 0
    assert status.agents == 0
    assert status.hard_deck == 0

from configs import simulation as SimulationConfig
from numba.experimental import jitclass
import numpy as np
from numpy.typing import NDArray

type AgentState = NDArray[np.float64]
"""(id: float64, position: float, velocity: float, acceleration: float)"""

type Action = NDArray[np.float64]
"""[thrust: float]"""

thrusts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
actions: NDArray[np.float64] = np.array([[thrust] for thrust in thrusts])


@jitclass
class Agent:
    id: int
    position: float
    velocity: float
    acceleration: float

    def __init__(
        self,
        id: int,
        position: float = 0,
        velocity: float = 0,
        acceleration: float = 0,
    ):
        self.id = id
        self.position = position  # metres
        self.velocity = velocity  # metres/second
        self.acceleration = acceleration  # metres/second^2

    def randomise(self):
        self.position = np.random.uniform(0, SimulationConfig.WIDTH)
        self.velocity = np.random.uniform(-25, 25)

        n_actions: int = actions.shape[0]
        idx = np.random.randint(0, n_actions)
        action = actions[idx, :]

        self.update_from_action(action)

    def update_from_action(self, action: Action):
        self.acceleration = action[0]

    def step(self):
        time = 1 / SimulationConfig.STEPS_PER_SECOND

        # SUVAT!
        self.position += self.velocity * time + 0.5 * self.acceleration * time**2
        self.velocity += self.acceleration * time

    def get_state(self) -> AgentState:
        return np.array(
            [float(self.id), self.position, self.velocity, self.acceleration]
        )

    def set_state(self, state: AgentState):
        _, self.position, self.velocity, self.acceleration = state

    # def __repr__(self):
    #     return f"Agent(id={self.id}, pos={self.position:.2f}, vel={self.velocity:.2f}, acc={self.acceleration:.2f})"

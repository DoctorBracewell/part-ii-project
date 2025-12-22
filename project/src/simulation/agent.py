from config import SimulationConfig
from numpy.typing import NDArray
import numpy as np

type State = NDArray[(np.float64, np.float64, np.float64)]
"""[pursuer_pos, evader_pos, evader_vel]"""

type Action = tuple[float]
"""(thrust: float)"""

thrusts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
actions: list[Action] = [(thrust,) for thrust in thrusts]


class Agent:
    position: float
    velocity: float
    acceleration: float

    def __init__(
        self, position: float = 0, velocity: float = 0, acceleration: float = 0
    ):
        self.position = position  # metres
        self.velocity = velocity  # metres/second
        self.acceleration = acceleration  # metres/second^2

    def randomise(self):
        import random

        self.position = random.uniform(0, SimulationConfig.WIDTH)
        self.velocity = random.uniform(-25, 25)

        action = random.choice(actions)
        self.update_from_action(action)

    def update_from_action(self, action: Action):
        self.acceleration = action[0]

    def step(self):
        time = 1 / SimulationConfig.STEPS_PER_SECOND

        # SUVAT!
        self.position += self.velocity * time + 0.5 * self.acceleration * time**2
        self.velocity += self.acceleration * time

    def get_state(self) -> tuple[float, float, float]:
        """
        :return: (position, velocity, acceleration)
        """
        return (self.position, self.velocity, self.acceleration)

    def set_state(self, state: tuple[float, float, float]):
        """
        :param state: (position, velocity, acceleration)
        """
        self.position, self.velocity, self.acceleration = state

from configs import simulation as SimulationConfig
import numpy as np

type Position = float

type AgentState = tuple[int, Position, float, float]
"""(id: int, position: Position, velocity: float, acceleration: float)"""

type Action = tuple[float]
"""[thrust: float]"""

thrusts: list[float] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
actions: list[Action] = [(thrust,) for thrust in thrusts]


class Agent:
    id: int
    next_action: Action | None

    position: Position
    velocity: float
    acceleration: float

    def __init__(
        self,
        id: int,
        position: Position = 0,
        velocity: float = 0,
        acceleration: float = 0,
    ):
        self.id = id
        self.position = position  # metres
        self.velocity = velocity  # metres/second
        self.acceleration = acceleration  # metres/second^2
        self.next_action = None

    def randomise(self):
        import random

        self.position = np.random.uniform(0, SimulationConfig.WIDTH)
        self.velocity = np.random.uniform(-25, 25)

        # random choice using numpy for compatibility with numba
        # n_actions: int = actions.shape[0]
        # idx = np.random.randint(0, n_actions)
        # action = actions[idx, :]

        action = random.choice(actions)
        self.update_from_action(action)

    def update_from_next_action(self):
        if self.next_action is not None:
            self.update_from_action(self.next_action)
            self.next_action = None

    def update_from_action(self, action: Action):
        self.acceleration = action[0]

    def step(self):
        time = 1 / SimulationConfig.STEPS_PER_SECOND

        # SUVAT!
        self.position += self.velocity * time + 0.5 * self.acceleration * time**2
        self.velocity += self.acceleration * time

    def get_state(self) -> AgentState:
        return (self.id, self.position, self.velocity, self.acceleration)

    def set_state(self, state: AgentState):
        _, self.position, self.velocity, self.acceleration = state

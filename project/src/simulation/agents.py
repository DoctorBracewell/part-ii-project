class Agent:
    position: float
    velocity: float
    acceleration: float

    def __init__(self, position: float):
        self.position = position  # metres
        self.velocity = 0.0  # metres/second
        self.acceleration = 0.0  # metres/second^2

    def move(self, time: float):
        self.position += self.velocity * time + 0.5 * self.acceleration * time**2
        self.velocity += self.acceleration * time

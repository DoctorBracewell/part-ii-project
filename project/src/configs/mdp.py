import numpy as np
from configs import simulation as SimulationConfig

# FORWARD_PROJECTION_STEPS = int(SimulationConfig.STEPS_PER_SECOND * 0.25)
FORWARD_PROJECTION_STEPS = 3

ACTION_THRUSTS = np.linspace(0.0, 0.9, 10)
ACTION_ATTACK_ANGLE_RATES = np.array(
    [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
)
ACTION_ROLL_ANGLE_RATES = np.array(
    [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
)

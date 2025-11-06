class SimulationConfig:
    STEPS_PER_SECOND = 30
    HARD_DECK = 0


class OutputConfig:
    OUTPUT_DIRECTORY = "results/"
    Y_OFFSET_STEP = 0.1
    Y_LIM = (-0.5, 0.5)
    DPI = 300
    FIGSIZE = (8, 4)
    X_PADDING_RATIO = 0.2


class PlotConfig(OutputConfig):
    OUTPUT_FILENAME = "agent_movement.png"


class VideoConfig(OutputConfig):
    OUTPUT_FILENAME = "agent_movement.mp4"

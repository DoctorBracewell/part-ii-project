class SimulationConfig:
    MAX_STEPS = 1_000
    STEPS_PER_SECOND = 10
    HARD_DECK = 0
    WIDTH = 25_000
    HEIGHT = 25_000
    LENGTH = 25_000


class OutputConfig:
    OUTPUT_DIRECTORY = "results/intermediate"
    Y_OFFSET_STEP = 0.1
    Y_LIM = (-0.5, 0.5)
    DPI = 300
    FIGSIZE = (8, 4)
    X_PADDING_RATIO = 0.2


class PlotConfig(OutputConfig):
    OUTPUT_FILENAME = "agent_movement"
    OUTPUT_EXTENSION = "png"


class VideoConfig(OutputConfig):
    OUTPUT_FILENAME = "agent_movement"
    OUTPUT_EXTENSION = "mp4"

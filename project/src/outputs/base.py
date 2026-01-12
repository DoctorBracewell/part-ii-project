import matplotlib.pyplot as plt
from collections import defaultdict
from logging import Logger
from abc import ABC, abstractmethod

from configs import simulation as SimulationConfig
from configs import output as OutputConfig

from simulation.simulation import Simulation, Vector, Scalar


class OutputManager:
    def __init__(self, logger: Logger):
        from outputs.plot import PlotOutput
        from outputs.video import VideoOutput

        self.logger = logger
        self.agent_paths: list[list[tuple[Scalar, Vector]]] = [
            [] for _ in range(SimulationConfig.AGENTS)
        ]
        self.outputs: list[BaseOutput] = [
            PlotOutput(logger, self),
            VideoOutput(logger, self),
        ]

    # Called by simulation by callback after each step
    def add_agent_data(self, simulation: Simulation):
        for agent_idx in range(simulation.N):
            heading: Scalar = simulation.headings[agent_idx]
            position: Vector = simulation.positions[agent_idx]
            self.agent_paths[agent_idx].append((heading, position))

    def create_outputs(self):
        for output in self.outputs:
            output.setup_plot_axes()
            output.create()
            output.save()


class BaseOutput(ABC):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        self.logger = logger
        self.output_manager = output_manager
        self.fig, self.ax = plt.subplots(figsize=OutputConfig.FIGSIZE)

    def setup_plot_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("Position")
        self.ax.set_title("Agent Movement")

        # Determine overall x-axis limits
        x_padding: float = SimulationConfig.WIDTH * OutputConfig.X_PADDING_RATIO
        y_padding: float = SimulationConfig.LENGTH * OutputConfig.Y_PADDING_RATIO

        self.ax.set_xlim(0 - x_padding, SimulationConfig.WIDTH + x_padding)
        self.ax.set_ylim(0 - y_padding, SimulationConfig.LENGTH + y_padding)
        self.ax.set_aspect("equal")

    @abstractmethod
    def create(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

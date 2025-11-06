import matplotlib.pyplot as plt
from collections import defaultdict
from logging import Logger
from simulation.simulation import SimulationStatus
from abc import ABC, abstractmethod
from config import OutputConfig
from typing import Any


class OutputManager:
    def __init__(self, logger: Logger):
        from outputs.plot import PlotOutput
        from outputs.video import VideoOutput

        self.logger = logger
        self.agent_paths: dict[int, list[float]] = defaultdict(list)
        self.outputs: list[BaseOutput] = [
            PlotOutput(logger, self),
            VideoOutput(logger, self),
        ]

    # Called by simulation by callback after each step
    def add_agent_positions(self, status: SimulationStatus):
        self.add_point(0, status.pursuer.position)
        self.add_point(1, status.evader.position)

    def add_point(self, agent_id: int, position: float):
        self.agent_paths[agent_id].append(position)

    def create_outputs(self):
        for output in self.outputs:
            output.create()
            output.save()


class BaseOutput(ABC):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        self.logger = logger
        self.output_manager = output_manager
        self.fig, self.ax = plt.subplots(figsize=OutputConfig.FIGSIZE)

    def _setup_plot_axes(self):
        self.ax.clear()
        self.ax.set_xlabel("Position")
        self.ax.set_title("Agent Movement on a 1D Line")
        self.ax.set_yticks([])
        self.ax.set_ylim(OutputConfig.Y_LIM)

        # Determine overall x-axis limits
        all_positions: list[float] = []
        for path in self.output_manager.agent_paths.values():
            all_positions.extend(path)

        if all_positions:
            min_x: float = min(all_positions)
            max_x: float = max(all_positions)
            x_range: float = max_x - min_x
            padding: float = (
                x_range * OutputConfig.X_PADDING_RATIO if x_range > 0 else 1.0
            )
            self.ax.set_xlim(min_x - padding, max_x + padding)
        else:
            self.ax.set_xlim(-1, 1)  # Default limits if no positions

    @abstractmethod
    def create(self) -> None:
        pass

    @abstractmethod
    def save(self) -> None:
        pass

from matplotlib.lines import Line2D  # Import Line2D
from logging import Logger
from configs import output as OutputConfig
from configs.outputs import plot as PlotConfig
from outputs.base import OutputManager, BaseOutput
import os
from datetime import datetime
import numpy as np


class PlotOutput(BaseOutput):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        super().__init__(logger, output_manager)

    def create(self):
        self._setup_plot_axes()

        for agent_id, agent_path in enumerate(self.output_manager.agent_paths):
            path = np.array(agent_path)

            if len(agent_path) > 1:
                # Plot the path line
                self.ax.plot(
                    path[:, 0],
                    path[:, 1],
                    label=f"Agent {agent_id}",
                    linewidth=1.5,
                )

                # Plot start and end circles
                self.ax.plot(
                    path[0, 0],
                    path[0, 1],
                    "o",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=8,
                )
                self.ax.plot(
                    path[-1, 0],
                    path[-1, 1],
                    "o",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=8,
                )

        # Add legend and draw plot
        self.add_legend()
        self.fig.canvas.draw_idle()

    def add_legend(self):
        start_handle = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
        )
        end_handle = Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8,
            linestyle="None",
        )

        self.ax.legend(
            handles=[start_handle, end_handle],
            labels=["Start Positions", "End Positions"],
        )

    def save(self):
        os.makedirs(OutputConfig.OUTPUT_DIRECTORY, exist_ok=True)

        filename = f"{PlotConfig.OUTPUT_FILENAME}_{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}.{PlotConfig.OUTPUT_EXTENSION}"
        filepath = os.path.join(
            OutputConfig.OUTPUT_DIRECTORY,
            filename,
        )

        self.fig.savefig(filepath, dpi=OutputConfig.DPI)
        self.logger.info(f"Plot saved to {filepath}")

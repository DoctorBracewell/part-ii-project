import os
from datetime import datetime
import numpy as np
from matplotlib.lines import Line2D
from outputs.base import BaseOutput, OutputManager
from logging import Logger
from configs import output as OutputConfig
from configs.outputs import plot as PlotConfig


class PlotOutput(BaseOutput):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        super().__init__(logger, output_manager)

    def create(self):
        linestyles = ["--", "-.", ":", "-"]

        for agent_idx, agent_path in enumerate(self.output_manager.agent_paths):
            # Extract positions and stack into (N,3) array
            positions = np.stack([pos for pos, _, _, _ in agent_path])

            if len(positions) > 1:
                # Plot the path
                self.ax.plot(
                    *positions.T,
                    label=f"Agent {agent_idx}",
                    linewidth=1.5,
                    linestyle=linestyles[agent_idx % len(linestyles)],
                )

            self.ax.scatter(
                *positions[0], color="black", s=50, marker="o", label="_nolegend_"
            )  # start
            self.ax.scatter(
                *positions[-1],
                facecolor="white",
                edgecolor="black",
                s=50,
                marker="o",
                label="_nolegend_",
            )  # end

        # Add custom start/end legend
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
        time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        os.makedirs(OutputConfig.OUTPUT_DIRECTORY, exist_ok=True)
        os.makedirs(f"{OutputConfig.OUTPUT_DIRECTORY}/{time}", exist_ok=True)

        filename = f"{PlotConfig.OUTPUT_FILENAME}.{PlotConfig.OUTPUT_EXTENSION}"
        filepath = os.path.join(
            f"{OutputConfig.OUTPUT_DIRECTORY}/{time}",
            filename,
        )

        self.fig.savefig(filepath, dpi=OutputConfig.DPI)
        self.logger.info(f"Plot saved to {filepath}")

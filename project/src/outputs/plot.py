from matplotlib.lines import Line2D  # Import Line2D
from logging import Logger
from config import PlotConfig, OutputConfig
from outputs.base import OutputManager, BaseOutput
from typing import Any
import os
from datetime import datetime


class PlotOutput(BaseOutput):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        super().__init__(logger, output_manager)

    def create(self):
        self._setup_plot_axes()

        y_offset_step = OutputConfig.Y_OFFSET_STEP
        agent_ids = sorted(self.output_manager.agent_paths.keys())

        for i, agent_id in enumerate(agent_ids):
            path = self.output_manager.agent_paths[agent_id]
            y_offset = i * y_offset_step - (len(agent_ids) - 1) * y_offset_step / 2

            if len(path) > 1:
                # Plot the path line
                (line,) = self.ax.plot(
                    path,
                    [y_offset] * len(path),
                    label=f"Agent {agent_id}",
                    linewidth=1.5,
                )

                # Add arrows to indicate direction
                self.add_arrows(path, line.get_color(), y_offset)

                # Plot start and end circles
                self.ax.plot(
                    path[0],
                    y_offset,
                    "o",
                    markerfacecolor="black",
                    markeredgecolor="black",
                    markersize=8,
                )
                self.ax.plot(
                    path[-1],
                    y_offset,
                    "o",
                    markerfacecolor="white",
                    markeredgecolor="black",
                    markersize=8,
                )

        # Add legend and draw plot
        self.add_legend()
        self.fig.canvas.draw_idle()

    def add_arrows(self, path: list[float], line_colour: Any, y_offset: float):
        x_min_path = min(path)
        x_max_path = max(path)
        path_x_range = x_max_path - x_min_path

        if path_x_range > 0:
            num_arrows = 5  # Number of arrows to display per line
            arrow_x_interval = path_x_range / (num_arrows + 1)
            arrow_length_ratio = 0.02  # Length of the arrow relative to x_range

            for arrow_idx in range(num_arrows):
                target_x = x_min_path + (arrow_idx + 1) * arrow_x_interval

                # Find the segment that contains the target_x
                for j in range(len(path) - 1):
                    x1, x2 = path[j], path[j + 1]

                    # Ensure x1 is always less than x2 for consistent comparison
                    if x1 > x2:
                        x1, x2 = x2, x1  # Swap if path is decreasing

                    if x1 <= target_x <= x2:
                        # Found the segment, now determine direction and place arrow
                        segment_direction = 1 if path[j + 1] > path[j] else -1
                        arrow_dx = segment_direction * path_x_range * arrow_length_ratio

                        self.ax.annotate(
                            "",
                            xy=(
                                target_x + arrow_dx / 2,
                                y_offset,
                            ),  # Arrow tip
                            xytext=(
                                target_x - arrow_dx / 2,
                                y_offset,
                            ),  # Arrow base
                            arrowprops=dict(
                                arrowstyle="->",
                                color=line_colour,
                                linewidth=1,
                                shrinkA=0,
                                shrinkB=0,
                            ),
                            annotation_clip=False,
                        )
                        break  # Move to the next arrow after placing one

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

        # Get handles and labels from the current axes (self.ax)
        current_handles, current_labels = self.ax.get_legend_handles_labels()
        current_handles.extend([start_handle, end_handle])
        current_labels.extend(["Start Positions", "End Positions"])

        self.ax.legend(handles=current_handles, labels=current_labels)

    def save(self):
        os.makedirs(PlotConfig.OUTPUT_DIRECTORY, exist_ok=True)

        filename = f"{PlotConfig.OUTPUT_FILENAME}_{datetime.now().strftime('%Y.%m.%d')}.{PlotConfig.OUTPUT_EXTENSION}"
        filepath = os.path.join(
            PlotConfig.OUTPUT_DIRECTORY,
            filename,
        )

        self.fig.savefig(filepath, dpi=OutputConfig.DPI)
        self.logger.info(f"Plot saved to {filepath}")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D  # Import Line2D
from logging import Logger

from configs import output as OutputConfig
from configs import simulation as SimulationConfig
from configs.outputs import video as VideoConfig

from outputs.base import OutputManager, BaseOutput
import os
import numpy as np
from datetime import datetime


class VideoOutput(BaseOutput):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        super().__init__(logger, output_manager)
        self.agent_circles: list[Line2D] = []
        self.agent_colours: dict[int, tuple[float, float, float, float]] = {}

    def create(self):
        self._setup_plot_axes()
        y_offset_step = OutputConfig.Y_OFFSET_STEP
        agent_ids = sorted(self.output_manager.agent_paths.keys())

        # Assign unique colors to agents
        colours = plt.cm.get_cmap("tab10", len(agent_ids))
        for i, agent_id in enumerate(agent_ids):
            self.agent_colours[agent_id] = colours(i)

        # Create plots
        self.agent_circles = []
        for i, agent_id in enumerate(agent_ids):
            path = self.output_manager.agent_paths[agent_id]
            y_offset = i * y_offset_step - (len(agent_ids) - 1) * y_offset_step / 2

            (line,) = self.ax.plot(
                path[0],
                y_offset,
                marker="o",
                color=self.agent_colours[agent_id],
                label=f"Agent {agent_id}",
                markersize=10,  # size of the circle
                linestyle="None",
                zorder=5,  # ensure circles are on top
            )
            self.agent_circles.append(line)

        self.ax.legend()

    def _update(self, frame: int) -> list[Line2D]:
        updated_circles: list[Line2D] = []

        for i, agent_id in enumerate(sorted(self.output_manager.agent_paths.keys())):
            path = self.output_manager.agent_paths[agent_id]

            if frame < len(path):
                self.agent_circles[i].set_data(
                    np.array([path[frame]]),
                    np.array([self.agent_circles[i].get_ydata()[0]]),  # type: ignore
                )
                self.agent_circles[i].set_color(
                    self.agent_colours[agent_id]
                )  # Ensure color is set
                self.agent_circles[i].set_alpha(1)  # Make visible

            else:
                # If an agent's path is shorter than the current frame, keep it at its last position or hide it
                if path:
                    self.agent_circles[i].set_data(
                        np.array([path[-1]]),
                        np.array([self.agent_circles[i].get_ydata()[0]]),  # type: ignore
                    )
                self.agent_circles[i].set_color(
                    self.agent_colours[agent_id]
                )  # Ensure color is set
                self.agent_circles[i].set_alpha(0.5)  # Fade out if path ended
            updated_circles.append(self.agent_circles[i])

        return updated_circles

    def save(self):
        os.makedirs(OutputConfig.OUTPUT_DIRECTORY, exist_ok=True)

        filename = f"{VideoConfig.OUTPUT_FILENAME}_{datetime.now().strftime('%Y.%m.%d')}.{VideoConfig.OUTPUT_EXTENSION}"
        filepath = os.path.join(OutputConfig.OUTPUT_DIRECTORY, filename)

        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            blit=True,
            frames=len(self.output_manager.agent_paths[0]),
            repeat=False,
            cache_frame_data=False,
        )
        ani.save(
            filepath,
            writer="ffmpeg",
            fps=SimulationConfig.STEPS_PER_SECOND,
            dpi=OutputConfig.DPI,
        )

        self.logger.info(f"Video saved to {filepath}")

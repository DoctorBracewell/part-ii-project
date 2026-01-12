import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.quiver import Quiver
from logging import Logger

from configs import output as OutputConfig
from configs import simulation as SimulationConfig
from configs.outputs import video as VideoConfig

from outputs.base import OutputManager, BaseOutput
import os
from datetime import datetime


class VideoOutput(BaseOutput):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        super().__init__(logger, output_manager)
        self.agent_symbols: list[Quiver] = []
        self.agent_colours: list[tuple[float, float, float, float]] = []

    def heading_to_vector(self, heading: float) -> tuple[float, float]:
        """Convert a heading angle (in radians) to a 2D unit vector."""
        import math

        return (math.cos(heading), math.sin(heading))

    def create(self):
        num_agents = len(self.output_manager.agent_paths)

        # Assign unique colors to agents
        colours = plt.cm.get_cmap("tab10", num_agents)

        # Create a circle for each agent at its starting position
        for agent_idx, agent_path in enumerate(self.output_manager.agent_paths):
            vx, vy = self.heading_to_vector(agent_path[0][0])

            symbol = self.ax.quiver(
                agent_path[0][1][0],
                agent_path[0][1][1],
                vx,
                vy,
                color=colours(agent_idx),
                label=f"Agent {agent_idx}",
                zorder=5,  # ensure circles are on top
            )

            self.agent_symbols.append(symbol)
            self.agent_colours.append(colours(agent_idx))

        self.ax.legend()

    def _update(self, frame: int) -> list[Quiver]:
        updated_symbols: list[Quiver] = []

        for agent_idx, agent_path in enumerate(self.output_manager.agent_paths):
            if frame < len(agent_path):
                # Update position
                self.agent_symbols[agent_idx].set_offsets(
                    [agent_path[frame][1][0], agent_path[frame][1][1]]
                )

                # Update heading
                vx, vy = self.heading_to_vector(agent_path[frame][0])
                self.agent_symbols[agent_idx].set_UVC(vx, vy)

            updated_symbols.append(self.agent_symbols[agent_idx])

        return updated_symbols

    def save(self):
        os.makedirs(OutputConfig.OUTPUT_DIRECTORY, exist_ok=True)

        filename = f"{VideoConfig.OUTPUT_FILENAME}_{datetime.now().strftime('%Y.%m.%d-%H:%M:%S')}.{VideoConfig.OUTPUT_EXTENSION}"
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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from logging import Logger
import math
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.artist import Artist

from configs import output as OutputConfig
from configs import simulation as SimulationConfig
from configs.outputs import video as VideoConfig

from outputs.base import OutputManager, BaseOutput
import os
from datetime import datetime


class VideoOutput(BaseOutput):
    def __init__(self, logger: Logger, output_manager: OutputManager):
        super().__init__(logger, output_manager)
        self.agent_quivers: list[Line3DCollection] = []
        self.agent_colours: list[tuple[float, float, float, float]] = []

    def angles_to_orientation_vector(
        self,
        attack_angle: float,
        azimuth_angle: float,
        roll_angle: float,
        length: float = 1.0,
    ) -> tuple[float, float, float]:
        """Converts orientation angles to a 3D vector."""
        # This is a simplified model for orientation visualization.
        # It assumes attack angle is pitch and azimuth is yaw. Roll is not directly used for the vector direction.
        u = length * math.cos(attack_angle) * math.cos(azimuth_angle)
        v = length * math.cos(attack_angle) * math.sin(azimuth_angle)
        w = length * math.sin(attack_angle)
        return u, v, w

    def create(self):
        num_agents = len(self.output_manager.agent_paths)
        colours = plt.cm.get_cmap("tab10", num_agents)

        for agent_idx, agent_path in enumerate(self.output_manager.agent_paths):
            # Quiver for orientation
            pos, attack_angle, _, azimuth_angle, roll_angle = agent_path[0]
            u, v, w = self.angles_to_orientation_vector(
                attack_angle, azimuth_angle, roll_angle, length=500
            )

            quiver = self.ax.quiver(
                [pos[0]],
                [pos[1]],
                [pos[2]],
                [u],
                [v],
                [w],
                color=colours(agent_idx),
                arrow_length_ratio=0.3,
                label=f"Agent {agent_idx}",
            )
            self.agent_quivers.append(quiver)
            self.agent_colours.append(colours(agent_idx))

        self.ax.legend()

    def _update(self, frame: int) -> list[Artist]:
        updated_artists = []

        for agent_idx, agent_path in enumerate(self.output_manager.agent_paths):
            if frame < len(agent_path):
                # Quiver update
                pos, attack_angle, _, azimuth_angle, roll_angle = agent_path[frame]
                u, v, w = self.angles_to_orientation_vector(
                    attack_angle, azimuth_angle, roll_angle, length=500
                )

                segments = [
                    [[pos[0], pos[1], pos[2]], [pos[0] + u, pos[1] + v, pos[2] + w]]
                ]
                self.agent_quivers[agent_idx]._segments3d = segments
                updated_artists.append(self.agent_quivers[agent_idx])

        return updated_artists

    def save(self):
        time = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        os.makedirs(OutputConfig.OUTPUT_DIRECTORY, exist_ok=True)
        os.makedirs(f"{OutputConfig.OUTPUT_DIRECTORY}/{time}", exist_ok=True)

        filename = f"{VideoConfig.OUTPUT_FILENAME}.{VideoConfig.OUTPUT_EXTENSION}"
        filepath = os.path.join(
            f"{OutputConfig.OUTPUT_DIRECTORY}/{time}",
            filename,
        )

        ani = animation.FuncAnimation(
            self.fig,
            self._update,
            blit=False,
            frames=len(self.output_manager.agent_paths[0]),
            repeat=False,
        )
        ani.save(
            filepath,
            writer="ffmpeg",
            fps=SimulationConfig.STEPS_PER_SECOND,
            dpi=OutputConfig.DPI,
        )
        self.logger.info(f"Video saved to {filepath}")

from logging import Logger
import pyvista as pv
import pyvistaqt as pvqt
from datetime import datetime
import os

import numpy as np
from PyQt5 import QtWidgets, QtCore

from configs import output as OutputConfig
from configs.outputs import visualisation as VisualisationConfig
from configs import simulation as SimulationConfig

from outputs.base import OutputManager, BaseOutput
from simulation.simulation import Scalar


class VisualisationOutput(BaseOutput):
    # Setup
    def __init__(self, logger: Logger, output_manager: OutputManager):
        super().__init__(logger, output_manager)

        self.plotter = pvqt.BackgroundPlotter(window_size=(750, 750))

        self.actors: list[pv.Actor] = []
        self.current_timestep = 0
        self.is_playing = True

    def create(self):
        self.timesteps = len(self.output_manager.agent_paths[0])

        self.setup_scene()
        self.set_timestep(0)

        self.setup_controls()
        self.setup_animation()

    def save(self):
        pass

    def setup_scene(self):
        # Background
        self.plotter.set_background(
            color="white",
            top="lightblue",
        )  # type: ignore

        # Ground plane
        floor = pv.Plane(
            center=(5000, 5000, 0),
            i_size=10000,
            j_size=10000,
            i_resolution=20,
            j_resolution=20,
        )

        # Axis lines
        x_axis = pv.Line(pointa=(0, 0, 0), pointb=(10000, 0, 0))
        y_axis = pv.Line(pointa=(0, 0, 0), pointb=(0, 10000, 0))
        z_axis = pv.Line(pointa=(0, 0, 0), pointb=(0, 0, 10000))

        self.plotter.add_mesh(floor, color="lightgray", style="wireframe", opacity=0.5)
        self.plotter.add_mesh(x_axis, color="red", line_width=2)
        self.plotter.add_mesh(y_axis, color="green", line_width=2)
        self.plotter.add_mesh(z_axis, color="blue", line_width=2)

        num_agents = len(self.output_manager.agent_paths)
        for i in range(num_agents):
            arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=200)
            actor = self.plotter.add_mesh(
                arrow,
                color=VisualisationConfig.COLOURS[i % len(VisualisationConfig.COLOURS)],
            )
            self.actors.append(actor)

    # Animation
    def setup_controls(self):
        dock = QtWidgets.QDockWidget("Controls", self.plotter.app_window)
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(self.timesteps - 1)
        self.slider.setValue(0)
        self.slider.valueChanged.connect(self.set_timestep)

        self.play_button = QtWidgets.QPushButton("Pause")
        self.play_button.clicked.connect(self.toggle_play)

        layout.addWidget(self.slider)
        layout.addWidget(self.play_button)

        dock.setWidget(widget)

        self.plotter.app_window.addDockWidget(
            QtCore.Qt.BottomDockWidgetArea,
            dock,
        )

    def setup_animation(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.animate)
        self.timer.start(int(1000 / SimulationConfig.STEPS_PER_SECOND))

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.play_button.setText("Pause" if self.is_playing else "Play")

    def animate(self):
        if not self.is_playing:
            return

        self.current_timestep += 1
        if self.current_timestep >= self.timesteps:
            self.current_timestep = 0

        self.slider.blockSignals(True)
        self.slider.setValue(self.current_timestep)
        self.slider.blockSignals(False)

        self.set_timestep(self.current_timestep)

    def set_timestep(self, t: int):
        t = int(t)
        self.current_timestep = t

        for i, actor in enumerate(self.actors):
            agent_data = self.output_manager.agent_paths[i][t]
            actor.SetPosition(agent_data[0])  # type: ignore
            actor.SetOrientation(*get_pyvista_orientation(*agent_data[1:5]))

        self.plotter.render()


def get_pyvista_orientation(
    attack_angle: Scalar,
    flight_path_angle: Scalar,
    azimuth_angle: Scalar,
    roll_angle: Scalar,
) -> tuple[float, float, float]:
    pitch = flight_path_angle + attack_angle
    pitch_deg = np.rad2deg(pitch)
    yaw_deg = np.rad2deg(azimuth_angle)
    roll_deg = np.rad2deg(roll_angle)

    return (pitch_deg, roll_deg, yaw_deg)

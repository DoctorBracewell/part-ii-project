from logging import Logger
import pyvista as pv
import pyvistaqt as pvqt
from datetime import datetime
import os

import numpy as np
from PyQt5 import QtWidgets, QtCore

from configs import visualisation as VisualisationConfig
from configs import simulation as SimulationConfig

from simulation.simulation import Simulation, Scalar


class VisualisationManager:
    # Setup
    def __init__(self, logger: Logger, agent_count: int):
        self.logger = logger

        self.plotter = pvqt.BackgroundPlotter(window_size=(750, 750))
        self.planes: list[pv.Actor] = []
        self.capture_points: list[pv.Actor] = []

        self.setup_scene(agent_count)

    def setup_scene(self, agent_count: int):
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

        for i in range(agent_count):
            arrow = pv.Arrow(start=(0, 0, 0), direction=(1, 0, 0), scale=200)
            plane = self.plotter.add_mesh(
                arrow,
                color=VisualisationConfig.COLOURS[i % len(VisualisationConfig.COLOURS)],
            )
            self.planes.append(plane)

            sphere = pv.Sphere(radius=30, center=(0, 0, 0))
            capture_point = self.plotter.add_mesh(  # type: ignore
                sphere,
                color=VisualisationConfig.COLOURS[i % len(VisualisationConfig.COLOURS)],
            )
            self.capture_points.append(capture_point)

    def update(self, simulation: Simulation):
        for i, plane in enumerate(self.planes):
            position = simulation.positions[i]
            attack_angle = simulation.attack_angles[i]
            flight_path_angle = simulation.flight_path_angles[i]
            azimuth_angle = simulation.azimuth_angles[i]
            roll_angle = simulation.roll_angles[i]

            plane.SetPosition(position)  # type: ignore
            plane.SetOrientation(
                *get_pyvista_orientation(
                    attack_angle, flight_path_angle, azimuth_angle, roll_angle
                )
            )

        for i, capture_point in enumerate(self.capture_points):
            position = simulation.capture_points[i]
            capture_point.SetPosition(position)  # type: ignore

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

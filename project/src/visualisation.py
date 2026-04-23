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
    def __init__(self, logger: Logger, agent_count: int):
        self.logger = logger

        self.plotter = pvqt.BackgroundPlotter(window_size=(750, 750))
        self.planes: list[pv.Actor] = []
        self.capture_points: list[pv.Actor] = []
        self.texts: list[pv.Actor] = []
        self.velocity_arrows: list[pv.Actor] = []

        # Written by sim thread, read by Qt timer — plain Python types, GIL-safe
        self._show_speeds = False
        self._agent_data: list[tuple[np.ndarray, float, bool]] = []

        self.setup_scene(agent_count)

        # One QLabel per agent, overlaid on the render widget
        label_style = (
            "color: black; font-size: 12pt; font-weight: bold;"
            " background: rgba(255,255,255,180); padding: 3px; border-radius: 3px;"
        )
        self._speed_labels: list[QtWidgets.QLabel] = []
        for _ in range(agent_count):
            label = QtWidgets.QLabel("", self.plotter)
            label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
            label.setStyleSheet(label_style)
            label.hide()
            self._speed_labels.append(label)

        # Qt timer fires on the Qt thread — safe to update labels and query the renderer here
        self._text_timer = QtCore.QTimer()
        self._text_timer.timeout.connect(self._refresh_speed_labels)
        self._text_timer.start(50)

    def setup_scene(self, agent_count: int):
        self.plotter.set_background(color="white", top="lightblue")  # type: ignore

        floor = pv.Plane(
            center=(5000, 5000, 0),
            i_size=10000,
            j_size=10000,
            i_resolution=20,
            j_resolution=20,
        )
        x_axis = pv.Line(pointa=(0, 0, 0), pointb=(10000, 0, 0))
        y_axis = pv.Line(pointa=(0, 0, 0), pointb=(0, 10000, 0))
        z_axis = pv.Line(pointa=(0, 0, 0), pointb=(0, 0, 10000))

        self.plotter.add_mesh(floor, color="lightgray", style="wireframe", opacity=0.5)
        self.plotter.add_mesh(x_axis, color="red", line_width=2)
        self.plotter.add_mesh(y_axis, color="green", line_width=2)
        self.plotter.add_mesh(z_axis, color="blue", line_width=2)

        for i in range(agent_count):
            model = pv.read("src/assets/jet.stl")
            model.scale(3, inplace=True)
            model.rotate_z(90, inplace=True)
            plane = self.plotter.add_mesh(
                model,
                color=VisualisationConfig.COLOURS[i % len(VisualisationConfig.COLOURS)],
            )
            self.planes.append(plane)

            sphere = pv.Sphere(radius=30, center=(0, 0, 0))
            self.plotter.add_mesh(  # type: ignore
                sphere,
                color=VisualisationConfig.COLOURS[i % len(VisualisationConfig.COLOURS)],
            )

            # Velocity arrow — pre-created and hidden; shown/hidden via opacity only.
            # direction=(1,0,0) so the arrow starts pointing in +x with identity orientation,
            # matching the jet mesh which ends up in +x after its rotate_z(90).
            arrow_mesh = pv.Arrow(
                start=(0, 0, 0),
                direction=(1, 0, 0),
                scale=250,
                shaft_radius=0.02,
                tip_radius=0.06,
                tip_length=0.2,
            )
            vel_arrow = self.plotter.add_mesh(arrow_mesh, color="gray")
            vel_arrow.GetProperty().SetOpacity(0)
            self.velocity_arrows.append(vel_arrow)

    def _refresh_speed_labels(self):
        """Qt timer callback — runs on the Qt thread. Safe to query renderer and update Qt widgets."""
        renderer = self.plotter.renderer
        # Qt sizes are in logical pixels; VTK display coords are in device pixels on Retina
        dpr = self.plotter.devicePixelRatioF()
        win_h = self.plotter.size().height()

        for label, (pos, speed, active) in zip(self._speed_labels, self._agent_data):
            if not active or not self._show_speeds:
                label.hide()
                continue

            # Project world position (50m above agent) to display coordinates
            above = pos + np.array([0.0, 0.0, 50.0])
            renderer.SetWorldPoint(above[0], above[1], above[2], 1.0)
            renderer.WorldToDisplay()
            dx, dy, _ = renderer.GetDisplayPoint()

            # Convert device pixels → logical pixels, flip y (VTK: bottom-left, Qt: top-left)
            lx = dx / dpr
            ly = win_h - dy / dpr

            # Update text first so adjustSize gives the correct dimensions
            label.setText(f"{speed:.0f} m/s")
            label.adjustSize()

            # Centre label horizontally, sit its bottom edge at the projected point
            label.move(int(lx - label.width() / 2), int(ly - label.height()))
            label.show()
            label.raise_()

    def update(self, simulation: Simulation):
        # Update jet positions and orientations
        for i, plane in enumerate(self.planes[:simulation.N]):
            plane.SetPosition(simulation.positions[i])  # type: ignore
            plane.SetOrientation(
                *get_pyvista_orientation(
                    simulation.attack_angles[i],
                    simulation.flight_path_angles[i],
                    simulation.azimuth_angles[i],
                    simulation.roll_angles[i],
                )
            )
            if not simulation.active[i]:
                plane.GetProperty().SetOpacity(0.2)
                plane.GetProperty().SetColor(0.5, 0.5, 0.5)

        for i, capture_point in enumerate(self.capture_points):
            capture_point.SetPosition(simulation.capture_points[i])  # type: ignore

        # Zoom check — compare camera distance to nearest active agent
        camera_pos = np.array(self.plotter.camera.position)
        min_dist = min(
            (
                np.linalg.norm(camera_pos - simulation.positions[i])
                for i in range(simulation.N)
                if simulation.active[i]
            ),
            default=float("inf"),
        )
        zoomed_in = min_dist < 6000

        # Velocity arrows — only safe property setters, no add/remove
        for i, arrow in enumerate(self.velocity_arrows[:simulation.N]):
            if simulation.active[i] and zoomed_in:
                arrow.SetPosition(simulation.positions[i])
                arrow.SetOrientation(
                    *get_pyvista_orientation(
                        0,
                        simulation.flight_path_angles[i],
                        simulation.azimuth_angles[i],
                        0,
                    )
                )
                arrow.GetProperty().SetOpacity(0.4)
            else:
                arrow.GetProperty().SetOpacity(0)

        # Hand data to Qt timer — plain Python writes, GIL-safe
        self._show_speeds = zoomed_in
        self._agent_data = [
            (simulation.positions[i].copy(), float(simulation.speeds[i]), bool(simulation.active[i]))
            for i in range(simulation.N)
        ]

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

    return (roll_deg, -pitch_deg, yaw_deg)

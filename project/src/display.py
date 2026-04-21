from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from rich.table import Table
from rich.box import SQUARE
from rich.text import Text

import numpy as np
from datetime import datetime
from typing import Type
from types import TracebackType

from simulation.simulation import Simulation, Vector

from configs import simulation as SimulationConfig
from configs import display as DisplayConfig


class Display:
    def __init__(self, console: Console):
        self.start_time = datetime.now().timestamp()

        self.values = Layout(ratio=1)
        self.map = Layout(ratio=2)
        self.full = Layout()
        self.full.split_row(self.values, self.map)
        self.console = console
        self.live = Live(
            self.full,
            auto_refresh=False,
            console=self.console,
            redirect_stdout=False,
        )

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        self.live.update(Group())
        self.live.stop()

    def update(self, simulation: Simulation):
        self.values.update(self.make_values(simulation))
        self.map.update(self.make_map(simulation))
        self.live.refresh()

    def make_values(self, simulation: Simulation) -> Panel:
        timestep = simulation.timestep
        simulation_time = timestep / SimulationConfig.STEPS_PER_SECOND
        real_time = datetime.now().timestamp() - self.start_time
        agent_count = simulation.N
        hard_deck = SimulationConfig.HARD_DECK
        capture_buffer = simulation.capture_buffer

        values = [
            f"Timestep: {timestep}",
            f"Simulation Time: {simulation_time:.3f}s",
            f"Real Time: {real_time:.3f}s",
            "",
            f"Agents: {agent_count}",
            f"Hard Deck: {hard_deck}",
            f"Thrust Ratio: {simulation.thrust_ratio}",
            f"Attack Angle Ratio: {simulation.attack_angle_ratio}",
            f"Roll Angle Ratio: {simulation.roll_angle_ratio}",
            f"",
            f"Capture Buffer: {capture_buffer}",
            f"Distance Check: {simulation.distance_check}",
            f"Nose Check: {simulation.nose_check}",
            f"Asymmetry Check: {simulation.asymmetry_check}",
            f"",
            f"Positions: {', '.join([f'({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})' for pos in simulation.positions])}",
            f"Speeds: {', '.join([f'{s:.1f}' for s in simulation.speeds])}",
            f"Thrusts: {', '.join([f'{t:.1f}' for t in simulation.thrusts])}",
            f"Attack Angles: {', '.join([f'{a:.1f}' for a in simulation.attack_angles])}",
            f"Roll Angles: {', '.join([f'{r:.1f}' for r in simulation.roll_angles])}",
            f"Azimuth Angles: {', '.join([f'{np.rad2deg(a % (2 * np.pi)):.1f}°' for a in simulation.azimuth_angles])}",
            f"Flight Path Angles: {', '.join([f'{np.rad2deg(a):.1f}°' for a in simulation.flight_path_angles])}",
            f"Chosen Actions: {', '.join([f'(T: {a[0]:.1f}, AAR: {a[1]:.1f}, RAR: {a[2]:.1f})' for a in simulation.chosen_actions])}",
        ]

        return Panel(
            self.console.render_str("\n".join(values)),
            title="[bold]Simulation Status[/bold]",
            title_align="left",
        )

    def make_map(self, simulation: Simulation) -> Panel:
        # --- Symbol Calculation ---
        symbols = ["→", "↘", "↓", "↙", "←", "↖", "↑", "↗"]
        # Predefined color palette for agents
        agent_colors = ["red", "green", "blue", "magenta", "cyan", "white"]

        headings = (-simulation.azimuth_angles) % (2 * np.pi)
        indices = np.floor((headings + np.pi / 8) / (2 * np.pi) * 8).astype(int) % 8
        agent_symbols: list[str] = [symbols[i] for i in indices]

        # --- Grid Logic ---
        grid_size = DisplayConfig.MAP_HEIGHT
        BLANK = "　"
        # Initialize a 2D list of Rich Text objects for formatting
        map_grid = [[Text(BLANK) for _ in range(grid_size)] for _ in range(grid_size)]

        scale_x, scale_y = (
            grid_size / SimulationConfig.WIDTH,
            grid_size / SimulationConfig.LENGTH,
        )

        # Use enumerate to get a stable index for each agent
        for i, (pos, sym) in enumerate(zip(simulation.positions[:, :2], agent_symbols)):
            x = int(max(0, min(grid_size - 1, pos[0] * scale_x)))
            y = int(max(0, min(grid_size - 1, pos[1] * scale_y)))

            # Assign a stable color based on the agent's unique index; gray if captured
            color = "bright_black" if not simulation.active[i] else agent_colors[i % len(agent_colors)]
            map_grid[grid_size - 1 - y][x] = self.console.render_str(
                f"[bold][{color}]{sym}[/{color}][/bold]"
            )

        # --- Rendering ---
        map_display = Table.grid()
        for row in map_grid:
            combined_row = Text("")
            for char in row:
                combined_row.append(char)
            map_display.add_row(combined_row)

        map_panel = Panel(
            map_display,
            box=SQUARE,
            border_style="bold yellow",
            padding=0,
        )

        layout = Table.grid(padding=0)
        layout.add_column(justify="right", vertical="top")
        layout.add_column()

        # Y-Axis Column
        y_axis_labels = self.console.render_str(
            f"{SimulationConfig.LENGTH}{('\n' * (grid_size + 1))}0　　　"
        )
        layout.add_row(y_axis_labels, map_panel)

        # X-Axis Row
        x_max = str(SimulationConfig.WIDTH)
        x_row = self.console.render_str(f" 0{' ' * (grid_size * 2 - 2)}{x_max}")
        layout.add_row("", x_row)

        return Panel(
            Align.center(layout),
            title="[bold]Simulation Preview[/bold]",
            title_align="left",
            padding=(1, 1),
            expand=True,
        )




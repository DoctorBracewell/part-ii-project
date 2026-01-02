from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console, Group, ConsoleOptions, RenderResult
from rich.ansi import AnsiDecoder

# import asciichartpy as achart
import plotext as plt
import psutil
import os
import math
from datetime import datetime

from typing import Type
from types import TracebackType
from simulation.simulation import Simulation
from configs import simulation as SimulationConfig


class Display:
    def __init__(self, console: Console):
        self.charts_updated_at = 0
        self.start_time = datetime.now().timestamp()
        self.system_metrics = SystemMetrics()
        self.cpu_chart = Chart(colour="red")
        self.mem_chart = Chart(colour="blue")

        # Upper layouts
        self.values = Layout(ratio=1)
        self.map = Layout(ratio=2)
        self.upper = Layout()
        self.upper.split_row(self.values, self.map)

        # Lower layouts
        self.cpu = Layout(ratio=1)
        self.mem = Layout(ratio=1)
        self.lower = Layout()
        self.lower.split_row(self.cpu, self.mem)

        # Full layout
        self.full = Layout()
        self.full.split_column(self.upper, self.lower)
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

        if datetime.now().timestamp() - self.charts_updated_at > 1:
            self.charts_updated_at = datetime.now().timestamp()
            self.cpu.update(self.make_cpu())
            self.mem.update(self.make_mem())

        self.live.refresh()

    def make_values(self, simulation: Simulation) -> Panel:
        timestep = simulation.timestep
        simulation_time = timestep / SimulationConfig.STEPS_PER_SECOND
        real_time = round(datetime.now().timestamp() - self.start_time, 3)
        agent_count = simulation.N
        hard_deck = SimulationConfig.HARD_DECK

        return Panel(
            self.console.render_str(
                f"Timestep: {timestep}\nSimulation Time: {simulation_time}s\nReal Time: {real_time}s\n\nAgents: {agent_count}\nHard Deck: {hard_deck}"
            ),
            title="[bold]Simulation Status[/bold]",
            title_align="left",
        )

    def make_cpu(self) -> Panel:
        self.cpu_chart.add(self.system_metrics.query_cpu_usage())

        return Panel(
            self.cpu_chart,
            title="[bold]CPU Usage - %[/bold]",
            title_align="left",
        )

    def make_mem(self) -> Panel:
        self.mem_chart.add(self.system_metrics.query_memory_usage() / 1024**2)

        return Panel(
            self.mem_chart,
            title="[bold]Memory Usage - MB[/bold]",
            title_align="left",
        )


class SystemMetrics:
    def __init__(self):
        self.process = psutil.Process(os.getpid())

    def query_memory_usage(self) -> float:
        return self.process.memory_info().rss

    def query_cpu_usage(self) -> float:
        return self.process.cpu_percent()


class Chart:
    def __init__(self, colour: str):
        self.data: list[float] = []
        self.colour = colour
        self.decoder = AnsiDecoder()

    def add(self, value: float):
        self.data.append(value)

    def make_plot(self, width: int, height: int) -> str:
        short_data = self.data[3:]
        data = short_data[-width + 10 :]

        xs = list(range(len(data)))
        ys = data

        plt.clear_figure()
        plt.plot(xs, ys, marker="hd", color=self.colour)

        plt.xlim(-3, width - 10)
        plt.xticks([])
        plt.ylim((min(ys) if ys else 0) - 3, (max(ys) if ys else 0) + 3)

        plt.frame(False)  # hides the frame
        plt.clear_color()
        plt.grid(False)  # hides grid
        plt.plotsize(width, height)

        return plt.build()  # type: ignore

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        # sliced_data = self.data[5:]
        # data = sliced_data[(-options.max_width + 10) :]

        self.width = options.max_width or console.width
        self.height = options.height or console.height
        canvas = self.make_plot(self.width, self.height)

        self.rich_canvas = Group(*self.decoder.decode(canvas))
        yield self.rich_canvas

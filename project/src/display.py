from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.console import Console, Group

from typing import Type
from types import TracebackType
from simulation.simulation import SimulationStatus


class Display:
    def __init__(self, console: Console):
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

    def update(self, status: SimulationStatus):
        self.values.update(self.make_values(status))
        self.live.refresh()

    def make_values(self, status: SimulationStatus) -> Panel:
        return Panel(
            self.console.render_str(
                f"Timestep: {status.timestep}\nAgents: {status.agents}\nHard Deck: {status.hard_deck}"
            ),
            title="[bold]Simulation Status[/bold]",
            title_align="left",
        )

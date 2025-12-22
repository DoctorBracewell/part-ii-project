from unittest.mock import patch, MagicMock
from display import Display
from simulation.simulation import SimulationStatus
from rich.console import Console


@patch("display.Live")
def test_display_initialization(mock_live: MagicMock):
    console = Console()
    display = Display(console)
    assert display.console == console
    mock_live.assert_called_once_with(
        display.full,
        auto_refresh=False,
        console=console,
        redirect_stdout=False,
    )


@patch("display.Live")
def test_display_context_manager(mock_live: MagicMock):
    console = Console()
    with Display(console):
        mock_live.return_value.start.assert_called_once()
    mock_live.return_value.stop.assert_called_once()


@patch("display.Live")
def test_display_update_values(mock_live: MagicMock):
    console = Console()
    display = Display(console)
    status = SimulationStatus()
    with patch.object(display, "make_values") as mock_make_values:
        display.update(status)
        mock_make_values.assert_called_once_with(status)
        mock_live.return_value.refresh.assert_called_once()


@patch("display.Live")
def test_display_update_charts(mock_live: MagicMock):
    console = Console()
    display = Display(console)
    status = SimulationStatus()
    with patch.object(display, "make_cpu") as mock_make_cpu, patch.object(
        display, "make_mem"
    ) as mock_make_mem:
        display.charts_updated_at = 0
        display.update(status)
        mock_make_cpu.assert_called_once()
        mock_make_mem.assert_called_once()


def test_make_values():
    console = Console()
    display = Display(console)
    status = SimulationStatus()
    panel = display.make_values(status)
    assert "Timestep: 0" in str(panel.renderable)


@patch("display.SystemMetrics.query_cpu_usage", return_value=50.0)
def test_make_cpu(mock_query_cpu: MagicMock):
    console = Console()
    display = Display(console)
    panel = display.make_cpu()
    assert "[bold]CPU Usage - %[/bold]" in str(panel.title)
    mock_query_cpu.assert_called_once()


@patch("display.SystemMetrics.query_memory_usage", return_value=1024.0**2)
def test_make_mem(mock_query_mem: MagicMock):
    console = Console()
    display = Display(console)
    panel = display.make_mem()
    assert "[bold]Memory Usage - MB[/bold]" in str(panel.title)
    mock_query_mem.assert_called_once()

from unittest.mock import Mock, patch, MagicMock
from outputs.base import OutputManager
from simulation.simulation import SimulationStatus


@patch("outputs.plot.PlotOutput")
@patch("outputs.video.VideoOutput")
def test_output_manager_initialization(
    mock_video_output: MagicMock, mock_plot_output: MagicMock
):
    mock_logger = Mock()
    manager = OutputManager(logger=mock_logger)
    assert manager.logger == mock_logger
    assert len(manager.outputs) == 2


@patch("outputs.plot.PlotOutput")
@patch("outputs.video.VideoOutput")
def test_add_agent_positions(mock_video_output: MagicMock, mock_plot_output: MagicMock):
    mock_logger = Mock()
    manager = OutputManager(logger=mock_logger)
    status = SimulationStatus()
    status.pursuer.position = 10.0
    status.evader.position = 20.0
    manager.add_agent_positions(status)
    assert manager.agent_paths[0] == [10.0]
    assert manager.agent_paths[1] == [20.0]


@patch("outputs.plot.PlotOutput")
@patch("outputs.video.VideoOutput")
def test_create_outputs(mock_video_output: MagicMock, mock_plot_output: MagicMock):
    mock_logger = Mock()
    manager = OutputManager(logger=mock_logger)
    manager.create_outputs()
    mock_plot_output.return_value.create.assert_called_once()
    mock_plot_output.return_value.save.assert_called_once()
    mock_video_output.return_value.create.assert_called_once()
    mock_video_output.return_value.save.assert_called_once()

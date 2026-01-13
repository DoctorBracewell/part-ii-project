from unittest.mock import Mock, patch, MagicMock
from outputs.base import OutputManager
from simulation.simulation import Simulation
import numpy as np


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
def test_add_agent_data(mock_video_output: MagicMock, mock_plot_output: MagicMock):
    mock_logger = Mock()
    manager = OutputManager(logger=mock_logger)
    simulation = Simulation(N=2)
    simulation.positions = np.array([[10.0, 10.0], [20.0, 20.0]])
    simulation.headings = np.array([0.0, 1.0])

    manager.add_agent_data(simulation)

    assert len(manager.agent_paths) == 2
    assert len(manager.agent_paths[0]) == 1
    assert manager.agent_paths[0][0][0] == 0.0
    assert np.array_equal(manager.agent_paths[0][0][1], np.array([10.0, 10.0]))


@patch("outputs.plot.PlotOutput")
@patch("outputs.video.VideoOutput")
def test_create_outputs(mock_video_output: MagicMock, mock_plot_output: MagicMock):
    mock_logger = Mock()
    manager = OutputManager(logger=mock_logger)
    manager.create_outputs()
    mock_plot_output.return_value.setup_plot_axes.assert_called_once()
    mock_plot_output.return_value.create.assert_called_once()
    mock_plot_output.return_value.save.assert_called_once()
    mock_video_output.return_value.setup_plot_axes.assert_called_once()
    mock_video_output.return_value.create.assert_called_once()
    mock_video_output.return_value.save.assert_called_once()

import pytest
from unittest.mock import Mock, patch, MagicMock
from outputs.base import OutputManager
import numpy as np

pytestmark = pytest.mark.skip(reason="output tests not currently maintained")


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
def test_add_agent_data(mock_video_output: MagicMock, mock_plot_output: MagicMock, make_simulation):
    mock_logger = Mock()
    manager = OutputManager(logger=mock_logger)
    simulation = make_simulation(N=3)
    simulation.positions = np.array([[10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]])
    simulation.attack_angles = np.array([0.0, 1.0, 0.0])
    simulation.azimuth_angles = np.array([0.0, 1.0, 0.0])
    simulation.roll_angles = np.array([0.0, 1.0, 0.0])

    manager.add_agent_data(simulation)

    assert len(manager.agent_paths) == 3
    assert len(manager.agent_paths[0]) == 1
    assert np.array_equal(manager.agent_paths[0][0][0], np.array([10.0, 10.0, 10.0]))
    assert manager.agent_paths[0][0][1] == 0.0


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

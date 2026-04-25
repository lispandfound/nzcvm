"""Tests that each CLI subcommand can be invoked with --help."""

from typer.testing import CliRunner

from nzcvm.scripts.nzcvm import app

runner = CliRunner()


def test_root_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_generate_help():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0


def test_construct_mesh_help():
    result = runner.invoke(app, ["construct-mesh", "--help"])
    assert result.exit_code == 0


def test_convert_tomography_help():
    result = runner.invoke(app, ["convert-tomography", "--help"])
    assert result.exit_code == 0


def test_convert_topography_help():
    result = runner.invoke(app, ["convert-topography", "--help"])
    assert result.exit_code == 0


def test_tree_stats_help():
    result = runner.invoke(app, ["tree-stats", "--help"])
    assert result.exit_code == 0


def test_view_basin_help():
    result = runner.invoke(app, ["view-basin", "--help"])
    assert result.exit_code == 0

"""Tests that each CLI subcommand can be invoked with --help."""

import pytest
from typer.testing import CliRunner

from nzcvm.scripts.nzcvm import app

runner = CliRunner()


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["generate"],
        ["basin"],
        ["tomography"],
        ["surface"],
        ["tree-stats"],
        ["view"],
    ],
)
def test_help(args: list[str]):
    result = runner.invoke(app, args + ["--help"])
    assert result.exit_code == 0

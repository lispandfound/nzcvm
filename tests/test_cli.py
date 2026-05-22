"""CLI smoke tests.

The only guarantee tested here is that every top-level subcommand can be
invoked with ``--help`` and exits cleanly (exit code 0).  This avoids any
dependency on data files or model configuration.
"""

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
def test_help_exits_cleanly(args: list[str]) -> None:
    result = runner.invoke(app, args + ["--help"])
    assert result.exit_code == 0

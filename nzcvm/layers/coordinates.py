"""Pipeline layer for applying a :class:`~nzcvm.coordinates.CoordinateSystem` transform."""
import xarray as xr
from rich.console import Console, ConsoleOptions, RenderResult
from rich.tree import Tree

from nzcvm.coordinates import Coordinate, CoordinateSystem
from nzcvm.layers import helpers
from nzcvm.layers.protocol import QueryLayer

NO_ORIGIN = 0


class CoordinateTransformLayer:
    """Pipeline layer that converts local grid coordinates to a projected CRS.

    Applies ``coordinate_system.transform`` to the ``x``, ``y``, ``z``
    variables of every ``/block/*`` node, then passes the result to
    *next_layer*.

    Parameters
    ----------
    coordinate_system :
        Defines the origin, azimuth, and target CRS.
    next_layer :
        Downstream layer to invoke after the coordinate transform.

    See Also
    --------
    nzcvm.coordinates.CoordinateSystem : The transform applied by this layer.
    """

    def __init__(
        self, coordinate_system: CoordinateSystem, next_layer: QueryLayer
    ) -> None:
        """
        Parameters
        ----------
        coordinate_system :
            Defines origin, azimuth, and target CRS.
        next_layer :
            Downstream layer invoked after the transform.
        """
        self.coordinate_system = coordinate_system
        self.next_layer = next_layer

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:
        """Transform coordinates and delegate to the next layer.

        Parameters
        ----------
        velocity_model :
            DataTree with local-grid ``x``, ``y``, ``z`` coordinate variables.

        Returns
        -------
        xarray.DataTree
        """

        def _coordinate_transform(_path, block: xr.Dataset) -> xr.Dataset:
            """Apply the coordinate system transform to a single block."""
            block = block.copy()

            x, y, z = self.coordinate_system.transform(
                block["x"], block["y"], block["z"]
            )
            block[Coordinate.X] = x
            block[Coordinate.Y] = y
            block[Coordinate.Z] = z
            return block

        return self.next_layer(helpers.block_map(velocity_model, _coordinate_transform))

    def __rich_console__(
        self, _console: Console, _options: ConsoleOptions
    ) -> RenderResult:
        """Render the pipeline chain as a rich tree."""
        tree = Tree("[bold blue]Change in Coordinates[/]")
        tree.add(self.coordinate_system)  # ty: ignore[invalid-argument-type]
        tree.add(self.next_layer)

        yield tree

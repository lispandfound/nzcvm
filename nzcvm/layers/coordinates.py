from nzcvm.coordinates import Coordinate, CoordinateSystem
from nzcvm.layers.protocol import QueryLayer
from nzcvm.layers import helpers
from rich.tree import Tree
import xarray as xr


from rich.console import Console, ConsoleOptions, RenderResult

NO_ORIGIN = 0


class CoordinateTransformLayer:
    def __init__(
        self, coordinate_system: CoordinateSystem, next_layer: QueryLayer
    ) -> None:
        self.coordinate_system = coordinate_system
        self.next_layer = next_layer

    def __call__(self, velocity_model: xr.DataTree) -> xr.DataTree:
        def _coordinate_transform(_path, block: xr.Dataset) -> xr.Dataset:
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

        tree = Tree("[bold blue]Change in Coordinates[/]")
        tree.add(self.coordinate_system)
        tree.add(self.next_layer)

        yield tree

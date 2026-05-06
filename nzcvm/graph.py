from pathlib import Path
import dask
import xarray as xr


def export_datatree_graph(dt: xr.DataTree, filename: str | Path):
    """
    Traverses an xarray.DataTree to find all lazy Dask arrays,
    and exports their combined computational flow to a browser-viewable file.

    Parameters
    ----------
    dt : xr.DataTree
        The DataTree containing the velocity model grids.
    filename : str
        Output filename. Use '.svg' to allow deep zooming in web browsers.
    """
    lazy_arrays = []

    for node in dt.subtree:
        variables = list(node.data_vars.values()) + list(node.coords.values())

        for var in variables:
            if hasattr(var.data, "dask"):
                lazy_arrays.append(var.data)

    dask.visualize(
        *lazy_arrays, filename=str(filename), color="order", optimize_graph=True
    )

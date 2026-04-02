from .nzcvm import ModelBuilder, PyModel
from dataclasses import dataclass
import xarray as xr
import numpy as np
from pathlib import Path
import xoak


@dataclass
class Quality:
    rho: float
    vp: float
    vs: float
    qp: float
    qs: float


class Model:
    """A high-level wrapper for the Rust ModelTree."""

    def __init__(self, internal_py_model: PyModel):
        self._raw = internal_py_model
        self._builder = ModelBuilder()

    @classmethod
    def from_dataset(cls, ds: xr.Dataset):
        """
        Creates a Model from an xarray Dataset.
        Expects coords 'x', 'y', 'z' and vars 'rho', 'vp', 'vs', 'qp', 'qs'.
        """
        # 1. Extract dimensions (ni, nj, nk)
        # We assume the data is structured such that the first 3 dims are the spatial ones
        shape = ds.rho.shape
        if len(shape) != 3:
            # Handle 1D/2D by padding the shape to 3D for the Rust 'chart' logic
            padded_shape = shape + (1,) * (3 - len(shape))
        else:
            padded_shape = shape

        # 2. Stack vertices into (N, 3) float32
        # Order: x, y, z
        vertices = np.stack(
            [ds.x.values.ravel(), ds.y.values.ravel(), ds.z.values.ravel()], axis=-1
        ).astype(np.float32)

        # 3. Stack qualities into (N, 5) float32
        # Order must match your Rust Quality struct: rho, vp, vs, qp, qs
        qualities = np.stack(
            [
                ds.rho.values.ravel(),
                ds.vp.values.ravel(),
                ds.vs.values.ravel(),
                ds.qp.values.ravel(),
                ds.qs.values.ravel(),
            ],
            axis=-1,
        ).astype(np.float32)

        # 4. Call the Rust mesh method
        # We use a temporary builder to generate the PyModel
        builder = ModelBuilder()
        internal_model = builder.mesh(vertices, qualities, padded_shape)

        return cls(internal_model)

    @classmethod
    def from_mesh(cls, path: Path | str):
        raw = ModelBuilder().load_mesh(str(path))
        return cls(raw)

    @classmethod
    def from_layers(cls, directory: Path | str):
        raw = ModelBuilder().load_layers_from_dir(str(directory))
        return Model(raw)

    def __add__(self, other):
        """Allows stacking using the '+' operator: model = top + bottom"""
        if not isinstance(other, Model):
            raise TypeError("Can only stack with another Model")
        stacked_raw = self._builder.stack(self._raw, other._raw)
        return Model(stacked_raw)

    def query(self, x, y, z):
        quality_rs, dist = self._raw.query(x, y, z)
        quality = Quality(
            rho=quality_rs.rho,
            vp=quality_rs.vp,
            vs=quality_rs.vs,
            qp=quality_rs.qp,
            qs=quality_rs.qs,
        )
        return quality, dist

    def query_many(self, x, y, z) -> xr.Dataset:
        original_shape = x.shape
        ndim = x.ndim

        dims = tuple(f"d{i}" for i in range(ndim))

        quality_array = self._raw.query_many(
            x.ravel().astype(np.float32),
            y.ravel().astype(np.float32),
            z.ravel().astype(np.float32),
        )

        var_names = ["rho", "vp", "vs", "qp", "qs", "dist"]
        data_vars = {}
        for i, name in enumerate(var_names):
            data_vars[name] = (dims, quality_array[:, i].reshape(original_shape))

        dset = xr.Dataset(
            data_vars=data_vars,
            coords={
                "x": (dims, x),
                "y": (dims, y),
                "z": (dims, z),
            },
        )

        dset = dset.set_xindex(
            ["x", "y", "z"],
            xr.indexes.NDPointIndex,
            tree_adapter_cls=xoak.SklearnBallTreeAdapter,
        )

        return dset

    def inspect(self):
        self._raw.print_structure()

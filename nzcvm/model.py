from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr
import xoak


from nzcvm import nzcvm, mesh

from .nzcvm import PyModel


@dataclass
class Quality:
    rho: float
    vp: float
    vs: float
    qp: float
    qs: float
    alpha: float


class Model:
    """A high-level wrapper for the Rust ModelTree."""

    def __init__(self, internal_py_model: PyModel):
        self._raw = internal_py_model

    @classmethod
    def from_mesh(cls, mesh_model: mesh.Mesh):
        # Determine the model type from the available data
        # Point data => tomography model
        # Cell data => Basin model
        # TODO: Map this explicitly to field data in the vtkhdf

        types = mesh_model.cell_data["model_type"]
        model_idx = mesh_model.cell_data["models"]
        rho = mesh_model.field_data["rho"]
        vp = mesh_model.field_data["vp"]
        vs = mesh_model.field_data["vs"]
        qp = mesh_model.field_data["qp"]
        qs = mesh_model.field_data["qs"]
        alpha = mesh_model.field_data["alpha"]

        qualities = np.c_[rho, vp, vs, qp, qs, alpha]
        raw = nzcvm.mesh_model(
            mesh_model.points.astype(np.float32),
            mesh_model.connectivity.astype(np.uint64),
            types,
            model_idx,
            qualities.astype(np.float32),
            mesh_model.cell_data["priority"],
        )
        return cls(raw)

    def query(self, x, y, z):
        quality_rs = self._raw.query(x, y, z)
        quality = Quality(
            rho=quality_rs.rho,
            vp=quality_rs.vp,
            vs=quality_rs.vs,
            qp=quality_rs.qp,
            qs=quality_rs.qs,
        )
        return quality

    def query_many(self, x, y, z) -> xr.Dataset:
        x, y, z = np.broadcast_arrays(x, y, z)

        original_shape = x.shape
        ndim = x.ndim
        dims = tuple(f"d{i}" for i in range(ndim))

        quality_array = self._raw.query_many(
            x.astype(np.float32, copy=False).ravel(),
            y.astype(np.float32, copy=False).ravel(),
            z.astype(np.float32, copy=False).ravel(),
        )

        var_names = ["rho", "vp", "vs", "qp", "qs"]
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

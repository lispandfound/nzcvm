from dataclasses import dataclass
from pathlib import Path
from printree import ftree

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

    @classmethod
    def _from_rs_quality(cls, quality_rs: Any) -> Self:
        return cls(
            rho=quality_rs.rho,
            vp=quality_rs.vp,
            vs=quality_rs.vs,
            qp=quality_rs.qp,
            qs=quality_rs.qs,
            alpha=quality_rs.alpha
        )



@dataclass
class Point:
    x: float
    y: float
    z: float

    @classmethod
    def _from_rs_point(cls, point_rs: Any) -> Self:
        return cls(point_rs.x, point_rs.y, point_rs.z)

    def __str__(self) -> str:
        return f'({self.x:.3g}, {self.y:.3g}, {self.z:.3g})'

@dataclass
class Simplex:
    c0: Point
    c1: Point
    c2: Point
    c3: Point
    priority: int

    @classmethod
    def _from_simplex_method(cls, simplex_rs: Any) -> Self:
        return cls(
            c0=Point.simplex_rs._from_rs_point(c0),
            c1=Point.simplex_rs._from_rs_point(c1),
            c2=Point.simplex_rs._from_rs_point(c2),
            c3=Point.simplex_rs._from_rs_point(c3),
            priority=simplex_rs.priority
        )

    def __str__(self) -> str:
        c0 = str(self.c0)
        c1 = str(self.c1)
        c2 = str(self.c2)
        c3 = str(self.c3)
        return f'Tetrahedron with corners c0={c0}, c1={c1}, c2={c2}, c3={c3}'

        

@dataclass
class ConstantModel:
    quality: Quality

    def __str__(self) -> str:
        return f'Constant model with quality = {str(quality)}'

@dataclass
class InterpolatedModel:
    x: Quality
    y: Quality
    z: Quality
    w: Quality

    def __str__(self) -> str: 
        return (f'Barycentric interpolation between:\n'
               f'x={str(self.x)} y={str(self.y)} z={str(self.z)} w={str(self.w)}') 

SimplexModel = ConstantModel | InterpolatedModel

def _from_rs_simplex_model(model: Any) -> SimplexModel:
    if hasinstance(model, 'quality'):
        return ConstantModel(quality=model.quality)
    else:
        return InterpolatedModel(
            x=model.x,
            y=model.y,
            z=model.z,
            w=model.w
        )

@dataclass
class Explanation:
    simplices: list[Simplex]
    qualities: list[Quality]
    models: list[SimplexModel]
    output: Quality | None

    @classmethod
    def _from_rs_explanation(cls, explanation: Any) -> Self:
        return cls(
            simplices=[Simplex._from_rs_simplex(simplex) for simplex in explanation.simplices],
            qualities=[Quality._from_rs_quality(quality) for quality in explanation.qualities],
            models=[_from_rs_simplex_model(model) for model in explanation.models],
            output=Quality._from_rs_quality(model.output)
        )

    def __str__(self) -> str:
        if not output:
            return 'No model coverage for query point.'
        tree = {
            str(output): {
                f'Simplex {i} (priority = {p})': {
                    'Model': str(model),
                    'Simplex quality': str(quality),
                    'Geometry': str(simplex) 
                }
            }
        }
        output = ftree(tree)
        if len(model.qualities) > 1:
            output += '\nQualities alpha-blended together until exhaustion or combined quality ~ 1.0'
        return output
                
    
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

    def query(self, x, y, z) -> Quality:
        quality_rs = self._raw.query(x, y, z)
        return Quality._from_rs_quality(quality_rs)

    def get_explanation(self, x, y, z) -> Explanation:
        explanation_rs = self._raw.explain(x, y, z)
        return Explanation._from_rs_explanation(explanation_rs)

    def explain(self, x: float, y: float, z: float) -> None:
        explanation = self.get_explanation(x, y, z)
        print(str(explanation))

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

"""Tests for the nzcvm.model Python API.

Only tests runtime behaviour that ``ty`` (static types) and ``cargo test``
(BVH correctness, priority ordering, alpha blending) cannot verify:
numpy array shapes/dtypes, ModelRange enum tuple values, and dask integration.
"""

import numpy as np
import pytest
import xarray as xr

from nzcvm import nzcvm as _nzcvm  # ty: ignore[unresolved-import]

from nzcvm.model import Model, ModelRange


def _make_model(rho: float = 2700.0, vs: float = 3500.0, priority: int = 0) -> Model:
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    q = np.array([[rho, 6000.0, vs, 200.0, 100.0, 1.0]], dtype=np.float32)
    mesh = _nzcvm.mesh_model(
        v,
        faces,
        np.array([0], dtype=np.uint8),
        np.array([0], dtype=np.uint64),
        q,
        np.uint8(priority),
        None,
    )
    return Model(_nzcvm.model_tree([mesh]))


class TestQueryManyRaw:
    """query_many_raw returns a float32 ndarray of shape (*x.shape, 6)."""

    def test_shape_and_dtype(self):
        x = np.array([0.1, 0.2], dtype=np.float32)
        z = np.zeros(2, dtype=np.float32)
        result = _make_model().query_many_raw(x, z, z)
        assert result.shape == (2, 6) and result.dtype == np.float32

    def test_nd_input(self):
        x = np.full((3, 4), 0.1, dtype=np.float32)
        assert _make_model().query_many_raw(x, x, x).shape == (3, 4, 6)

    def test_model_range_filters(self):
        x = np.array([0.1], dtype=np.float32)
        result = _make_model(vs=3500.0, priority=10).query_many_raw(
            x, x, x, model_range=ModelRange.BASINS
        )
        assert abs(float(result[0, 2]) - 3500.0) < 1.0


class TestQueryMany:
    """query_many returns a labelled xarray Dataset with a qualities DataArray."""

    def test_qualities_variable_and_coords(self):
        x = np.array([0.1, 0.2], dtype=np.float32)
        z = np.zeros(2, dtype=np.float32)
        ds = _make_model(rho=2700.0).query_many(x, z, z)
        assert "qualities" in ds
        assert "component" in ds.coords
        rho_vals = ds["qualities"].sel(component="rho").values
        assert np.allclose(rho_vals, 2700.0, rtol=1e-3)

    def test_component_coordinate_labels(self):
        x = np.array([0.1], dtype=np.float32)
        z = np.zeros(1, dtype=np.float32)
        ds = _make_model().query_many(x, z, z)
        assert list(ds.coords["component"].values) == [
            "rho",
            "vp",
            "vs",
            "qp",
            "qs",
            "alpha",
        ]


class TestDask:
    def test_query_many_raw_via_apply_ufunc(self):
        dask = pytest.importorskip("dask.array")
        model = _make_model()
        x = xr.DataArray(
            dask.from_array(np.array([0.1, 0.2], dtype=np.float32), chunks=1),
            dims=["d0"],
        )
        z = xr.DataArray(dask.zeros_like(x.data), dims=["d0"])
        result = xr.apply_ufunc(
            model.query_many_raw,
            x,
            z,
            z,
            input_core_dims=[[], [], []],
            output_core_dims=[["quality_dim"]],
            dask="parallelized",
            output_dtypes=[np.float32],
            dask_gufunc_kwargs={"output_sizes": {"quality_dim": 6}},
        ).compute()
        assert result.shape == (2, 6)

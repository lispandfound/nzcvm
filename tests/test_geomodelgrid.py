"""Tests for GeoModelGrid and dask laziness of empty_block / empty_surface."""
import numpy as np
import pytest
import dask.array as da
import xarray as xr

from nzcvm.geomodelgrid import (
    Block,
    Surface as GeoSurface,
    empty_block,
    empty_surface,
)
from nzcvm.coordinates import Coordinate


class TestEmptyBlockDaskLaziness:
    def _make_block(self, ni=10, nj=10, nk=5):
        return Block(
            resolution_horiz=100.0,
            resolution_vert=50.0,
            z_top=0.0,
            shape={Coordinate.I: ni, Coordinate.J: nj, Coordinate.K: nk},
            name="test",
        )

    def test_returns_xarray_dataset(self):
        ds = empty_block(self._make_block())
        assert isinstance(ds, xr.Dataset)

    def test_x_y_z_present(self):
        ds = empty_block(self._make_block())
        assert Coordinate.X in ds
        assert Coordinate.Y in ds
        assert Coordinate.Z in ds

    def test_x_is_dask_backed(self):
        ds = empty_block(self._make_block())
        x_data = ds[Coordinate.X].data
        assert isinstance(x_data, da.Array), "x should be dask-backed (lazy)"

    def test_shape_matches_block(self):
        ni, nj, nk = 8, 6, 4
        ds = empty_block(self._make_block(ni=ni, nj=nj, nk=nk))
        assert ds[Coordinate.X].shape == (ni, nj, nk)

    def test_x_coordinates_use_resolution(self):
        block = self._make_block(ni=5, nj=3, nk=2)
        ds = empty_block(block)
        x_vals = ds[Coordinate.X].values
        expected_x0 = np.arange(5) * 100.0
        assert x_vals[:, 0, 0] == pytest.approx(expected_x0, rel=1e-5)

    def test_z_top_offset(self):
        block = Block(
            resolution_horiz=100.0,
            resolution_vert=50.0,
            z_top=1000.0,
            shape={Coordinate.I: 2, Coordinate.J: 2, Coordinate.K: 3},
            name="test",
        )
        ds = empty_block(block)
        z_vals = ds[Coordinate.Z].values
        expected_z = np.array([1000.0, 1050.0, 1100.0])
        assert z_vals[0, 0, :] == pytest.approx(expected_z, rel=1e-5)

    def test_chunks_capped_at_dim_size(self):
        block = Block(
            resolution_horiz=1.0,
            resolution_vert=1.0,
            z_top=0.0,
            shape={Coordinate.I: 2, Coordinate.J: 2, Coordinate.K: 2},
            name="tiny",
        )
        ds = empty_block(block)
        x = ds[Coordinate.X].data
        assert x.chunks[0][0] <= 2
        assert x.chunks[1][0] <= 2
        assert x.chunks[2][0] <= 2


class TestEmptySurface:
    def test_returns_xarray_dataset(self):
        surf = GeoSurface(shape=(10, 8), resolution_horiz=200.0, name="topo")
        ds = empty_surface(surf)
        assert isinstance(ds, xr.Dataset)

    def test_coord_sizes(self):
        surf = GeoSurface(shape=(10, 8), resolution_horiz=200.0, name="topo")
        ds = empty_surface(surf)
        assert ds.sizes[Coordinate.I] == 10
        assert ds.sizes[Coordinate.J] == 8

    def test_resolution_in_coords(self):
        surf = GeoSurface(shape=(5, 3), resolution_horiz=500.0, name="topo")
        ds = empty_surface(surf)
        i_vals = ds.coords[Coordinate.I].values
        assert float(i_vals[1] - i_vals[0]) == pytest.approx(500.0, rel=1e-5)

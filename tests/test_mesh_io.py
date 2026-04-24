"""Tests for the nzcvm.mesh.Mesh class (VTK-HDF I/O and union)."""
import numpy as np
import pytest

from nzcvm.mesh import Mesh, CellType


def _make_tetra_mesh(rho=2700.0, priority=0):
    """Build a minimal single-tetrahedron Mesh with all required fields."""
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.uint64)
    cell_type = CellType.TETRA
    cell_data = {
        "model_type": np.array([0], dtype=np.uint8),
        "models": np.array([0], dtype=np.uint64),
    }
    field_data = {
        "rho": np.array([rho], dtype=np.float32),
        "vp": np.array([6000.0], dtype=np.float32),
        "vs": np.array([3500.0], dtype=np.float32),
        "qp": np.array([200.0], dtype=np.float32),
        "qs": np.array([100.0], dtype=np.float32),
        "alpha": np.array([1.0], dtype=np.float32),
        "priority": np.array([priority], dtype=np.uint8),
    }
    return Mesh(
        points=points,
        connectivity=connectivity,
        cell_type=cell_type,
        cell_data=cell_data,
        field_data=field_data,
    )


class TestMeshCreation:
    def test_create_minimal_mesh(self):
        mesh = _make_tetra_mesh()
        assert len(mesh.points) == 4
        assert len(mesh.connectivity) == 1
        assert mesh.cell_type == CellType.TETRA

    def test_field_data_present(self):
        mesh = _make_tetra_mesh(rho=1234.0)
        assert "rho" in mesh.field_data
        np.testing.assert_allclose(mesh.field_data["rho"], [1234.0])


class TestMeshVtkHdfRoundtrip:
    def test_write_and_read_roundtrip(self, tmp_path):
        mesh = _make_tetra_mesh(rho=3000.0)
        path = tmp_path / "test.vtkhdf"
        mesh.write_vtkhdf(path)
        loaded = Mesh.read_vtkhdf(path)

        np.testing.assert_allclose(loaded.points, mesh.points, rtol=1e-5)
        np.testing.assert_array_equal(loaded.connectivity, mesh.connectivity)
        np.testing.assert_allclose(
            loaded.field_data["rho"], mesh.field_data["rho"], rtol=1e-5
        )

    def test_read_bad_file_raises(self, tmp_path):
        import h5py
        path = tmp_path / "bad.vtkhdf"
        with h5py.File(path, "w") as f:
            f.create_group("not_vtkhdf")
        with pytest.raises(ValueError, match="not a VTKHDF file"):
            Mesh.read_vtkhdf(path)

    def test_priority_roundtrip(self, tmp_path):
        mesh = _make_tetra_mesh(priority=3)
        path = tmp_path / "priority.vtkhdf"
        mesh.write_vtkhdf(path)
        loaded = Mesh.read_vtkhdf(path)
        assert int(loaded.field_data["priority"][0]) == 3


class TestMeshUnion:
    def test_union_single_mesh_returns_same(self):
        mesh = _make_tetra_mesh()
        result = Mesh.union(mesh)
        assert result is mesh

    def test_union_two_meshes_doubles_points(self):
        m1 = _make_tetra_mesh(rho=1000.0)
        m2 = _make_tetra_mesh(rho=2000.0)
        result = Mesh.union(m1, m2)
        assert len(result.points) == 8
        assert len(result.connectivity) == 2

    def test_union_offsets_connectivity(self):
        m1 = _make_tetra_mesh()
        m2 = _make_tetra_mesh()
        result = Mesh.union(m1, m2)
        # Second cell should reference points 4-7
        assert result.connectivity[1].min() == 4

    def test_union_empty_raises(self):
        with pytest.raises(ValueError):
            Mesh.union()

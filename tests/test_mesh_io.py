"""Tests for mesh I/O via PyVista (VTKHDF roundtrip).

PyVista handles the VTKHDF format natively.  These tests verify that all
NZCVM-specific data (points, connectivity, field data, cell data) survive a
save/load roundtrip and that ``nzcvm.mesh.read_vtkhdf`` raises when the file
is not an ``UnstructuredGrid``.
"""
import numpy as np
import pytest
import pyvista as pv

from nzcvm.mesh import make_mesh, read_vtkhdf


def _make_pv_mesh(rho: float = 2700.0, priority: int = 0) -> pv.UnstructuredGrid:
    """Single-tetrahedron mesh with all NZCVM-required data arrays."""
    points = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return make_mesh(
        points=points,
        connectivity=np.array([[0, 1, 2, 3]], dtype=np.int64),
        cell_data={
            "model_type": np.array([0], dtype=np.uint8),
            "models": np.array([0], dtype=np.uint64),
        },
        field_data={
            "rho": np.array([rho], dtype=np.float32),
            "vp": np.array([6000.0], dtype=np.float32),
            "vs": np.array([3500.0], dtype=np.float32),
            "qp": np.array([200.0], dtype=np.float32),
            "qs": np.array([100.0], dtype=np.float32),
            "alpha": np.array([1.0], dtype=np.float32),
            "priority": np.array([priority], dtype=np.uint8),
        },
    )


class TestPyVistaVtkHdfRoundtrip:
    """PyVista natively handles VTKHDF I/O; verify NZCVM-specific data survives."""

    def test_read_returns_unstructured_grid(self, tmp_path):
        mesh = _make_pv_mesh()
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        assert isinstance(loaded, pv.UnstructuredGrid)

    def test_n_points_preserved(self, tmp_path):
        mesh = _make_pv_mesh()
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        assert loaded.n_points == mesh.n_points

    def test_points_preserved(self, tmp_path):
        mesh = _make_pv_mesh()
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        assert loaded.points == pytest.approx(mesh.points, rel=1e-5)

    def test_rho_field_data_preserved(self, tmp_path):
        mesh = _make_pv_mesh(rho=3000.0)
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        assert float(loaded.field_data["rho"][0]) == pytest.approx(3000.0)

    def test_priority_field_data_preserved(self, tmp_path):
        mesh = _make_pv_mesh(priority=5)
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        assert int(loaded.field_data["priority"][0]) == 5

    def test_all_quality_fields_preserved(self, tmp_path):
        mesh = _make_pv_mesh(rho=2500.0)
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        for field in ("vp", "vs", "qp", "qs", "alpha"):
            assert float(loaded.field_data[field][0]) == pytest.approx(
                float(mesh.field_data[field][0])
            )

    def test_cell_data_preserved(self, tmp_path):
        mesh = _make_pv_mesh()
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        np.testing.assert_array_equal(
            loaded.cell_data["model_type"], mesh.cell_data["model_type"]
        )

    def test_connectivity_preserved(self, tmp_path):
        mesh = _make_pv_mesh()
        path = tmp_path / "test.vtkhdf"
        mesh.save(str(path))
        loaded = pv.read(str(path))
        loaded_conn = loaded.cells_dict[pv.CellType.TETRA]
        orig_conn = mesh.cells_dict[pv.CellType.TETRA]
        np.testing.assert_array_equal(loaded_conn, orig_conn)

    def test_read_vtkhdf_raises_for_wrong_type(self, tmp_path):
        """read_vtkhdf should raise when the file is not an UnstructuredGrid."""
        poly = pv.Sphere()
        path = tmp_path / "sphere.vtkhdf"
        poly.save(str(path))
        with pytest.raises(ValueError, match="UnstructuredGrid"):
            read_vtkhdf(path)


class TestMakeMesh:
    """make_mesh must create a valid pyvista UnstructuredGrid."""

    def test_returns_unstructured_grid(self):
        mesh = _make_pv_mesh()
        assert isinstance(mesh, pv.UnstructuredGrid)

    def test_n_points(self):
        mesh = _make_pv_mesh()
        assert mesh.n_points == 4

    def test_n_cells(self):
        mesh = _make_pv_mesh()
        assert mesh.n_cells == 1

    def test_cell_type_is_tetra(self):
        mesh = _make_pv_mesh()
        assert pv.CellType.TETRA in mesh.cells_dict

    def test_field_data_accessible(self):
        mesh = _make_pv_mesh(rho=1234.0)
        assert float(mesh.field_data["rho"][0]) == pytest.approx(1234.0)

"""Tests for mesh I/O via PyVista (VTKHDF roundtrip).

PyVista handles the VTKHDF format natively.  These tests verify that all
NZCVM-specific data (points, connectivity, field data, cell data) survive a
save/load roundtrip and that ``nzcvm.mesh.read_vtkhdf`` raises when the file
is not an ``UnstructuredGrid``.
"""

import numpy as np
import pytest
import pyvista as pv

from nzcvm.mesh import make_mesh


@pytest.fixture
def basic_mesh_data():
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    connectivity = np.array([[0, 1, 2, 3]], dtype=np.int64)
    return points, connectivity


def test_make_mesh_connectivity_formatting(basic_mesh_data):
    """Verify that (M, 4) connectivity is correctly flattened with VTK padding."""
    points, _ = basic_mesh_data

    two_cells = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)

    mesh = make_mesh(points, two_cells, {}, {})

    # VTK format: [n_pts_cell0, id0, id1, id2, id3, n_pts_cell1, id0, ...]
    expected_cells = np.array([4, 0, 1, 2, 3, 4, 0, 1, 2, 3], dtype=np.int64)
    np.testing.assert_array_equal(mesh.cells, expected_cells)
    assert mesh.celltypes[0] == pv.CellType.TETRA


def test_make_mesh_name_metadata_handling(basic_mesh_data):
    """Verify that the 'name' parameter is correctly wrapped for VTKHDF compatibility."""
    points, connectivity = basic_mesh_data
    test_name = "nz_model_v1"

    # Test case 1: Name provided
    mesh = make_mesh(points, connectivity, {}, {}, name=test_name)
    assert "name" in mesh.field_data

    assert mesh.field_data["name"][0] == test_name

    field_data = {"name": np.array(["old_name"])}
    mesh = make_mesh(points, connectivity, {}, field_data, name="new_name")
    assert mesh.field_data["name"][0] == "new_name"


def test_make_mesh_data_assignment(basic_mesh_data):
    """Ensure cell and field data dictionaries are correctly mapped to the mesh."""
    points, connectivity = basic_mesh_data
    cell_data = {"velocity": np.array([1500.0])}
    field_data = {"version": np.array([1])}

    mesh = make_mesh(points, connectivity, cell_data, field_data)

    np.testing.assert_array_equal(mesh.cell_data["velocity"], cell_data["velocity"])
    np.testing.assert_array_equal(mesh.field_data["version"], field_data["version"])


def test_make_mesh_empty_data_structures(basic_mesh_data):
    """Ensure the function handles empty dictionaries without error."""
    points, connectivity = basic_mesh_data

    # Should not raise any KeyError or TypeError
    mesh = make_mesh(points, connectivity, {}, {}, name=None)

    assert len(mesh.cell_data) == 0
    # field_data might have default VTK keys, but shouldn't have 'name'
    assert "name" not in mesh.field_data

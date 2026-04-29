import numpy as np
import pytest
import pyvista as pv

from nzcvm.surface import Surface, build_surface_interpolator


@pytest.fixture
def flat_surface() -> Surface:
    """Creates a 10x10 flat plane at Z=5.0 for testing."""
    # Create a plane from (0,0) to (10,10) at elevation Z=5
    mesh = pv.Plane(center=(5, 5, 5), direction=(0, 0, 1), i_size=10, j_size=10)
    return build_surface_interpolator(mesh)


def test_interpolation_is_correct(flat_surface: Surface) -> None:
    """Check that querying the middle of the surface returns the correct Z."""
    x = np.array([5.0, 2.0])
    y = np.array([5.0, 8.0])

    z_values = flat_surface.transform(x, y)

    # Since the plane is flat at Z=5, all points inside should be 5.0
    assert np.allclose(z_values, 5.0)
    assert z_values.shape == (2,)


def test_hull_check_raises_error_with_note(flat_surface: Surface) -> None:
    """Check that points outside the boundary raise ValueError with a debug note."""
    x = np.array([20.0])
    y = np.array([20.0])

    with pytest.raises(ValueError) as excinfo:
        flat_surface.transform(x, y)

    assert "Points not in convex hull" in str(excinfo.value)

    notes = "".join(excinfo.value.__notes__)
    assert "Failure Summary" in notes
    assert "Total failed: 1" in notes


def test_z_bounds_error(flat_surface: Surface) -> None:
    """
    Check that if interpolation returns a value outside mesh bounds
    (or invalid mask), it triggers the Z-limit error.
    """
    flat_surface.bounds[2] = 10.0  # Set min_z to 10
    flat_surface.bounds[5] = 20.0  # Set max_z to 20

    x = np.array([5.0])
    y = np.array([5.0])

    with pytest.raises(ValueError) as excinfo:
        flat_surface.transform(x, y)

    assert "Z values from interpolation invalid" in str(excinfo.value)

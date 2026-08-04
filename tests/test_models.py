from pathlib import Path

import pytest

from nzcvm.models.model import MeshModel


@pytest.mark.real_data
def test_model_loading(model_path: Path) -> None:
    m = MeshModel.from_path(model_path)
    assert isinstance(m.priority, int)
    aabb_min, aabb_max = m.aabb
    assert aabb_min.shape == (3,) and aabb_max.shape == (3,)
    assert all(aabb_min[i] <= aabb_max[i] for i in range(3))

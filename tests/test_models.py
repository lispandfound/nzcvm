from pathlib import Path

from nzcvm.models.model import MeshModel


def test_model_loading(model_path: Path) -> None:
    _ = MeshModel.from_path(model_path)

"""Backward-compatibility shim – use :mod:`nzcvm.models.mesh` instead."""

from nzcvm.models.mesh import *  # noqa: F401, F403
from nzcvm.models.mesh import (  # noqa: F401
    StructuredMesh,
    TetrahedralMesh,
    VTK_TETRA,
    make_mesh,
    read_structured_vtkhdf,
    read_unstructured_vtkhdf,
    write_structured_vtkhdf,
    write_unstructured_vtkhdf,
)

"""Phase 2: E(3)-equivariant primitives implemented from scratch.

Nothing in this subpackage calls ``e3nn``.  It exists to demonstrate that the library's
machinery -- spherical harmonics, Clebsch-Gordan tensor products, Wigner-D matrices --
is understood from the representation theory up, and it is validated against ``e3nn``
in the test suite rather than borrowed from it.
"""

from .clebsch_gordan import clebsch_gordan, decomposition_degrees, wigner_3j
from .layer import EquivariantConv, ScratchTFN
from .spherical_harmonics import spherical_harmonics, spherical_harmonics_l
from .tensor_product import EquivariantLinear, Gate, TensorProduct
from .wigner import random_rotation, rotation_matrix, wigner_D, wigner_D_blocks

__all__ = [
    "clebsch_gordan",
    "decomposition_degrees",
    "wigner_3j",
    "spherical_harmonics",
    "spherical_harmonics_l",
    "random_rotation",
    "rotation_matrix",
    "wigner_D",
    "wigner_D_blocks",
    "EquivariantLinear",
    "TensorProduct",
    "Gate",
    "EquivariantConv",
    "ScratchTFN",
]

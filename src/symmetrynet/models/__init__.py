"""Models under comparison.

``InvariantGNN``
    Distance-only baseline.  Rotation invariant, angle blind.
``TensorFieldNetwork``
    E(3)-equivariant TFN built on ``e3nn``, using Clebsch-Gordan tensor products.
``PaiNN``
    E(3)-equivariant via scalar/vector algebra only -- no tensor products, no ``e3nn``.
    Restricted to ``l <= 1``, and much cheaper for it.
``NaiveCoordinateGNN`` / ``NaiveCoordinateMLP``
    Raw-coordinate controls.  Not rotation invariant -- the failure case.
``ScratchTFN``
    The from-scratch equivariant model (see :mod:`symmetrynet.scratch`).
"""

from ..scratch.layer import ScratchTFN
from .angular import AngularInvariantGNN
from .baseline import InvariantGNN
from .forces import ForceModel
from .naive import NaiveCoordinateGNN, NaiveCoordinateMLP
from .painn import PaiNN
from .tfn import TensorFieldNetwork, hidden_irreps

MODEL_REGISTRY = {
    "baseline": InvariantGNN,
    "angular": AngularInvariantGNN,
    "tfn": TensorFieldNetwork,
    "painn": PaiNN,
    "naive": NaiveCoordinateGNN,
    "naive_mlp": NaiveCoordinateMLP,
    "scratch": ScratchTFN,
}

__all__ = [
    "InvariantGNN",
    "AngularInvariantGNN",
    "TensorFieldNetwork",
    "PaiNN",
    "NaiveCoordinateGNN",
    "NaiveCoordinateMLP",
    "ScratchTFN",
    "ForceModel",
    "hidden_irreps",
    "MODEL_REGISTRY",
]

"""Models under comparison.

``InvariantGNN``
    Distance-only baseline.  Rotation invariant, angle blind.
``TensorFieldNetwork``
    E(3)-equivariant TFN built on ``e3nn``.  The proposed model.
``NaiveCoordinateGNN`` / ``NaiveCoordinateMLP``
    Raw-coordinate controls.  Not rotation invariant -- the failure case.
``ScratchTFN``
    The from-scratch equivariant model (see :mod:`symmetrynet.scratch`).
"""

from ..scratch.layer import ScratchTFN
from .baseline import InvariantGNN
from .naive import NaiveCoordinateGNN, NaiveCoordinateMLP
from .tfn import TensorFieldNetwork, hidden_irreps

MODEL_REGISTRY = {
    "baseline": InvariantGNN,
    "tfn": TensorFieldNetwork,
    "naive": NaiveCoordinateGNN,
    "naive_mlp": NaiveCoordinateMLP,
    "scratch": ScratchTFN,
}

__all__ = [
    "InvariantGNN",
    "TensorFieldNetwork",
    "NaiveCoordinateGNN",
    "NaiveCoordinateMLP",
    "ScratchTFN",
    "hidden_irreps",
    "MODEL_REGISTRY",
]

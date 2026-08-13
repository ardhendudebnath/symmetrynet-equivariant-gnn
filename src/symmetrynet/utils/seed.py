"""Determinism helpers.

Every headline number in this project is a comparison between two models, so runs need
to be reproducible for the comparison to mean anything.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch

__all__ = ["seed_everything"]


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    ``deterministic=True`` additionally forces deterministic CUDA kernels.  It is off by
    default because it slows training noticeably and the scatter-add used for message
    aggregation is non-deterministic on GPU regardless -- float addition is not
    associative, so run-to-run differences at the 1e-7 level are expected and harmless.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

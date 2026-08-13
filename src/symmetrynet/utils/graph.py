"""Graph construction utilities.

``torch_geometric.nn.radius_graph`` needs ``torch-cluster``, which has no reliable
Windows wheels and must otherwise be compiled from source.  QM9 molecules top out at 29
atoms, so a dense pairwise computation restricted to within-molecule pairs is both
simpler and fast enough -- and it keeps the project installable with plain ``pip``.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["radius_graph", "scatter_sum", "scatter_mean"]


def radius_graph(
    pos: Tensor,
    cutoff: float,
    batch: Tensor | None = None,
    *,
    max_num_neighbors: int | None = None,
    chunk_size: int = 4096,
) -> Tensor:
    """Edges between all pairs of nodes closer than ``cutoff``, excluding self-loops.

    Parameters
    ----------
    pos:
        ``(N, 3)`` node positions.
    cutoff:
        Radial cutoff.  Use ``float("inf")`` for a fully connected molecular graph.
    batch:
        ``(N,)`` graph assignment.  Edges never cross graphs.  ``None`` treats the
        input as a single graph.
    max_num_neighbors:
        Optional cap on the in-degree, keeping the nearest neighbours.  ``None`` means
        no cap, which is what you want for exact reproducibility on QM9.

    Returns
    -------
    ``(2, E)`` tensor of ``[source, destination]`` indices, so that a message flows
    ``edge_index[0] -> edge_index[1]``.
    """
    num_nodes = pos.shape[0]
    if batch is None:
        batch = pos.new_zeros(num_nodes, dtype=torch.long)

    srcs: list[Tensor] = []
    dsts: list[Tensor] = []
    for start in range(0, num_nodes, chunk_size):
        stop = min(start + chunk_size, num_nodes)
        block = pos[start:stop]
        dist = torch.cdist(block, pos)  # (chunk, N)

        same_graph = batch[start:stop, None] == batch[None, :]
        mask = (dist <= cutoff) & same_graph
        # Exclude self-loops by index, not by distance: two atoms are never truly
        # coincident in QM9, but masking on `dist > 0` would silently drop duplicates.
        rows = torch.arange(start, stop, device=pos.device)
        mask[rows - start, rows] = False

        if max_num_neighbors is not None:
            keep = torch.zeros_like(mask)
            masked_dist = dist.masked_fill(~mask, float("inf"))
            k = min(max_num_neighbors, num_nodes)
            idx = masked_dist.topk(k, dim=1, largest=False).indices
            keep.scatter_(1, idx, True)
            mask &= keep

        dst_local, src = mask.nonzero(as_tuple=True)
        dsts.append(dst_local + start)
        srcs.append(src)

    return torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0)


def scatter_sum(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Sum ``src`` into ``dim_size`` bins along dim 0 according to ``index``."""
    shape = (dim_size, *src.shape[1:])
    out = src.new_zeros(shape)
    idx = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
    return out.scatter_add_(0, idx, src)


def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Mean-pool ``src`` into ``dim_size`` bins along dim 0."""
    total = scatter_sum(src, index, dim_size)
    ones = src.new_ones(src.shape[0], *([1] * (src.dim() - 1)))
    count = scatter_sum(ones, index, dim_size).clamp_min(1.0)
    return total / count

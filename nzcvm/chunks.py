import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class ChunkingStrategy:
    chunks: tuple[int, ...]
    waste: float
    intent_loss: float


def optimise_chunks(
    model_dims: tuple[int, ...],
    user_chunks: tuple[int, ...],
    w_intent: float = 0.5,
) -> ChunkingStrategy:

    def waste(c):
        num_chunks = np.ceil(model_dims / c)
        allocated_voxels = np.prod(num_chunks * c)
        actual_voxels = np.prod(model_dims)
        waste = (allocated_voxels - actual_voxels) / allocated_voxels
        return waste

    def intent_loss(c):
        return np.mean(np.abs(c - user_chunks) / user_chunks)

    def objective(c):
        return waste(c) + (w_intent * intent_loss(c))

    bounds = [(1, d) for d in model_dims]

    res = minimize(
        objective, x0=user_chunks, bounds=bounds, method="SLSQP", options={"ftol": 1e-9}
    )

    optimized_chunks = np.round(res.x).astype(int)
    optimised_waste = waste(optimized_chunks)
    optimised_intent_loss = intent_loss(optimized_chunks)
    return ChunkingStrategy(
        tuple(optimized_chunks), optimised_waste, optimised_intent_loss
    )

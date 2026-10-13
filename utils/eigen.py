import torch
import numpy as np

def eigenvalue_computation_pmcl(modalities):
    # Preserve the reference SVD and its sign convention; only make precision
    # explicit so mixed-precision training does not feed Half to cuSOLVER.
    with torch.cuda.amp.autocast(enabled=False):
        stacked_reps = torch.stack(modalities, dim=-1).float()
        U_V, S_V, W_V = torch.linalg.svd(stacked_reps, full_matrices=True)
    return U_V, S_V

"""Minimal torch_sparse compatibility shim (for runtimes without an ARM64 wheel).

Use before importing the official GSAT package:
    import sys, torch_sparse_compat
    sys.modules["torch_sparse"] = torch_sparse_compat
"""
import torch

def transpose(index, value, m, n, coalesced=False):
 """COO (2,E) ： torch_sparse.transpose ， / ，
    if index is None:
        return None, value
    trans_idx = torch.stack([index[1], index[0]])
    return trans_idx, value

"""torch_sparse 兼容实现（macOS arm64 无 wheel，仅 GSAT 用到 transpose）。

用法：在 import 官方 gsat 前执行
    import sys, torch_sparse_compat
    sys.modules["torch_sparse"] = torch_sparse_compat
"""
import torch


def transpose(index, value, m, n, coalesced=False):
    """COO (2,E) 转置：交换行列（torch_sparse.transpose 的极简等价，不做合并/排序，
    调用方 reorder_like 会自行 sort_edge_index 对齐）。"""
    if index is None:
        return None, value
    trans_idx = torch.stack([index[1], index[0]])
    return trans_idx, value

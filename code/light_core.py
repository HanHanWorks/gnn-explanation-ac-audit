#!/usr/bin/env python3
"""Lightweight (numpy/pandapower only) re-implementation of the evaluation
pieces used in the RESS manuscript revision.

It mirrors ``run_audit.load_sample`` and
``run_supervised_edge_audit.edge_features`` exactly, but avoids importing
torch/torch-geometric so that the matched-sample checks can be reproduced in a
small environment.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import mat73
import networkx as nx
import numpy as np

from audit_core import audit, audit_baseline, build_net, derive_injections

DATA_ROOT = "datasets/PowerGraph/data_cascades"


@dataclass
class Mats:
    system: str
    blist: np.ndarray
    Ef: object
    exp: object
    of_bi: object
    Bf: object


def load_mats(system: str, root: str = ".") -> Mats:
    raw = os.path.join(root, DATA_ROOT, system, "raw")
    blist = np.asarray(mat73.loadmat(os.path.join(raw, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(raw, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(raw, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(raw, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(raw, "Bf.mat"))["B_f_tot"]
    return Mats(system, blist, Ef, exp, of_bi, Bf)


def truth_rows(m: Mats, i: int) -> set[int]:
    e = m.exp[i][0]
    if e is None:
        return set()
    return set(int(j) - 1 for j in np.asarray(e, dtype=int).ravel())


def positive_explainable(m: Mats) -> list[int]:
    out = []
    for i in range(len(m.of_bi)):
        y = int(np.asarray(m.of_bi[i][0]).ravel()[0])
        if y == 1 and m.exp[i][0] is not None:
            out.append(i)
    return out


def split_indices(m: Mats, seed: int, train_num: int, eval_start: int, num: int):
    """Exactly the split used by run_supervised_edge_audit.py (random split)."""
    pos = positive_explainable(m)
    random.seed(seed)
    random.shuffle(pos)
    train_idx = pos[: min(train_num, len(pos))]
    if eval_start < len(train_idx):
        raise ValueError("eval_start must be >= train_num")
    eval_idx = pos[eval_start : min(eval_start + num, len(pos))]
    if len(eval_idx) < num:
        raise ValueError(f"requested {num} eval samples, got {len(eval_idx)}")
    return train_idx, eval_idx, pos


def contingency_rows(m: Mats, i: int) -> list[int]:
    f = np.asarray(m.Ef[i][0], dtype=float).reshape(-1, 4)
    return [j for j in range(len(m.blist)) if np.all(np.abs(f[j]) < 1e-12)]


def sample_arrays(m: Mats, i: int):
    f = np.asarray(m.Ef[i][0], dtype=float).reshape(-1, 4)
    cont = set(contingency_rows(m, i))
    keep = [j for j in range(len(m.blist)) if j not in cont]
    x = np.asarray(m.Bf[i][0], dtype=float).reshape(-1, 3)
    src = m.blist[keep, 0] - 1
    dst = m.blist[keep, 1] - 1
    ea = f[keep]
    y = int(np.asarray(m.of_bi[i][0]).ravel()[0])
    return x, ea, src, dst, keep, cont, y, f


def edge_features(ea: np.ndarray, x: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [ea, x[src], x[dst], np.abs(x[src] - x[dst])], axis=1
    ).astype(float)


def bacc(keep_n: int, selected, truth: set[int], keep=None) -> float:
    if keep is None:
        raise ValueError("keep (physical row indices) is required")
    gt = np.asarray([1 if int(keep[j]) in truth else 0 for j in range(keep_n)], dtype=int)
    pred = np.zeros(keep_n, dtype=int)
    for j in selected:
        pred[int(j)] = 1
    tp = int(((pred == 1) & (gt == 1)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    return 0.5 * (tpr + tnr)


# --------------------------------------------------------------------------- #
# connectivity control
# --------------------------------------------------------------------------- #
def _topology_graph(net) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(int(x) for x in net.bus.index)
    for _, row in net.line.iterrows():
        if row["in_service"]:
            g.add_edge(int(row["from_bus"]), int(row["to_bus"]))
    for _, row in net.trafo.iterrows():
        if row["in_service"]:
            g.add_edge(int(row["hv_bus"]), int(row["lv_bus"]))
    for _, row in net.impedance.iterrows():
        if row["in_service"]:
            g.add_edge(int(row["from_bus"]), int(row["to_bus"]))
    return g


def is_connected_after(net, mapping, disconnect) -> bool:
    """Check bus-graph connectivity after removing the given elements.

    The slack bus (original ext_grid) is used as the root; every bus must be
    reachable.  This is the graph-level control used to separate islanding from
    voltage/non-convergence failures.
    """
    import copy

    n = copy.deepcopy(net)
    for kind, idx in disconnect:
        n[kind].loc[idx, "in_service"] = False
    g = _topology_graph(n)
    root = int(net.ext_grid["bus"].iloc[0])
    return nx.is_connected(g) and len(g.nodes) == len(net.bus) and root in g


def audit_connected_baseline(net, mapping, disconnect, volt_min=0.85, volt_max=1.10):
    ok, status, detail = audit_baseline(net, disconnect, volt_min=volt_min, volt_max=volt_max)
    return ok, status, detail


# --------------------------------------------------------------------------- #
# finite-budget wrapper, instrumented
# --------------------------------------------------------------------------- #
def select_feasible_instrumented(
    order, scores, mapping, keep, net, topk, strategy="local_swap", candidate_pool=8,
    ac_call_counter=None,
):
    """Mirror of run_pgexplainer_audit.select_feasible with AC-call counting."""
    if ac_call_counter is None:
        ac_call_counter = [0]

    def audit_candidate(candidate):
        ac_call_counter[0] += 1
        return audit_baseline(net, [mapping[keep[k]] for k in candidate])

    chosen = []
    for j in order:
        if len(chosen) >= topk:
            break
        candidate = chosen + [int(j)]
        ok, _, _ = audit_candidate(candidate)
        if ok:
            chosen = candidate
    if strategy == "greedy" or len(chosen) != topk:
        return chosen

    pool = [int(j) for j in order] if candidate_pool <= 0 else [
        int(j) for j in order[: min(candidate_pool, len(order))]
    ]
    current = list(chosen)
    current_score = float(sum(scores[j] for j in current))
    best, best_score = current, current_score
    for old in list(current):
        for j in pool:
            if j in current:
                continue
            candidate = [x for x in current if x != old] + [j]
            ok, _, _ = audit_candidate(candidate)
            score = float(sum(scores[k] for k in candidate))
            if ok and score > best_score + 1e-12:
                best, best_score = candidate, score
    return best

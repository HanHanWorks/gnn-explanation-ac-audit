#!/usr/bin/env python3
"""Unsupervised physical edge ranking under the unified AC audit.

The ranking uses only the observed graph state and static topology. Ground-truth
explanation rows are read only for reporting BACC/PFR anchors, never for ranking
or the feasible wrapper.
"""
import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mat73
import networkx as nx
import numpy as np
from sklearn.metrics import balanced_accuracy_score

from audit_core import audit_baseline, build_net, pfr_summary
from run_audit import load_sample
from run_pgexplainer_audit import select_feasible

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")

def edge_score(edge_attr, blist, keep_rows, method):
    """Return one score per current physical edge; larger means preferred."""
    f = np.asarray(edge_attr, dtype=float).reshape(-1, 4)
    p, q = f[:, 0], f[:, 1]
    rating = np.abs(f[:, 3])
    stress = np.hypot(p, q) / (rating + 0.05)
    graph = nx.Graph()
    graph.add_edges_from((int(blist[j, 0]) - 1, int(blist[j, 1]) - 1)
                         for j in keep_rows)
    bridges = set(tuple(sorted(e)) for e in nx.bridges(graph))
    redundancy = np.asarray([
        0.0 if tuple(sorted((int(blist[j, 0]) - 1, int(blist[j, 1]) - 1))) in bridges else 1.0
        for j in keep_rows
    ])
    if method == "low_stress":
        return -stress
    if method == "redundant_low_stress":
        return 10.0 * redundancy - stress
    raise ValueError(method)

def truth_set(exp, i):
    return set(int(j) - 1 for j in np.asarray(exp[i][0], dtype=int).ravel())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--eval_start", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--method", choices=["low_stress", "redundant_low_stress"], default="redundant_low_stress")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    raw = os.path.join(DATA_ROOT, args.dataset, "raw")
    blist = np.asarray(mat73.loadmat(os.path.join(raw, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(raw, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(raw, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(raw, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(raw, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)

    pos_idx = [i for i in range(len(of_bi)
               ) if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    eval_idx = pos_idx[args.eval_start:args.eval_start + args.num]
    if len(eval_idx) < args.num:
        raise ValueError(f"not enough evaluation samples: requested {args.num}, got {len(eval_idx)}")

    results = {"base": [], "physical": [], "physical_wrapped": [], "random_wrapped": [], "truth": []}
    bacc, wrapped_bacc, random_bacc = [], [], []
    for n, i in enumerate(eval_idx):
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        scores = edge_score(data.edge_attr[:len(keep)].numpy(), blist, keep, args.method)
        order = np.argsort(scores)[::-1]
        top = order[:min(args.topk, len(keep))]
        results["base"].append(audit_baseline(net, []))
        results["physical"].append(audit_baseline(net, [mapping[keep[j]] for j in top]))
        chosen = select_feasible(order, scores, mapping, keep, net, args.topk,
                                 strategy="local_swap", candidate_pool=8)
        results["physical_wrapped"].append(
            audit_baseline(net, [mapping[keep[j]] for j in chosen])
            if len(chosen) == args.topk else (False, "incomplete", {}))
        rng = random.Random(200000 * args.seed + n)
        ro = list(range(len(keep)))
        rng.shuffle(ro)
        rchosen = select_feasible(ro, np.zeros(len(keep)), mapping, keep, net, args.topk,
                                  strategy="greedy", candidate_pool=0)
        results["random_wrapped"].append(
            audit_baseline(net, [mapping[keep[j]] for j in rchosen])
            if len(rchosen) == args.topk else (False, "incomplete", {}))
        results["truth"].append(audit_baseline(net, [mapping[j] for j in truth_set(exp, i)]))

        gt = np.asarray([int(row in truth_set(exp, i)) for row in keep])
        for selected, target in ((top, bacc), (chosen, wrapped_bacc), (rchosen, random_bacc)):
            pred = np.zeros(len(keep), dtype=int)
            pred[selected] = 1
            target.append(float(balanced_accuracy_score(gt, pred)))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(eval_idx)} completed")

    summary = {k: pfr_summary(v) for k, v in results.items()}
    summary["_meta"] = {"system": args.dataset, "method": args.method,
        "protocol": "unified-eval-start-physical-unsupervised", "seed": args.seed,
        "eval_start": args.eval_start, "n": len(eval_idx), "topk": args.topk,
        "bacc_raw": float(np.mean(bacc)), "bacc_wrapped": float(np.mean(wrapped_bacc)),
        "random_wrapper_bacc": float(np.mean(random_bacc)),
        "ranking_uses_truth": False}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print(f"physical PFR={summary['physical']['PFR']:.3f}; wrapped={summary['physical_wrapped']['PFR']:.3f}; random={summary['random_wrapped']['PFR']:.3f}")
    print(f"BACC raw={np.mean(bacc):.3f}; wrapped={np.mean(wrapped_bacc):.3f}; random={np.mean(random_bacc):.3f}")
 print(f" : {args.output}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare AC and DC feasibility for the same fixed-slice interventions."""
import argparse
import copy
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mat73
import numpy as np
import pandapower as pp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from audit_core import audit_baseline, build_net, pfr_summary
from run_audit import load_sample
from run_supervised_edge_audit import edge_features, truth_rows

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")


def audit_dc(net, disconnect):
    n = copy.deepcopy(net)
    for kind, idx in disconnect:
        if kind == "line":
            n.line.loc[idx, "in_service"] = False
        elif kind == "trafo":
            n.trafo.loc[idx, "in_service"] = False
        else:
            n.impedance.loc[idx, "in_service"] = False
    try:
        pp.rundcpp(n, numba=False)
    except Exception:
        return False, "non_converged", {}
    if not getattr(n, "converged", False):
        return False, "non_converged", {}
    if len(n.res_bus) == 0 or np.isnan(n.res_bus.vm_pu.values).any():
        return False, "island", {}
    return True, "feasible", {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--train_num", type=int, default=20)
    ap.add_argument("--eval_start", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    raw = os.path.join(DATA_ROOT, args.dataset, "raw")
    blist = np.asarray(mat73.loadmat(os.path.join(raw, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(raw, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(raw, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(raw, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(raw, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)
    pos = [i for i in range(len(of_bi)) if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos)
    train_idx = pos[:args.train_num]
    eval_idx = pos[args.eval_start:args.eval_start + args.num]
    if len(eval_idx) < args.num:
        raise ValueError(f"not enough eval samples: requested {args.num}, got {len(eval_idx)}")

    xs, ys = [], []
    for i in train_idx:
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        xs.append(edge_features(data, keep)); truth = truth_rows(exp, i)
        ys.append(np.asarray([int(row in truth) for row in keep]))
    scorer = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed, solver="liblinear")
    scorer.fit(np.concatenate(xs), np.concatenate(ys))

    results = {"v7r_ac": [], "v7r_dc": [], "random_ac": [], "random_dc": [], "truth_ac": [], "truth_dc": []}
    bacc, random_bacc = [], []
    for n, i in enumerate(eval_idx):
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        scores = scorer.predict_proba(edge_features(data, keep))[:, 1]
        order = np.argsort(scores)[::-1][:min(args.topk, len(keep))]
        rng = random.Random(100000 * args.seed + n)
        ro = list(range(len(mapping))); rng.shuffle(ro); ro = ro[:min(args.topk, len(mapping))]
        truth = [mapping[j] for j in truth_rows(exp, i)]
        v7r = [mapping[keep[j]] for j in order]
        rnd = [mapping[j] for j in ro]
        results["v7r_ac"].append(audit_baseline(net, v7r))
        results["v7r_dc"].append(audit_dc(net, v7r))
        results["random_ac"].append(audit_baseline(net, rnd))
        results["random_dc"].append(audit_dc(net, rnd))
        results["truth_ac"].append(audit_baseline(net, truth))
        results["truth_dc"].append(audit_dc(net, truth))
        gt = np.asarray([int(row in set(truth_rows(exp, i))) for row in keep])
        for selected, target in ((order, bacc), (np.asarray(ro[:0]), random_bacc)):
            if target is random_bacc:
                # random edges are global; report quality only for comparable current edges.
                continue
            pred = np.zeros(len(keep), dtype=int); pred[selected] = 1
            target.append(float(balanced_accuracy_score(gt, pred)))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(eval_idx)} completed")

    summary = {k: pfr_summary(v) for k, v in results.items()}
    summary["_meta"] = {"system": args.dataset, "method": "supervised_edge_logistic",
        "protocol": "fixed-slice-public-ac-dc", "seed": args.seed, "n": len(eval_idx),
        "train_num": len(train_idx), "eval_start": args.eval_start, "topk": args.topk,
        "ranking_uses_eval_truth": False, "v7r_bacc": float(np.mean(bacc))}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print({k: summary[k]["PFR"] for k in results})
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()

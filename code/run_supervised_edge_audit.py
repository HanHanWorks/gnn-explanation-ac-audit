#!/usr/bin/env python3
"""Train/evaluate a leakage-free supervised edge scorer under the v3r AC audit.

The scorer uses only training-split edge/node features and truth edge labels.
Evaluation truth labels are used only for reporting balanced accuracy, never for
ranking or physical selection.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mat73
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from audit_core import build_net, audit_baseline, pfr_summary
from run_audit import load_sample
from run_pgexplainer_audit import select_feasible

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")


def truth_rows(exp, i):
    return set(int(j) - 1 for j in np.asarray(exp[i][0], dtype=int).ravel())


def edge_features(data, keep_rows):
    half = len(keep_rows)
    src = data.edge_index[0, :half].cpu().numpy()
    dst = data.edge_index[1, :half].cpu().numpy()
    x = data.x.cpu().numpy()
    ea = data.edge_attr[:half].cpu().numpy()
    return np.concatenate([ea, x[src], x[dst], np.abs(x[src] - x[dst])], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--train_num", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--max_graphs", type=int, default=0,
                    help="Restrict candidate graph indices; 0 means all graphs")
    ap.add_argument("--eval_start", type=int, default=-1,
                    help="Fixed evaluation offset; -1 uses train_num")
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--shuffle_train_labels", action="store_true",
                    help="Shuffle only training edge labels for a negative control")
    ap.add_argument("--split_mode", choices=["random", "contingency"], default="random",
                    help="random sample split or disjoint initial-contingency scenario split")
    ap.add_argument("--wrapper_strategy", choices=["greedy", "local_swap"], default="local_swap")
    ap.add_argument("--candidate_pool", type=int, default=8)
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

    upper = min(len(of_bi), args.max_graphs) if args.max_graphs > 0 else len(of_bi)
    pos_idx = [i for i in range(upper)
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    eval_start = args.train_num if args.eval_start < 0 else args.eval_start
    if args.split_mode == "random":
        random.shuffle(pos_idx)
        train_idx = pos_idx[:min(args.train_num, len(pos_idx))]
        if eval_start < len(train_idx):
            raise ValueError("eval_start must be >= train_num to prevent train/eval overlap")
        eval_idx = pos_idx[eval_start:min(eval_start + args.num, len(pos_idx))]
        train_scenarios = eval_scenarios = None
    else:
        # Group by initial contingency signature (zero-flow physical edges).
        # Entire groups stay on one side of the split; operating points from a
        # contingency never appear in both train and evaluation.
        scenario_of = {}
        for i in range(upper):
            f = np.asarray(Ef[i][0], dtype=float).reshape(-1, 4)
            scenario_of[i] = tuple(np.flatnonzero(np.all(np.abs(f) < 1e-12, axis=1)).tolist())
        groups = {}
        for i in pos_idx:
            groups.setdefault(scenario_of[i], []).append(i)
        group_keys = list(groups)
        random.Random(args.seed).shuffle(group_keys)
        train_keys, eval_keys = [], []
        train_count = eval_count = 0
        for key in group_keys:
            if train_count < args.train_num:
                train_keys.append(key); train_count += len(groups[key])
            elif eval_count < args.num:
                eval_keys.append(key); eval_count += len(groups[key])
            if train_count >= args.train_num and eval_count >= args.num:
                break
        train_idx = [i for key in train_keys for i in groups[key]]
        eval_idx = [i for key in eval_keys for i in groups[key]][:args.num]
        train_scenarios = len(train_keys); eval_scenarios = len(eval_keys)
    if len(eval_idx) < args.num:
        raise ValueError(f"not enough disjoint eval samples: requested {args.num}, got {len(eval_idx)}")

    train_x, train_y = [], []
    for i in train_idx:
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        train_x.append(edge_features(data, keep))
        truth = truth_rows(exp, i)
        train_y.append(np.asarray([int(row in truth) for row in keep]))
    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    if args.shuffle_train_labels:
        label_rng = np.random.RandomState(900000 + args.seed)
        train_y = train_y[label_rng.permutation(len(train_y))]
    scorer = LogisticRegression(C=args.C, class_weight="balanced", max_iter=1000,
                                random_state=args.seed, solver="liblinear")
    scorer.fit(train_x, train_y)

    results = {"base": [], "supervised_edge": [], "random": [], "truth": [],
               "supervised_edge_wrapped": [], "random_wrapped": []}
    bacc_raw, bacc_wrapped, bacc_random = [], [], []
    wrapper_edges, replace_rates = [], []
    t0 = time.time()
    for n, i in enumerate(eval_idx):
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        scores = scorer.predict_proba(edge_features(data, keep))[:, 1]
        order = np.argsort(scores)[::-1]
        top = order[:min(args.topk, len(keep))]
        results["base"].append(audit_baseline(net, []))
        results["supervised_edge"].append(
            audit_baseline(net, [mapping[keep[j]] for j in top]))

        chosen = select_feasible(order, scores, mapping, keep, net, args.topk,
                                 strategy=args.wrapper_strategy,
                                 candidate_pool=args.candidate_pool)
        wrapped = (audit_baseline(net, [mapping[keep[j]] for j in chosen])
                   if len(chosen) == args.topk else (False, "incomplete", {}))
        results["supervised_edge_wrapped"].append(wrapped)

        raw_rng = random.Random(100000 * args.seed + n)
        random_order = list(range(len(keep)))
        raw_rng.shuffle(random_order)
        random_top = random_order[:min(args.topk, len(keep))]
        results["random"].append(
            audit_baseline(net, [mapping[keep[j]] for j in random_top]))
        wrapper_rng = random.Random(200000 * args.seed + n)
        random_order = list(range(len(keep)))
        wrapper_rng.shuffle(random_order)
        random_chosen = select_feasible(random_order, np.zeros(len(keep)), mapping,
                                        keep, net, args.topk, strategy="greedy",
                                        candidate_pool=0)
        random_wrapped = (audit_baseline(net, [mapping[keep[j]] for j in random_chosen])
                          if len(random_chosen) == args.topk else (False, "incomplete", {}))
        results["random_wrapped"].append(random_wrapped)
        results["truth"].append(audit_baseline(net, [mapping[j] for j in truth_rows(exp, i)]))

        gt = np.asarray([int((row + 1) in {x + 1 for x in truth_rows(exp, i)}) for row in keep])
        def bacc(selected):
            pred = np.zeros(len(keep), dtype=int)
            pred[selected] = 1
            return balanced_accuracy_score(gt, pred)
        bacc_raw.append(bacc(top))
        bacc_wrapped.append(bacc(chosen) if chosen else 0.5)
        bacc_random.append(bacc(random_chosen) if random_chosen else 0.5)
        wrapper_edges.append(len(chosen))
        replace_rates.append(1 - len(set(chosen) & set(top)) / max(1, len(top)))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(eval_idx)} 完成")

    summary = {name: pfr_summary(values) for name, values in results.items()}
    summary["_meta"] = {
        "system": args.dataset, "method": "supervised_edge_logistic",
        "protocol": "v4_train_eval_disjoint_exactk", "seed": args.seed,
        "train_num": len(train_idx), "eval_start": eval_start,
        "n": len(eval_idx), "topk": args.topk,
        "max_graphs": args.max_graphs,
        "split_mode": args.split_mode,
        "train_scenarios": train_scenarios,
        "eval_scenarios": eval_scenarios,
        "C": args.C, "wrapper_strategy": args.wrapper_strategy,
        "candidate_pool": args.candidate_pool, "train_edge_n": int(len(train_y)),
        "train_positive_rate": float(np.mean(train_y)),
        "shuffle_train_labels": args.shuffle_train_labels,
        "runtime_seconds": time.time() - t0,
        "wrapper_mean_edges": float(np.mean(wrapper_edges)),
        "wrapper_mean_replace_rate": float(np.mean(replace_rates)),
        "wrapper_raw_bacc": float(np.mean(bacc_raw)),
        "wrapper_bacc": float(np.mean(bacc_wrapped)),
        "random_wrapper_bacc": float(np.mean(bacc_random)),
        "wrapper_incomplete_n": int(sum(r[1] == "incomplete" for r in results["supervised_edge_wrapped"])),
        "random_wrapper_incomplete_n": int(sum(r[1] == "incomplete" for r in results["random_wrapped"])),
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    for name in ("supervised_edge", "supervised_edge_wrapped", "random_wrapped", "truth"):
        print(f"{name}: PFR={summary[name].get('PFR', 0):.3f}")
    print(f"bacc raw->wrapped={np.mean(bacc_raw):.3f}->{np.mean(bacc_wrapped):.3f}; random={np.mean(bacc_random):.3f}")
    print(f"结果已保存: {args.output}")


if __name__ == "__main__":
    main()

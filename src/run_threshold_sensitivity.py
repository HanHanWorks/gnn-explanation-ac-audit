#!/usr/bin/env python3
"""Fixed-slice top-k and voltage-threshold sensitivity for the proposed oracle.

Ranks are trained only on train truth labels. Evaluation truth is used solely
for BACC; no feasible wrapper is applied because this experiment measures the
raw explanation-to-intervention stress test across physical thresholds.
"""
import argparse
import json
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mat73
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

from audit_core import audit_baseline, build_net, pfr_summary
from run_audit import load_sample
from run_supervised_edge_audit import edge_features, truth_rows

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--train_num", type=int, default=20)
    ap.add_argument("--eval_start", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--ks", default="1,3,5,8")
    ap.add_argument("--voltage_mins", default="0.85,0.90,0.95")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    voltage_mins = [float(x) for x in args.voltage_mins.split(",")]

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
        raise ValueError(f"not enough eval samples: {len(eval_idx)}")

    x, y = [], []
    for i in train_idx:
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        x.append(edge_features(data, keep)); truth = truth_rows(exp, i)
        y.append(np.asarray([int(row in truth) for row in keep]))
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=args.seed, solver="liblinear")
    model.fit(np.concatenate(x), np.concatenate(y))

    cached = []
    for n, i in enumerate(eval_idx):
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        score = model.predict_proba(edge_features(data, keep))[:, 1]
        order = np.argsort(score)[::-1]
        rng = random.Random(100000 * args.seed + n)
        ro = list(range(len(keep))); rng.shuffle(ro)
        gt = np.asarray([int(row in truth_rows(exp, i)) for row in keep])
        cached.append((keep, order, ro, gt))

    result = {"_meta": {"system": args.dataset, "protocol": "proposed-fixed-slice-raw-threshold-sensitivity",
        "seed": args.seed, "n": len(eval_idx), "train_num": len(train_idx), "eval_start": args.eval_start,
        "ks": ks, "voltage_mins": voltage_mins, "ranking_uses_eval_truth": False}, "rows": {}}
    for k in ks:
        top_bacc, rand_bacc = [], []
        selections = []
        for keep, order, ro, gt in cached:
            top, rnd = list(order[:min(k, len(keep))]), list(ro[:min(k, len(keep))])
            selections.append((keep, top, rnd))
            for selected, out in ((top, top_bacc), (rnd, rand_bacc)):
                pred = np.zeros(len(keep), dtype=int); pred[selected] = 1
                out.append(float(balanced_accuracy_score(gt, pred)))
        key = str(k)
        result["rows"][key] = {"proposed_bacc": float(np.mean(top_bacc)), "random_bacc": float(np.mean(rand_bacc)), "thresholds": {}}
        for vmin in voltage_mins:
            proposed, rnd = [], []
            for keep, top, random_top in selections:
                proposed.append(audit_baseline(net, [mapping[keep[j]] for j in top], volt_min=vmin, volt_max=1.10))
                rnd.append(audit_baseline(net, [mapping[keep[j]] for j in random_top], volt_min=vmin, volt_max=1.10))
            result["rows"][key]["thresholds"][f"{vmin:.2f}"] = {
                "proposed": pfr_summary(proposed), "random": pfr_summary(rnd)}
        print(f"k={k} BACC={np.mean(top_bacc):.3f} random={np.mean(rand_bacc):.3f}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=float)
 print(f" : {args.output}")

if __name__ == "__main__":
    main()

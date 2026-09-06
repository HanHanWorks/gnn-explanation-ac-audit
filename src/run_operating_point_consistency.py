#!/usr/bin/env python3
"""Compare public-case and sample-conditioned AC intervention audits.

The sample-conditioned protocol uses Ef-derived injections and the sample's
initial contingency edges. Truth is used only for reporting BACC and anchors.
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

from audit_core import audit, audit_baseline, build_net, derive_injections, pfr_summary
from run_audit import load_sample
from run_supervised_edge_audit import edge_features, truth_rows

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")

def cont_rows(ef, blist):
    f = np.asarray(ef, dtype=float).reshape(-1, 4)
    return [j for j in range(len(blist)) if np.all(np.abs(f[j]) < 1e-12)]

def select_sample(order, mapping, keep, net, pq, cont, topk):
    chosen = []
    for j in order:
        if len(chosen) >= topk:
            break
        candidate = chosen + [int(j)]
        ok, _, _ = audit(net, pq, [mapping[keep[k]] for k in candidate], cont_edges=cont)
        if ok:
            chosen = candidate
    return chosen

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=50)
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
    pos_idx = [i for i in range(len(of_bi))
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    train_idx = pos_idx[:args.train_num]
    eval_idx = pos_idx[args.eval_start:args.eval_start + args.num]
    if len(eval_idx) < args.num:
        raise ValueError(f"not enough eval samples: requested {args.num}, got {len(eval_idx)}")

    train_x, train_y = [], []
    for i in train_idx:
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        train_x.append(edge_features(data, keep))
        truth = truth_rows(exp, i)
        train_y.append(np.asarray([int(row in truth) for row in keep]))
    scorer = LogisticRegression(class_weight="balanced", max_iter=1000,
                                random_state=args.seed, solver="liblinear")
    scorer.fit(np.concatenate(train_x), np.concatenate(train_y))

    results = {"public_proposed": [], "sample_proposed": [], "public_random": [],
               "sample_random": [], "sample_truth": [], "sample_base": []}
    bacc_public, bacc_sample, bacc_random = [], [], []
    for n, i in enumerate(eval_idx):
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        f = np.asarray(Ef[i][0], dtype=float).reshape(-1, 4)
        cont = [mapping[j] for j in cont_rows(f, blist)]
        pq = derive_injections(f, blist, nbus=data.x.size(0))
        score = scorer.predict_proba(edge_features(data, keep))[:, 1]
        order = np.argsort(score)[::-1]
        top = order[:min(args.topk, len(keep))]
        chosen = select_sample(order, mapping, keep, net, pq, cont, args.topk)
        rng = random.Random(100000 * args.seed + n)
        random_order = list(range(len(keep))); rng.shuffle(random_order)
        rchosen = select_sample(random_order, mapping, keep, net, pq, cont, args.topk)
        truth = [j for j in truth_rows(exp, i) if j in set(keep)]

        def public(selected):
            return audit_baseline(net, [mapping[keep[j]] for j in selected])
        def sample(selected):
            return audit(net, pq, [mapping[keep[j]] for j in selected], cont_edges=cont)
        results["public_proposed"].append(public(chosen))
        results["sample_proposed"].append(sample(chosen) if len(chosen) == args.topk else (False, "incomplete", {}))
        results["public_random"].append(public(rchosen))
        results["sample_random"].append(sample(rchosen) if len(rchosen) == args.topk else (False, "incomplete", {}))
        results["sample_truth"].append(sample(truth))
        results["sample_base"].append(sample([]))

        gt = np.asarray([int(row in set(truth_rows(exp, i))) for row in keep])
        for selected, target in ((chosen, bacc_public), (chosen, bacc_sample), (rchosen, bacc_random)):
            pred = np.zeros(len(keep), dtype=int); pred[selected] = 1
            target.append(float(balanced_accuracy_score(gt, pred)))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(eval_idx)} completed")

    summary = {k: pfr_summary(v) for k, v in results.items()}
    summary["_meta"] = {"system": args.dataset, "protocol": "public-vs-sample-conditioned",
        "method": "supervised_edge_logistic", "seed": args.seed, "n": len(eval_idx),
        "train_num": len(train_idx), "eval_start": args.eval_start, "topk": args.topk,
        "ranking_uses_eval_truth": False, "public_bacc": float(np.mean(bacc_public)),
        "sample_bacc": float(np.mean(bacc_sample)), "random_bacc": float(np.mean(bacc_random))}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=float)
    print({k: summary[k].get("PFR") for k in results})
    print(f"BACC public/sample/random={np.mean(bacc_public):.3f}/{np.mean(bacc_sample):.3f}/{np.mean(bacc_random):.3f}")
 print(f" : {args.output}")

if __name__ == "__main__":
    main()

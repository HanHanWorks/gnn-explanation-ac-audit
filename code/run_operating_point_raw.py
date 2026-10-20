#!/usr/bin/env python3
"""Identical raw top-k sets audited under two operating-point protocols.

The public-case protocol keeps the MATPOWER case dispatch and does not retain
the initiating outage.  The sample-conditioned protocol reconstructs net
injections from the stored post-contingency branch flows and keeps the sample's
initiating-outage branches out of service.  Crucially, both protocols audit the
*same raw top-k set*; no feasibility-guided selection is applied here.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import audit, audit_baseline, build_net, derive_injections, pfr_summary  # noqa: E402
import light_core as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--train_num", type=int, default=20)
    ap.add_argument("--eval_start", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    m = L.load_mats(args.dataset, args.root)
    net, mapping = build_net(args.dataset, root=args.root)
    train_idx, eval_idx, pos = L.split_indices(
        m, args.seed, args.train_num, args.eval_start, args.num
    )
    X, y = [], []
    for i in train_idx:
        x, ea, src, dst, keep, cont, yy, f = L.sample_arrays(m, i)
        X.append(L.edge_features(ea, x, src, dst))
        truth = L.truth_rows(m, i)
        y.append(np.asarray([int(int(keep[j]) in truth) for j in range(len(keep))]))
    clf = LogisticRegression(
        class_weight="balanced", max_iter=1000, random_state=args.seed, solver="liblinear"
    ).fit(np.concatenate(X), np.concatenate(y))

    res = {k: [] for k in [
        "public_raw", "sample_raw", "public_random", "sample_random",
        "sample_truth", "sample_base", "public_base",
    ]}
    bacc = []
    n_inject = []
    for n, i in enumerate(eval_idx):
        x, ea, src, dst, keep, cont, yy, f = L.sample_arrays(m, i)
        blist = m.blist
        pq = derive_injections(f, blist, nbus=len(net.bus), bus_labels=net.bus.index)
        cont_edges = [mapping[j] for j in cont]
        scores = clf.predict_proba(L.edge_features(ea, x, src, dst))[:, 1]
        order = np.argsort(scores)[::-1]
        top = [int(j) for j in order[: min(args.topk, len(keep))]]
        disc = [mapping[keep[j]] for j in top]

        rng = random.Random(100000 * args.seed + n)
        ro = list(range(len(keep)))
        rng.shuffle(ro)
        rtop = [int(j) for j in ro[: min(args.topk, len(keep))]]
        rdisc = [mapping[keep[j]] for j in rtop]

        res["public_base"].append(audit_baseline(net, []))
        res["public_raw"].append(audit_baseline(net, disc))
        res["public_random"].append(audit_baseline(net, rdisc))
        res["sample_base"].append(audit(net, pq, [], cont_edges=cont_edges))
        res["sample_raw"].append(audit(net, pq, disc, cont_edges=cont_edges))
        res["sample_random"].append(audit(net, pq, rdisc, cont_edges=cont_edges))
        truth = [j for j in L.truth_rows(m, i) if j in set(keep)]
        res["sample_truth"].append(
            audit(net, pq, [mapping[keep[j]] for j in truth], cont_edges=cont_edges)
        )

        gt = np.asarray([int(int(keep[j]) in L.truth_rows(m, i)) for j in range(len(keep))])
        pred = np.zeros(len(keep), dtype=int)
        pred[top] = 1
        bacc.append(float(balanced_accuracy_score(gt, pred)))
        n_inject.append(len(cont))

    summary = {k: pfr_summary(v) for k, v in res.items()}
    summary["_meta"] = {
        "system": args.dataset, "protocol": "raw-topk-public-vs-sample-conditioned",
        "seed": args.seed, "n": len(eval_idx), "train_num": args.train_num,
        "eval_start": args.eval_start, "topk": args.topk,
        "ranking_uses_eval_truth": False,
        "bacc": float(np.mean(bacc)),
        "mean_initiating_outage_branches": float(np.mean(n_inject)),
    }
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print({k: summary[k].get("PFR") for k in res})
    print(f"BACC={np.mean(bacc):.3f}; saved {args.output}")


if __name__ == "__main__":
    main()

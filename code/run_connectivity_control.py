#!/usr/bin/env python3
"""Connectivity-preserving control for the SEP ranking.

For each evaluation graph we scan the SEP ranking and retain only edges whose
*joint* removal leaves the bus graph connected.  We report the largest such
prefix (<= k), whether it reaches k, and the AC-screen outcome of the connected
set.  Comparing this control with the raw top-k outcome separates failures that
require loss of connectivity from failures that persist on a connected network
(voltage or non-convergence).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import audit_baseline, build_net, pfr_summary  # noqa: E402
import light_core as L  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--seeds", default="2,3,4,5,6")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--train_num", type=int, default=20)
    ap.add_argument("--eval_start", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    m = L.load_mats(args.dataset, args.root)
    net, mapping = build_net(args.dataset, root=args.root)
    out = {"_meta": vars(args), "seeds": {}}

    for seed in [int(x) for x in args.seeds.split(",")]:
        train_idx, eval_idx, pos = L.split_indices(
            m, seed, args.train_num, args.eval_start, args.num
        )
        X, y = [], []
        for i in train_idx:
            x, ea, src, dst, keep, cont, yy, f = L.sample_arrays(m, i)
            X.append(L.edge_features(ea, x, src, dst))
            truth = L.truth_rows(m, i)
            y.append(np.asarray([int(int(keep[j]) in truth) for j in range(len(keep))]))
        clf = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed, solver="liblinear"
        ).fit(np.concatenate(X), np.concatenate(y))

        raw, conn = [], []
        max_reached = []
        for i in eval_idx:
            x, ea, src, dst, keep, cont, yy, f = L.sample_arrays(m, i)
            scores = clf.predict_proba(L.edge_features(ea, x, src, dst))[:, 1]
            order = np.argsort(scores)[::-1]
            top = [int(j) for j in order[: min(args.topk, len(keep))]]
            raw.append(audit_baseline(net, [mapping[keep[j]] for j in top]))

            chosen = []
            for j in order:
                if len(chosen) >= args.topk:
                    break
                cand = chosen + [int(j)]
                if not L.is_connected_after(net, mapping, [mapping[keep[k]] for k in cand]):
                    continue
                chosen = cand
            max_reached.append(len(chosen))
            if len(chosen) == args.topk:
                conn.append(audit_baseline(net, [mapping[keep[j]] for j in chosen]))
            else:
                conn.append((False, "incomplete", {}))

        out["seeds"][str(seed)] = {
            "n": len(eval_idx),
            "raw": pfr_summary(raw),
            "connected": pfr_summary(conn),
            "raw_types": dict(Counter(s for _, s, _ in raw)),
            "connected_types": dict(Counter(s for _, s, _ in conn)),
            "largest_connected_prefix": {
                "mean": float(np.mean(max_reached)),
                "min": int(np.min(max_reached)),
                "max": int(np.max(max_reached)),
                "reached_k": int(np.sum(np.asarray(max_reached) == args.topk)),
                "distribution": {str(k): int(v) for k, v in sorted(Counter(max_reached).items())},
            },
        }
        print(
            f"{args.dataset} seed {seed}: raw={out['seeds'][str(seed)]['raw'].get('PFR')} "
            f"connected={out['seeds'][str(seed)]['connected'].get('PFR')} "
            f"reached_k={out['seeds'][str(seed)]['largest_connected_prefix']['reached_k']}/{len(eval_idx)} "
            f"dist={out['seeds'][str(seed)]['largest_connected_prefix']['distribution']}"
        )
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

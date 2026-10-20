#!/usr/bin/env python3
"""Wrapper AC-call accounting and stricter-voltage-threshold sensitivity.

For each evaluation graph we run the local-swap wrapper exactly as in the
manuscript, count every AC power-flow evaluation (greedy phase + local swaps),
and record the replacement rate and incompleteness.  Raw top-k sets are also
screened at 0.85, 0.90 and 0.95 p.u. to report the announced threshold
sensitivity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

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
    ap.add_argument("--thresholds", default="0.85,0.90,0.95")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    thresholds = [float(x) for x in args.thresholds.split(",")]

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

        rows = []
        for n, i in enumerate(eval_idx):
            x, ea, src, dst, keep, cont, yy, f = L.sample_arrays(m, i)
            scores = clf.predict_proba(L.edge_features(ea, x, src, dst))[:, 1]
            order = np.argsort(scores)[::-1]
            top = [int(j) for j in order[: min(args.topk, len(keep))]]
            r = {"i": int(i), "raw": {}, "connected": None}
            for vmin in thresholds:
                ok, st, _ = audit_baseline(
                    net, [mapping[keep[j]] for j in top], volt_min=vmin
                )
                r["raw"][f"{vmin:.2f}"] = st
            calls = [0]
            chosen = L.select_feasible_instrumented(
                order, scores, mapping, keep, net, args.topk,
                strategy="local_swap", candidate_pool=8, ac_call_counter=calls,
            )
            if len(chosen) == args.topk:
                okw, stw, _ = audit_baseline(net, [mapping[keep[j]] for j in chosen])
            else:
                okw, stw = False, "incomplete"
            r["wrapper"] = stw
            r["ac_calls"] = calls[0]
            r["replace_rate"] = 1 - len(set(chosen) & set(top)) / max(1, len(top))
            r["raw_bacc"] = float(
                balanced_accuracy_score(
                    np.asarray([int(int(keep[j]) in L.truth_rows(m, i)) for j in range(len(keep))]),
                    np.asarray([1 if j in top else 0 for j in range(len(keep))]),
                )
            )
            rows.append(r)

        calls = [r["ac_calls"] for r in rows]
        out["seeds"][str(seed)] = {
            "n": len(rows),
            "raw_pfr_by_threshold": {
                f"{v:.2f}": pfr_summary(
                    [(r["raw"][f"{v:.2f}"] == "feasible", r["raw"][f"{v:.2f}"], {}) for r in rows]
                )
                for v in thresholds
            },
            "raw_types_085": dict(Counter(r["raw"]["0.85"] for r in rows)),
            "wrapper": pfr_summary(
                [(r["wrapper"] == "feasible", r["wrapper"], {}) for r in rows]
            ),
            "ac_calls": {
                "mean": float(np.mean(calls)),
                "median": float(np.median(calls)),
                "p90": float(np.percentile(calls, 90)),
                "p95": float(np.percentile(calls, 95)),
                "max": int(np.max(calls)),
                "min": int(np.min(calls)),
                "values": calls,
            },
            "replace_rate_mean": float(np.mean([r["replace_rate"] for r in rows])),
            "replace_rate_sd": float(np.std([r["replace_rate"] for r in rows], ddof=1)),
            "incomplete": int(sum(r["wrapper"] == "incomplete" for r in rows)),
        }
        s = out["seeds"][str(seed)]
        print(
            f"{args.dataset} seed {seed}: raw085={s['raw_pfr_by_threshold']['0.85'].get('PFR'):.3f} "
            f"raw090={s['raw_pfr_by_threshold']['0.90'].get('PFR'):.3f} "
            f"raw095={s['raw_pfr_by_threshold']['0.95'].get('PFR'):.3f} "
            f"wrap={s['wrapper'].get('PFR'):.3f} calls mean/median/max="
            f"{s['ac_calls']['mean']:.1f}/{s['ac_calls']['median']:.0f}/{s['ac_calls']['max']} "
            f"incomplete={s['incomplete']}"
        )
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Matched-sample and control analyses for the RESS revision.

Produces, for the supervised edge-label predictor (SEP):
  * raw top-k AC screen outcomes at several voltage thresholds;
  * matched random top-k outcomes on the same samples and candidate edges;
  * a connectivity-preserving top-k control (skip bridges) and its AC outcome;
  * a connectivity-preserving random control;
  * the finite-budget local-swap wrapper with actual AC-solver call counts;
  * edge-label BACC for raw, connected, wrapped and random sets.

All PFR quantities are conditional on the accepted explainable-positive samples.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import audit_baseline, build_net, pfr_summary  # noqa: E402
import light_core as L  # noqa: E402


def summarize(values):
    return pfr_summary(values)


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
        t0 = time.time()
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

        records = []
        agg = {
            "raw": [], "random": [], "connected": [], "connected_random": [],
            "wrapper": [],
        }
        for n, i in enumerate(eval_idx):
            x, ea, src, dst, keep, cont, yy, f = L.sample_arrays(m, i)
            scores = clf.predict_proba(L.edge_features(ea, x, src, dst))[:, 1]
            order = np.argsort(scores)[::-1]
            top = [int(j) for j in order[: min(args.topk, len(keep))]]
            truth = L.truth_rows(m, i)

            # matched random set on the same candidate edges
            rng = random.Random(100000 * seed + n)
            rorder = list(range(len(keep)))
            rng.shuffle(rorder)
            rtop = [int(j) for j in rorder[: min(args.topk, len(keep))]]

            rec = {"i": int(i), "keep_n": len(keep), "truth_n": len(truth)}
            # raw at each threshold; full taxonomy at 0.85
            for vmin in thresholds:
                ok, st, _ = audit_baseline(net, [mapping[keep[j]] for j in top], volt_min=vmin)
                rec[f"raw_{vmin:.2f}"] = st
            agg["raw"].append((rec["raw_0.85"] == "feasible", rec["raw_0.85"], {}))
            okr, str_, _ = audit_baseline(net, [mapping[keep[j]] for j in rtop])
            rec["random"] = str_
            agg["random"].append((okr, str_, {}))

            # connectivity-preserving selection
            chosen = []
            skipped = 0
            for j in order:
                if len(chosen) >= args.topk:
                    break
                cand = chosen + [int(j)]
                if not L.is_connected_after(net, mapping, [mapping[keep[k]] for k in cand]):
                    skipped += 1
                    continue
                chosen = cand
            if len(chosen) == args.topk:
                okc, stc, _ = audit_baseline(net, [mapping[keep[j]] for j in chosen])
            else:
                okc, stc = False, "incomplete"
            rec["connected"] = stc
            rec["connected_skipped"] = skipped
            agg["connected"].append((okc, stc, {}))

            # connectivity-preserving random control
            crand = []
            for j in rorder:
                if len(crand) >= args.topk:
                    break
                cand = crand + [int(j)]
                if not L.is_connected_after(net, mapping, [mapping[keep[k]] for k in cand]):
                    continue
                crand = cand
            if len(crand) == args.topk:
                okcr, stcr, _ = audit_baseline(net, [mapping[keep[j]] for j in crand])
            else:
                okcr, stcr = False, "incomplete"
            rec["connected_random"] = stcr
            agg["connected_random"].append((okcr, stcr, {}))

            # instrumented local-swap wrapper
            calls = [0]
            wchosen = L.select_feasible_instrumented(
                order, scores, mapping, keep, net, args.topk,
                strategy="local_swap", candidate_pool=8, ac_call_counter=calls,
            )
            if len(wchosen) == args.topk:
                okw, stw, _ = audit_baseline(net, [mapping[keep[j]] for j in wchosen])
            else:
                okw, stw = False, "incomplete"
            rec["wrapper"] = stw
            rec["ac_calls"] = calls[0]
            rec["replace_rate"] = 1 - len(set(wchosen) & set(top)) / max(1, len(top))
            agg["wrapper"].append((okw, stw, {}))

            # edge-label agreement
            def bacc(selected):
                pred = np.zeros(len(keep), dtype=int)
                pred[selected] = 1
                gt = np.asarray([int(int(keep[j]) in truth) for j in range(len(keep))])
                return float(balanced_accuracy_score(gt, pred))

            rec["bacc_raw"] = bacc(top)
            rec["bacc_random"] = bacc(rtop)
            rec["bacc_connected"] = bacc(chosen) if len(chosen) == args.topk else None
            rec["bacc_wrapper"] = bacc(wchosen) if len(wchosen) == args.topk else None
            records.append(rec)

        def pfr(key):
            vals = agg[key]
            s = pfr_summary(vals)
            return s

        seed_out = {
            "n": len(eval_idx),
            "raw": pfr("raw"),
            "random": pfr("random"),
            "connected": pfr("connected"),
            "connected_random": pfr("connected_random"),
            "wrapper": pfr("wrapper"),
            "thresholds": {
                f"{v:.2f}": dict(Counter(r[f"raw_{v:.2f}"] for r in records))
                for v in thresholds
            },
            "bacc": {
                k: float(np.mean([r[k] for r in records if r[k] is not None]))
                for k in ("bacc_raw", "bacc_random", "bacc_connected", "bacc_wrapper")
            },
            "ac_calls": {
                "mean": float(np.mean([r["ac_calls"] for r in records])),
                "median": float(np.median([r["ac_calls"] for r in records])),
                "p95": float(np.percentile([r["ac_calls"] for r in records], 95)),
                "max": int(np.max([r["ac_calls"] for r in records])),
                "min": int(np.min([r["ac_calls"] for r in records])),
                "all": [int(r["ac_calls"]) for r in records],
            },
            "replace_rate": float(np.mean([r["replace_rate"] for r in records])),
            "connected_skipped": float(np.mean([r["connected_skipped"] for r in records])),
            "records": records,
            "runtime_seconds": time.time() - t0,
        }
        out["seeds"][str(seed)] = seed_out
        print(
            f"seed {seed}: n={seed_out['n']} raw={seed_out['raw'].get('PFR')} "
            f"random={seed_out['random'].get('PFR')} connected={seed_out['connected'].get('PFR')} "
            f"conn-rand={seed_out['connected_random'].get('PFR')} "
            f"bacc={seed_out['bacc']['bacc_raw']:.3f} ac_calls={seed_out['ac_calls']['mean']:.1f}"
        )
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

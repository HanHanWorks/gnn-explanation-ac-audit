#!/usr/bin/env python3
"""Pooled edge-label agreement and AC-screen outcomes for quality baselines.

Every ranking is evaluated on the same accepted positive-class slice and the
same candidate edges.  Edge balanced accuracy is pooled over all evaluated
edges (TP/FP/TN/FN summed across graphs), not averaged over per-graph scores;
this avoids the single-class artefact that inflates per-graph averages on
graphs without a cascade-path edge.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_core import audit_baseline, build_net, pfr_summary  # noqa: E402
import light_core as L  # noqa: E402


def pooled_bacc(tp, fp, tn, fn):
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    return 0.5 * (tpr + tnr), tpr, tnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--seeds", default="2,3,4,5,6")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--train_num", type=int, default=20)
    ap.add_argument("--eval_start", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--shuffles", type=int, default=10)
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
        X = np.concatenate(X)
        y = np.concatenate(y)
        sep = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=seed, solver="liblinear"
        ).fit(X, y)

        methods = ["SEP", "absQ", "absPQ", "random", "truth"]
        counts = {k: [0, 0, 0, 0] for k in methods}  # tp fp tn fn
        outcomes = {k: [] for k in methods}
        # shuffled-label classifiers: same split, different label permutations
        rng = np.random.RandomState(900000 + seed)
        shuffle_models = []
        for _ in range(args.shuffles):
            yy = y[rng.permutation(len(y))]
            if len(np.unique(yy)) < 2:
                continue
            shuffle_models.append(
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=seed,
                    solver="liblinear",
                ).fit(X, yy)
            )
        shuffle_counts = [[0, 0, 0, 0] for _ in shuffle_models]

        for n, i in enumerate(eval_idx):
            x, ea, src, dst, keep, cont, yy, f = L.sample_arrays(m, i)
            truth = L.truth_rows(m, i)
            gt = np.asarray([int(int(keep[j]) in truth) for j in range(len(keep))], dtype=int)
            feats = L.edge_features(ea, x, src, dst)

            rng_i = random.Random(100000 * seed + n)
            ro = list(range(len(keep)))
            rng_i.shuffle(ro)
            rankings = {
                "SEP": np.argsort(sep.predict_proba(feats)[:, 1])[::-1],
                "absQ": np.argsort(np.abs(ea[:, 1]))[::-1],
                "absPQ": np.argsort(np.abs(ea[:, 0]) + np.abs(ea[:, 1]))[::-1],
                "random": np.asarray(ro),
                "truth": np.asarray([j for j in range(len(keep)) if int(keep[j]) in truth]),
            }
            for name, order in rankings.items():
                top = order[: min(args.topk, len(keep))]
                pred = np.zeros(len(keep), dtype=int)
                pred[top] = 1
                counts[name][0] += int(((pred == 1) & (gt == 1)).sum())
                counts[name][1] += int(((pred == 1) & (gt == 0)).sum())
                counts[name][2] += int(((pred == 0) & (gt == 0)).sum())
                counts[name][3] += int(((pred == 0) & (gt == 1)).sum())
                outcomes[name].append(
                    audit_baseline(net, [mapping[keep[j]] for j in top])
                )

            for mi, clf in enumerate(shuffle_models):
                top = np.argsort(clf.predict_proba(feats)[:, 1])[::-1][: min(args.topk, len(keep))]
                pred = np.zeros(len(keep), dtype=int)
                pred[top] = 1
                shuffle_counts[mi][0] += int(((pred == 1) & (gt == 1)).sum())
                shuffle_counts[mi][1] += int(((pred == 1) & (gt == 0)).sum())
                shuffle_counts[mi][2] += int(((pred == 0) & (gt == 0)).sum())
                shuffle_counts[mi][3] += int(((pred == 0) & (gt == 1)).sum())

        seed_out = {"n": len(eval_idx), "methods": {}, "shuffled_label_bacc": {}}
        for name in methods:
            tp, fp, tn, fn = counts[name]
            b, tpr, tnr = pooled_bacc(tp, fp, tn, fn)
            seed_out["methods"][name] = {
                "bacc": b, "tpr": tpr, "tnr": tnr,
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                "pfr": pfr_summary(outcomes[name]),
                "types": dict(Counter(s for _, s, _ in outcomes[name])),
            }
        shuffled_bacc = []
        for tp, fp, tn, fn in shuffle_counts:
            if (tp + fn) and (tn + fp):
                shuffled_bacc.append(0.5 * (tp / (tp + fn) + tn / (tn + fp)))
        seed_out["shuffled_label_bacc"] = {
            "n_models": len(shuffle_models),
            "mean": float(np.mean(shuffled_bacc)) if shuffled_bacc else None,
            "min": float(np.min(shuffled_bacc)) if shuffled_bacc else None,
            "max": float(np.max(shuffled_bacc)) if shuffled_bacc else None,
            "values": shuffled_bacc,
            "n_ge_sep": int(sum(v >= seed_out["methods"]["SEP"]["bacc"] for v in shuffled_bacc)),
        }
        out["seeds"][str(seed)] = seed_out
        print(
            f"seed {seed}: " + " ".join(
                f"{k} BACC={v['bacc']:.3f} PFR={v['pfr'].get('PFR',0):.3f}"
                for k, v in seed_out["methods"].items()
            )
        )
        print(f"  shuffled-label BACC mean={seed_out['shuffled_label_bacc']['mean']:.3f} "
              f"range={seed_out['shuffled_label_bacc']['min']:.3f}-{seed_out['shuffled_label_bacc']['max']:.3f}")
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1, default=float)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

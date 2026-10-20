#!/usr/bin/env python3
"""Aggregate archived and newly generated results for the RESS revision."""
from __future__ import annotations

import glob
import json
import os
from collections import Counter

import numpy as np

ROOT = os.environ.get("RESS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
P1 = os.path.join(ROOT, "server_data", "phase1")


def load(path):
    with open(path) as f:
        return json.load(f)


def per_seed(method, system):
    """Return list of (seed, n, feasible, types, random_pfr, truth_pfr, bacc)."""
    rows = []
    pats = {
        "GNNExplainer": {
            "ieee24": "results_ieee24_gin_unified_v8_eval50_s{}.json",
            "ieee39": "results_ieee39_gin_unified_v8_eval50_s{}.json",
            "uk": "results_uk_gin_unified_v8_full_eval50_s{}.json",
        },
        "PGExplainer": {
            "ieee24": "results_ieee24_pg_unified_v8_eval50_s{}.json",
            "ieee39": "results_ieee39_pg_unified_v8_eval50_s{}.json",
            "uk": "results_uk_pg_unified_v8_full_eval50_s{}.json",
        },
        "Grad-CAM": {
            "ieee24": "results_ieee24_gradcam_v8_s{}_n200.json",
            "ieee39": "results_ieee39_gradcam_v8_s{}_n104.json",
            "uk": "results_uk_gradcam_v8_s{}_n200.json",
        },
        "GSAT": {
            "ieee24": "results_ieee24_gsat_v8_s{}_n200.json",
            "ieee39": "results_ieee39_gsat_v8_s{}_n104.json",
            "uk": "results_uk_gsat_v8_s2_n200.json",
        },
        "SubgraphX": {
            "ieee24": "results_ieee24_subgraphx_v8_s2_n200.json",
            "ieee39": "results_ieee39_subgraphx_v8_s2_n104.json",
            "uk": "results_uk_subgraphx_v8_pilot50_s2.json",
        },
    }
    keymap = {
        "GNNExplainer": "gnn_explainer", "PGExplainer": "pgexplainer",
        "Grad-CAM": "gradcam", "GSAT": "gsat", "SubgraphX": "subgraphx",
    }
    seeds = [2, 3, 4, 5, 6]
    if method == "GSAT":
        seeds = [2, 3, 4] if system != "uk" else [2]
    if method == "SubgraphX":
        seeds = [2]
    for s in seeds:
        p = os.path.join(P1, pats[method][system].format(s))
        if not os.path.exists(p):
            continue
        d = load(p)
        v = d[keymap[method]]
        rand = d.get("random", {})
        truth = d.get("truth", {})
        rows.append({
            "seed": s, "n": v["n"], "feasible": v["types"].get("feasible", 0),
            "types": v["types"], "pfr": v["PFR"],
            "random_pfr": rand.get("PFR"), "truth_pfr": truth.get("PFR"),
            "bacc": d.get("_meta", {}).get("wrapper_raw_bacc"),
        })
    return rows


def pooled(rows):
    n = sum(r["n"] for r in rows)
    c = Counter()
    for r in rows:
        c.update(r["types"])
    return n, c, c.get("feasible", 0) / n if n else None


def boot_ci(rows, reps=4000, seed=7):
    rng = np.random.RandomState(seed)
    if len(rows) < 2:
        return None
    rates = []
    for _ in range(reps):
        idx = rng.randint(0, len(rows), len(rows))
        n = sum(rows[i]["n"] for i in idx)
        f = sum(rows[i]["feasible"] for i in idx)
        rates.append(f / n)
    return [float(np.percentile(rates, 2.5)), float(np.percentile(rates, 97.5))]


def delta_stats(rows):
    d = [r["pfr"] - r["random_pfr"] for r in rows if r["random_pfr"] is not None]
    if not d:
        return None
    d = np.asarray(d)
    out = {"mean": float(d.mean()), "n": len(d), "values": d.tolist()}
    if len(d) > 1:
        out["sd"] = float(d.std(ddof=1))
        out["cohen_d"] = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None
        rng = np.random.RandomState(11)
        bs = [float(np.mean(d[rng.randint(0, len(d), len(d))])) for _ in range(4000)]
        out["ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return out


def main():
    out = {}
    for system in ["ieee24", "ieee39", "uk"]:
        out[system] = {}
        for method in ["GNNExplainer", "PGExplainer", "Grad-CAM", "GSAT", "SubgraphX"]:
            rows = per_seed(method, system)
            if not rows:
                continue
            n, c, pfr = pooled(rows)
            out[system][method] = {
                "rows": rows, "n": n, "types": dict(c), "pfr": pfr,
                "ci": boot_ci(rows), "delta": delta_stats(rows),
            }
    # new analyses
    out["quality"] = {}
    for system in ["ieee24", "ieee39", "uk"]:
        p = os.path.join(ROOT, "code", "results_quality", f"{system}_quality.json")
        if os.path.exists(p):
            out["quality"][system] = load(p)
    out["matched"] = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "code", "results_matched", "*.json"))):
        d = load(f)
        out["matched"][os.path.basename(f)] = {
            "n": d["supervised_edge"]["n"],
            "sep": d["supervised_edge"], "random": d["random"], "truth": d["truth"],
            "wrapped": d["supervised_edge_wrapped"],
            "bacc_pergraph": d["_meta"].get("wrapper_raw_bacc"),
            "replace": d["_meta"].get("wrapper_mean_replace_rate"),
        }
    for key, sub in [("conn", "results_conn"), ("wrapper", "results_wrapper"),
                     ("op", "results_op"), ("pg_quality", "results_pg_quality")]:
        out[key] = {}
        for f in sorted(glob.glob(os.path.join(ROOT, "code", sub, "*.json"))):
            out[key][os.path.basename(f)] = load(f)
    with open(os.path.join(ROOT, "revision_summary.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)

    # readable digest
    lines = []
    for system in ["ieee24", "ieee39", "uk"]:
        lines.append(f"## {system}")
        for method, v in out[system].items():
            ci = v["ci"]
            d = v["delta"]
            lines.append(
                f"{method:14s} n={v['n']:4d} PFR={v['pfr']:.4f} "
                f"CI={ci} types={v['types']} dPFR={d['mean'] if d else None} "
                f"dCI={d.get('ci') if d else None} BACC={[round(r['bacc'],3) for r in v['rows'] if r['bacc']]}"
            )
    with open(os.path.join(ROOT, "revision_summary.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Summarize same-evaluation-slice GNNExplainer PFR controls."""
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
PATTERNS = {
    "ieee24": "results_ieee24_gin_unified_v8_eval50_s*.json",
    "ieee39": "results_ieee39_gin_unified_v8_eval50_s*.json",
    "uk_full": "results_uk_gin_unified_v8_full_eval50_s*.json",
}


def ms(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    out = {"protocol": "unified_eval_start50_gnnexplainer_v8", "groups": {}, "gate": "NOT_SOTA"}
    for system, pattern in PATTERNS.items():
        paths = sorted(glob.glob(str(ROOT / pattern)))
        if len(paths) != 5:
            raise RuntimeError(f"{system}: expected 5 files, found {len(paths)}")
        rows = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            rows.append({
                "seed": d["_meta"]["seed"], "file": Path(path).name,
                "gnn_pfr": d["gnn_explainer"]["PFR"],
                "random_pfr": d["random"]["PFR"],
                "truth_pfr": d["truth"]["PFR"],
                "gnn_risk": d["gnn_explainer"].get("mean_risk"),
                "random_risk": d["random"].get("mean_risk"),
            })
        diff = np.asarray([r["gnn_pfr"] - r["random_pfr"] for r in rows])
        out["groups"][system] = {
            "rows": rows,
            "gnn_pfr": ms([r["gnn_pfr"] for r in rows]),
            "random_pfr": ms([r["random_pfr"] for r in rows]),
            "truth_pfr": ms([r["truth_pfr"] for r in rows]),
            "difference_by_seed": diff.tolist(),
            "difference_p": float(stats.ttest_1samp(diff, 0).pvalue),
        }
    tests = sorted(((s, d["difference_p"]) for s, d in out["groups"].items()), key=lambda x: x[1])
    for rank, (system, p) in enumerate(tests):
        out["groups"][system]["difference_holm_p"] = min(1.0, p * (len(tests) - rank))
    prefix = ROOT / "summary_unified_v8_gnnexplainer_eval50"
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f"{prefix}.md", "w", encoding="utf-8") as f:
        f.write("# 同评估切片 GNNExplainer v8 PFR 统计\n\n")
        f.write("| system | GNNExplainer PFR | random PFR | GNN-random by seed | raw p | Holm p | truth PFR |\n|---|---:|---:|---|---:|---:|---:|\n")
        for system, d in out["groups"].items():
            f.write(f"| {system} | {d['gnn_pfr'][0]:.4f} +/- {d['gnn_pfr'][1]:.4f} | {d['random_pfr'][0]:.4f} +/- {d['random_pfr'][1]:.4f} | `{d['difference_by_seed']}` | {d['difference_p']:.4f} | {d['difference_holm_p']:.4f} | {d['truth_pfr'][0]:.4f} +/- {d['truth_pfr'][1]:.4f} |\n")
        f.write("\n门审：GNNExplainer 在 IEEE24/39 的 PFR 稳定低于 random；UK 全量 PFR 饱和。该强基线不构成 v7r 质量-可行性优势的反例，但 v7r 仍需 PG/GSAT/Grad-CAM/SubgraphX 同切片主表后才可决定最终 SOTA。\n")
    print(f"JSON: {prefix}.json")
    print(f"MD: {prefix}.md")
    for system, d in out["groups"].items():
        print(system, d["gnn_pfr"], d["random_pfr"], d["difference_holm_p"])


if __name__ == "__main__":
    main()

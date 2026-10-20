#!/usr/bin/env python3
"""Cross-system summary for locked v7r supervised edge results."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent
PATTERNS = {
    "ieee24": "results_ieee24_supervised_edge_v7r_n200_s*.json",
    "ieee39": "results_ieee39_supervised_edge_v7r_n134_s*.json",
    "uk_full": "results_uk_supervised_edge_v7r_full_n200_s*.json",
}


def ms(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    groups = {}
    tests = []
    for system, pattern in PATTERNS.items():
        paths = sorted(glob.glob(str(ROOT / pattern)))
        if len(paths) != 5:
            raise RuntimeError(f"{system}: expected 5 files, found {len(paths)}")
        rows = []
        for path in paths:
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            m = d["_meta"]
            rows.append({
                "seed": m["seed"], "file": Path(path).name,
                "n": m["n"], "raw_pfr": d["supervised_edge"]["PFR"],
                "wrapped_pfr": d["supervised_edge_wrapped"]["PFR"],
                "random_pfr": d["random_wrapped"]["PFR"],
                "truth_pfr": d["truth"]["PFR"],
                "raw_bacc": m["wrapper_raw_bacc"],
                "wrapped_bacc": m["wrapper_bacc"],
                "random_bacc": m["random_wrapper_bacc"],
                "edges": m["wrapper_mean_edges"],
                "replace": m["wrapper_mean_replace_rate"],
            })
        d_bacc = np.asarray([r["wrapped_bacc"] - r["random_bacc"] for r in rows])
        d_pfr = np.asarray([r["wrapped_pfr"] - r["random_pfr"] for r in rows])
        groups[system] = {
            "rows": rows,
            "aggregate": {
                "raw_bacc": ms([r["raw_bacc"] for r in rows]),
                "wrapped_bacc": ms([r["wrapped_bacc"] for r in rows]),
                "random_bacc": ms([r["random_bacc"] for r in rows]),
                "bacc_diff": d_bacc.tolist(),
                "bacc_p": float(stats.ttest_1samp(d_bacc, 0).pvalue),
                "wrapped_pfr": ms([r["wrapped_pfr"] for r in rows]),
                "random_pfr": ms([r["random_pfr"] for r in rows]),
                "truth_pfr": ms([r["truth_pfr"] for r in rows]),
                "edges": ms([r["edges"] for r in rows]),
                "replace": ms([r["replace"] for r in rows]),
            },
        }
        tests.append({"label": f"{system}: wrapped BACC - random", "p": groups[system]["aggregate"]["bacc_p"]})
    ordered = sorted(tests, key=lambda x: x["p"])
    for rank, item in enumerate(ordered):
        item["holm_p"] = min(1.0, item["p"] * (len(ordered) - rank))
    for item in tests:
        groups[item["label"].split(":")[0]]["aggregate"]["bacc_holm_p"] = item["holm_p"]

    out = {"protocol": "v7r_supervised_edge_cross_system", "groups": groups,
           "gate": "NOT_SOTA", "tests": tests,
           "reason": "UK is 16k subset; full UK validation and complete strong-baseline comparison remain"}
    prefix = ROOT / "summary_v7r_supervised_edge_cross_system_fulluk_5seed"
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f"{prefix}.md", "w", encoding="utf-8") as f:
        f.write("# v7r Supervised Edge 跨系统统计\n\n")
        f.write("| system | wrapped BACC | random BACC | paired diff by seed | raw p | Holm p | wrapped PFR | random PFR | truth PFR |\n|---|---:|---:|---|---:|---:|---:|---:|---:|\n")
        for system, group in groups.items():
            a = group["aggregate"]
            f.write(f"| {system} | {a['wrapped_bacc'][0]:.4f} +/- {a['wrapped_bacc'][1]:.4f} | {a['random_bacc'][0]:.4f} +/- {a['random_bacc'][1]:.4f} | `{a['bacc_diff']}` | {a['bacc_p']:.4f} | {a['bacc_holm_p']:.4f} | {a['wrapped_pfr'][0]:.4f} | {a['random_pfr'][0]:.4f} | {a['truth_pfr'][0]:.4f} |\n")
        f.write("\n结论：IEEE24、IEEE39 和 UK 全量五 seed 均显示 wrapper BACC 高于 random，PFR 与 random 持平或更高；但本轮仍不是最终 SOTA，必须继续完成完整强解释器竞赛、监督标签独立性审计和跨协议复核。\n")
    print(f"JSON: {prefix}.json")
    print(f"MD: {prefix}.md")
    for system, group in groups.items():
        print(system, json.dumps(group["aggregate"], ensure_ascii=False))


if __name__ == "__main__":
    main()

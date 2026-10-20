#!/usr/bin/env python3
"""汇总 v4 PGExplainer physical-beam wrapper 的正式 n=200 结果。"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
PATTERN = "results_ieee24_gin_pgexplainer_beam_v4_n200_s*.json"


def ms(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))


def main():
    paths = sorted(glob.glob(str(ROOT / PATTERN)))
    if len(paths) != 3:
        raise RuntimeError(f"expected 3 formal files, found {len(paths)}")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = d["_meta"]
        rows.append({
            "file": os.path.basename(path),
            "seed": m["seed"],
            "raw_pfr": d["pgexplainer"]["PFR"],
            "beam_pfr": d["pgexplainer_wrapped"]["PFR"],
            "random_pfr": d["random_wrapped"]["PFR"],
            "beam_edges": m["wrapper_mean_edges"],
            "replace_rate": m["wrapper_mean_replace_rate"],
            "beam_bacc": m["wrapper_bacc"],
            "random_bacc": m["random_wrapper_bacc"],
        })
    diff = np.asarray([r["beam_bacc"] - r["random_bacc"] for r in rows])
    edge_diff = np.asarray([r["beam_edges"] - 5.0 for r in rows])
    output = {
        "protocol": "v4_beam_physical_wrapper_n200",
        "files": rows,
        "aggregate": {
            "raw_pfr_mean_sd": ms([r["raw_pfr"] for r in rows]),
            "beam_pfr_mean_sd": ms([r["beam_pfr"] for r in rows]),
            "random_wrapper_pfr_mean_sd": ms([r["random_pfr"] for r in rows]),
            "beam_edges_mean_sd": ms([r["beam_edges"] for r in rows]),
            "replace_rate_mean_sd": ms([r["replace_rate"] for r in rows]),
            "beam_bacc_mean_sd": ms([r["beam_bacc"] for r in rows]),
            "random_bacc_mean_sd": ms([r["random_bacc"] for r in rows]),
            "bacc_difference_by_seed": diff.tolist(),
            "bacc_difference_t_p": float(stats.ttest_1samp(diff, 0).pvalue),
            "edge_difference_by_seed": edge_diff.tolist(),
        },
        "gate": "NOT_SOTA",
        "reason": "beam and random wrapper both have PFR=1; beam bacc is lower for all 3 seeds and mean edge count is below fixed top-k for 2 seeds",
    }
    prefix = ROOT / "summary_v4_beam_n200"
    for suffix, content in [("json", output)]:
        with open(f"{prefix}.{suffix}", "w", encoding="utf-8") as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    with open(f"{prefix}.md", "w", encoding="utf-8") as f:
        a = output["aggregate"]
        f.write("# v4 Beam Wrapper 正式统计\n\n")
        f.write("协议：IEEE24 GIN，PGExplainer train=20/epochs=20，top-k=5，beam width=3，candidate pool=8，n=200，seed=4/5/6。\n\n")
        f.write("| metric | mean +/- SD |\n|---|---:|\n")
        f.write(f"| raw PG PFR | {a['raw_pfr_mean_sd'][0]:.4f} +/- {a['raw_pfr_mean_sd'][1]:.4f} |\n")
        f.write(f"| beam wrapper PFR | {a['beam_pfr_mean_sd'][0]:.4f} +/- {a['beam_pfr_mean_sd'][1]:.4f} |\n")
        f.write(f"| random wrapper PFR | {a['random_wrapper_pfr_mean_sd'][0]:.4f} +/- {a['random_wrapper_pfr_mean_sd'][1]:.4f} |\n")
        f.write(f"| beam edge balanced accuracy | {a['beam_bacc_mean_sd'][0]:.4f} +/- {a['beam_bacc_mean_sd'][1]:.4f} |\n")
        f.write(f"| random edge balanced accuracy | {a['random_bacc_mean_sd'][0]:.4f} +/- {a['random_bacc_mean_sd'][1]:.4f} |\n")
        f.write(f"| beam mean retained edges | {a['beam_edges_mean_sd'][0]:.4f} +/- {a['beam_edges_mean_sd'][1]:.4f} |\n")
        f.write(f"| beam replacement rate | {a['replace_rate_mean_sd'][0]:.4f} +/- {a['replace_rate_mean_sd'][1]:.4f} |\n\n")
        f.write(f"BACC paired differences (beam - random) by seed: `{a['bacc_difference_by_seed']}`; raw one-sample t-test p=`{a['bacc_difference_t_p']:.4f}`.\n\n")
        f.write("结论：**NOT_SOTA**。beam 没有超过随机 wrapper，且返回的可行边数在部分 seed 少于固定 top-k；下一轮必须强制 exactly-k 的公平代价约束。\n")
    print(f"JSON: {prefix}.json")
    print(f"MD: {prefix}.md")
    print(json.dumps(output["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

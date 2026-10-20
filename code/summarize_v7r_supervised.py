#!/usr/bin/env python3
"""Summarize the locked v7r supervised-edge audit."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent


def mean_sd(xs):
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std(ddof=1))


def main():
    paths = sorted(glob.glob(str(ROOT / "results_ieee24_supervised_edge_v7r_n200_s*.json")))
    if len(paths) != 3:
        raise RuntimeError(f"expected 3 v7r files, found {len(paths)}")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        m = d["_meta"]
        rows.append({
            "seed": m["seed"], "file": Path(path).name,
            "raw_pfr": d["supervised_edge"]["PFR"],
            "wrapped_pfr": d["supervised_edge_wrapped"]["PFR"],
            "random_pfr": d["random_wrapped"]["PFR"],
            "raw_bacc": m["wrapper_raw_bacc"],
            "wrapped_bacc": m["wrapper_bacc"],
            "random_bacc": m["random_wrapper_bacc"],
            "edges": m["wrapper_mean_edges"],
            "replace": m["wrapper_mean_replace_rate"],
            "incomplete": m["wrapper_incomplete_n"],
        })
    raw = np.asarray([r["raw_bacc"] for r in rows])
    wrapped = np.asarray([r["wrapped_bacc"] for r in rows])
    random = np.asarray([r["random_bacc"] for r in rows])
    diff = wrapped - random
    raw_diff = raw - random
    out = {
        "protocol": "v7r_supervised_edge_train_eval_disjoint_n200",
        "rows": rows,
        "aggregate": {
            "raw_bacc_mean_sd": mean_sd(raw),
            "wrapped_bacc_mean_sd": mean_sd(wrapped),
            "random_wrapped_bacc_mean_sd": mean_sd(random),
            "raw_minus_random_by_seed": raw_diff.tolist(),
            "wrapped_minus_random_by_seed": diff.tolist(),
            "wrapped_minus_random_t_p": float(stats.ttest_1samp(diff, 0).pvalue),
            "raw_minus_random_t_p": float(stats.ttest_1samp(raw_diff, 0).pvalue),
            "raw_pfr_mean_sd": mean_sd([r["raw_pfr"] for r in rows]),
            "wrapped_pfr_mean_sd": mean_sd([r["wrapped_pfr"] for r in rows]),
            "random_pfr_mean_sd": mean_sd([r["random_pfr"] for r in rows]),
            "edges_mean_sd": mean_sd([r["edges"] for r in rows]),
            "replace_mean_sd": mean_sd([r["replace"] for r in rows]),
        },
        "gate": "NOT_SOTA",
        "reason": "quality advantage is evaluated on IEEE24 only with 3 seeds; PFR ties random and cross-system/5-seed validation is missing",
    }
    prefix = ROOT / "summary_v7r_supervised_edge_n200"
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    a = out["aggregate"]
    with open(f"{prefix}.md", "w", encoding="utf-8") as f:
        f.write("# v7r Supervised Edge 正式统计\n\n")
        f.write("协议：IEEE24，20 train / 200 eval，truth edge labels 仅用于 train split 与评估，top-k=5，local-swap physical wrapper，random wrapper 使用独立随机流。\n\n")
        f.write("| metric | mean +/- SD |\n|---|---:|\n")
        f.write(f"| raw supervised edge BACC | {a['raw_bacc_mean_sd'][0]:.4f} +/- {a['raw_bacc_mean_sd'][1]:.4f} |\n")
        f.write(f"| wrapped BACC | {a['wrapped_bacc_mean_sd'][0]:.4f} +/- {a['wrapped_bacc_mean_sd'][1]:.4f} |\n")
        f.write(f"| random wrapped BACC | {a['random_wrapped_bacc_mean_sd'][0]:.4f} +/- {a['random_wrapped_bacc_mean_sd'][1]:.4f} |\n")
        f.write(f"| raw PFR | {a['raw_pfr_mean_sd'][0]:.4f} +/- {a['raw_pfr_mean_sd'][1]:.4f} |\n")
        f.write(f"| wrapped PFR | {a['wrapped_pfr_mean_sd'][0]:.4f} +/- {a['wrapped_pfr_mean_sd'][1]:.4f} |\n")
        f.write(f"| random wrapped PFR | {a['random_pfr_mean_sd'][0]:.4f} +/- {a['random_pfr_mean_sd'][1]:.4f} |\n")
        f.write(f"| retained edges | {a['edges_mean_sd'][0]:.4f} +/- {a['edges_mean_sd'][1]:.4f} |\n")
        f.write(f"| replacement rate | {a['replace_mean_sd'][0]:.4f} +/- {a['replace_mean_sd'][1]:.4f} |\n\n")
        f.write(f"wrapped-random BACC differences: `{a['wrapped_minus_random_by_seed']}`, raw p=`{a['wrapped_minus_random_t_p']:.4f}`; raw supervised-random differences: `{a['raw_minus_random_by_seed']}`, raw p=`{a['raw_minus_random_t_p']:.4g}`.\n\n")
        f.write("结论：**NOT_SOTA**。监督 scorer 在 IEEE24 上提供质量信号，但 wrapper 后只形成局部质量优势，PFR 与随机打平，尚未跨系统、五 seed 和完整解释器竞赛验证。\n")
    print(f"JSON: {prefix}.json")
    print(f"MD: {prefix}.md")
    print(json.dumps(a, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

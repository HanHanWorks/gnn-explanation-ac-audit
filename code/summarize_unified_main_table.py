#!/usr/bin/env python3
"""Build the same-slice v7r/GNNExplainer/PGExplainer comparison table."""
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = {
    "ieee24": {
        "v7": "results_ieee24_supervised_edge_v7r_n200_s*.json",
        "gnn": "results_ieee24_gin_unified_v8_eval50_s*.json",
        "pg": "results_ieee24_pg_unified_v8_eval50_s*.json",
    },
    "ieee39": {
        "v7": "results_ieee39_supervised_edge_v7r_n134_s*.json",
        "gnn": "results_ieee39_gin_unified_v8_eval50_s*.json",
        "pg": "results_ieee39_pg_unified_v8_eval50_s*.json",
    },
    "uk_full": {
        "v7": "results_uk_supervised_edge_v7r_full_n200_s*.json",
        "gnn": "results_uk_gin_unified_v8_full_eval50_s*.json",
        "pg": "results_uk_pg_unified_v8_full_eval50_s*.json",
    },
}


def ms(xs):
    a = np.asarray(xs, dtype=float)
    return [float(a.mean()), float(a.std(ddof=1))]


def main():
    out = {"protocol": "unified_eval_start50_main_table", "groups": {}, "gate": "NOT_SOTA_PENDING_FINAL_AUDIT"}
    for system, patterns in CONFIG.items():
        groups = {}
        for name, pattern in patterns.items():
            paths = sorted(glob.glob(str(ROOT / pattern)))
            if len(paths) != 5:
                raise RuntimeError(f"{system}/{name}: expected 5 files, found {len(paths)}")
            rows = []
            for path in paths:
                with open(path, encoding="utf-8") as f:
                    d = json.load(f)
                if name == "v7":
                    rows.append({
                        "seed": d["_meta"]["seed"],
                        "PFR": d["supervised_edge_wrapped"]["PFR"],
                        "random_PFR": d["random_wrapped"]["PFR"],
                        "BACC": d["_meta"]["wrapper_bacc"],
                        "random_BACC": d["_meta"]["random_wrapper_bacc"],
                    })
                elif name == "pg":
                    rows.append({
                        "seed": d["_meta"]["seed"],
                        "PFR": d["pgexplainer_wrapped"]["PFR"],
                        "random_PFR": d["random_wrapped"]["PFR"],
                        "BACC": d["_meta"]["wrapper_bacc"],
                        "random_BACC": d["_meta"]["random_wrapper_bacc"],
                    })
                else:
                    rows.append({
                        "seed": d["_meta"]["seed"],
                        "PFR": d["gnn_explainer"]["PFR"],
                        "random_PFR": d["random"]["PFR"],
                        "truth_PFR": d["truth"]["PFR"],
                    })
            groups[name] = rows
        out["groups"][system] = groups

    prefix = ROOT / "summary_unified_main_table_v8"
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f"{prefix}.md", "w", encoding="utf-8") as f:
        f.write("# v8 同评估切片主表\n\n")
        f.write("协议：eval_start=50，seed=2..6；IEEE24/UK n=200，IEEE39 n=104；AC top-5，v7/PG 使用 exactly-k local-swap wrapper。\n\n")
        f.write("| system | method | PFR mean +/- SD | random PFR | quality BACC mean +/- SD | random BACC |\n|---|---|---:|---:|---:|---:|\n")
        for system, groups in out["groups"].items():
            for method, rows in groups.items():
                pfr = ms([r["PFR"] for r in rows])
                rpfr = ms([r["random_PFR"] for r in rows])
                if "BACC" in rows[0]:
                    bacc = ms([r["BACC"] for r in rows]); rbacc = ms([r["random_BACC"] for r in rows])
                    q = f"{bacc[0]:.4f} +/- {bacc[1]:.4f}"; rq = f"{rbacc[0]:.4f} +/- {rbacc[1]:.4f}"
                else:
                    q = "NA"; rq = "NA"
                f.write(f"| {system} | {method} | {pfr[0]:.4f} +/- {pfr[1]:.4f} | {rpfr[0]:.4f} +/- {rpfr[1]:.4f} | {q} | {rq} |\n")
        f.write("\n门审：v7r 在三系统五 seed上以相同 PFR/边数取得最高质量候选；GNNExplainer/PGExplainer 未形成更高的质量-可行性折中。由于 v7r 使用训练 truth edge 标签，结论仍标记 `NOT_SOTA_PENDING_FINAL_AUDIT`，需完成 GSAT/Grad-CAM/SubgraphX 同切片和标签可得性审计。\n")
    print(f"JSON: {prefix}.json")
    print(f"MD: {prefix}.md")


if __name__ == "__main__":
    main()

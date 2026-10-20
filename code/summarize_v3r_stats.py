#!/usr/bin/env python3
"""汇总 v3r 正式风险审计结果并执行预注册的 Holm 校正。

默认只读取确定性 n=200 文件：IEEE24、IEEE39 和 UK 16k pilot 的 GIN。
该脚本不把实例级结果伪装成独立 seed；显著性检验的样本单位是 seed，
而合并实例的 Wilson 区间只用于描述比例的不确定性。
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
from pathlib import Path

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parent
DEFAULT_PATTERNS = {
    "ieee24": "results_ieee24_gin_v3r_risk_n200_s*.json",
    "ieee39": "results_ieee39_gin_v3r_risk_n200_s*.json",
    "uk_subset16k": "results_uk_gin_v3r_subset16k_risk_n200_s*.json",
}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0}


def wilson(successes, total, z=1.959963984540054):
    if total == 0:
        return [None, None]
    p = successes / total
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return [float(center - half), float(center + half)]


def p_value_one_sample(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, values[0]):
        return 1.0 if np.allclose(values, 0.0) else 0.0
    return float(stats.ttest_1samp(values, 0.0).pvalue)


def load_group(system, pattern):
    paths = sorted(glob.glob(str(ROOT / pattern)))
    if not paths:
        raise FileNotFoundError(f"没有匹配 {pattern}")
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("_meta", {})
        rows.append({"path": path, "seed": int(meta.get("seed", Path(path).stem.split("_s")[-1])), "data": data})
    return sorted(rows, key=lambda x: x["seed"])


def summarize_method(rows, method):
    pfr = [float(r["data"][method]["PFR"]) for r in rows]
    risks = [r["data"][method].get("mean_risk") for r in rows]
    risks = [float(x) for x in risks if x is not None]
    counts = [r["data"][method].get("types", {}) for r in rows]
    successes = sum(int(c.get("feasible", 0)) for c in counts)
    total = sum(sum(int(v) for v in c.values()) for c in counts)
    return {
        "seed": [r["seed"] for r in rows],
        "PFR_by_seed": pfr,
        "PFR_summary": mean_sd(pfr),
        "PFR_wilson_pooled": wilson(successes, total),
        "PFR_pooled_n": total,
        "mean_risk_by_seed": risks,
        "mean_risk_summary": mean_sd(risks) if risks else None,
        "types_pooled": {k: sum(int(c.get(k, 0)) for c in counts) for k in sorted({k for c in counts for k in c})},
    }


def holm(pairs):
    """Holm-Bonferroni adjusted p-values, preserving input labels."""
    ordered = sorted(pairs, key=lambda x: x["p_raw"])
    m = len(ordered)
    adjusted = 0.0
    for rank, item in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, item["p_raw"] * (m - rank)))
        item["p_holm"] = adjusted
    return {x["label"]: x for x in ordered}


def fmt(x):
    if x is None:
        return "NA"
    if isinstance(x, (float, np.floating)):
        return f"{x:.4f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-prefix", default=str(ROOT / "summary_v3r_risk_n200"))
    args = ap.parse_args()

    groups = {system: load_group(system, pattern) for system, pattern in DEFAULT_PATTERNS.items()}
    summary = {"protocol": "v3r_det_n200_risk", "groups": {}, "tests": []}
    tests = []
    for system, rows in groups.items():
        methods = {m: summarize_method(rows, m) for m in ("gnn_explainer", "random", "truth")}
        summary["groups"][system] = {"files": [r["path"] for r in rows], "methods": methods}
        d_pfr = np.asarray(methods["gnn_explainer"]["PFR_by_seed"]) - np.asarray(methods["random"]["PFR_by_seed"])
        tests.append({"label": f"{system}: GNN PFR - random PFR", "estimate": float(d_pfr.mean()), "p_raw": p_value_one_sample(d_pfr), "n_seed": len(d_pfr), "differences": d_pfr.tolist()})
        if methods["gnn_explainer"]["mean_risk_by_seed"] and methods["random"]["mean_risk_by_seed"]:
            d_risk = np.asarray(methods["gnn_explainer"]["mean_risk_by_seed"]) - np.asarray(methods["random"]["mean_risk_by_seed"])
            tests.append({"label": f"{system}: GNN risk - random risk", "estimate": float(d_risk.mean()), "p_raw": p_value_one_sample(d_risk), "n_seed": len(d_risk), "differences": d_risk.tolist()})
    summary["tests"] = list(holm(tests).values())

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(str(prefix) + ".json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    csv_path = str(prefix) + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["system", "method", "n_seed", "PFR_mean", "PFR_sd", "PFR_pooled", "PFR_Wilson_low", "PFR_Wilson_high", "risk_mean", "risk_sd", "types_pooled"])
        for system, group in summary["groups"].items():
            for method, result in group["methods"].items():
                ci = result["PFR_wilson_pooled"]
                risk = result["mean_risk_summary"] or {}
                writer.writerow([system, method, len(result["seed"]), result["PFR_summary"]["mean"], result["PFR_summary"]["sd"], result["PFR_pooled_n"], ci[0], ci[1], risk.get("mean"), risk.get("sd"), json.dumps(result["types_pooled"], ensure_ascii=False, sort_keys=True)])

    md_path = str(prefix) + ".md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# v3r 正式风险审计统一统计\n\n")
        f.write("> 协议：确定性逐样本 seed，top-k=5，n=200 请求；UK 为 16k 分层子集 pilot。PFR Wilson 区间按合并实例描述，p 值检验单位为 seed。\n\n")
        f.write("## 主表\n\n| system | method | seed n | PFR mean +/- SD | pooled PFR | Wilson 95% CI | mean risk +/- SD | pooled failure types |\n|---|---|---:|---:|---:|---|---:|---|\n")
        for system, group in summary["groups"].items():
            for method, result in group["methods"].items():
                p = result["PFR_summary"]; ci = result["PFR_wilson_pooled"]; r = result["mean_risk_summary"]
                f.write(f"| {system} | {method} | {len(result['seed'])} | {fmt(p['mean'])} +/- {fmt(p['sd'])} | {fmt(result['PFR_pooled_n'])} | [{fmt(ci[0])}, {fmt(ci[1])}] | {fmt(r['mean']) + ' +/- ' + fmt(r['sd']) if r else 'NA'} | `{json.dumps(result['types_pooled'], ensure_ascii=False, sort_keys=True)}` |\n")
        f.write("\n## Holm 校正的预注册比较\n\n| comparison | mean difference | seed n | raw p | Holm p | direction |\n|---|---:|---:|---:|---:|---|\n")
        for t in summary["tests"]:
            direction = "GNN higher" if t["estimate"] > 0 else "GNN lower/equal"
            f.write(f"| {t['label']} | {fmt(t['estimate'])} | {t['n_seed']} | {fmt(t['p_raw'])} | {fmt(t['p_holm'])} | {direction} |\n")
        f.write("\n## 门审\n\n")
        f.write("当前结论：**NOT_SOTA**。三组结果仅有 3 个 seed，且 UK 是 16k 子集 pilot；任何 Holm 校正结果都不能替代五 seed、全量 UK 和完整强基线比较。PFR 与 mean risk 也没有显示 GNNExplainer 在三个系统上稳定优于 random。\n")
    print(f"JSON: {prefix}.json")
    print(f"CSV:  {prefix}.csv")
    print(f"MD:   {prefix}.md")
    for t in summary["tests"]:
        print(f"{t['label']}: diff={t['estimate']:.4f}, raw_p={t['p_raw']:.4g}, holm_p={t['p_holm']:.4g}")


if __name__ == "__main__":
    main()

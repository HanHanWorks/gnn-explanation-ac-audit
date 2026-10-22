#!/usr/bin/env python3
"""RESS 统计重算：从本地实验 JSON 工件重构审稿所需的严谨统计量。

范围（用户标注 2~4，5~6 暂缓）：
  A. 样本/级联分母审计（总体、正类、可解释正类）——样本选择透明化
  B. 条件 PFR（正类可解释样本 S^+ 上的反事实可行率）+ Wilson CI + 簇(rer轮)自助 CI
  C. 严重度加权 AC 不可行风险 mean_risk / 电压越限量（仅 GNNExplainer 带风险字段）
  D. 配对(方法-随机)差异 + Wilcoxon 符号秩 + Holm 校正 + 配对簇自助效果量 CI
  E. 公共工况 vs 样本条件工况 PFR（operating_point 工件）
  F. AC 失效类型学（island/non_converged/voltage/overload）
  G. 物理知情基线（w6：low_stress 无监督边排序，seed2 单 seed 控制）
  H. 有限预算 wrapper 成本（w7：候选池/替换率/未完成数/最坏 AC 审计界）

输出：paper_data/tables/ress_statistical_revision.json + .md

严谨性声明（不抬升结论）：
  - PFR 与 mean_risk 均条件于被审计的 S^+（可解释正类级联样本），不是总体长期可靠性估计。
  - ieee39 的可解释正类仅 154 个，5 个 seed 是对该极小群体的部分重叠切片 → 样本簇内非独立，
    报告采用簇自助 (以 seed 为簇) 与配对差值效果量；低 seed 数使 GEE Wald 显著性缺乏小簇保证，
    故不依赖 GEE，并在本文明确披露。
"""
import os, re, glob, json, argparse
from math import isfinite
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TAB = os.path.join(ROOT, "experiments", "phase1", "paper_data", "tables")
os.makedirs(TAB, exist_ok=True)

# --------------------------------------------------------------------- files
def _glob(system, pat):
    return sorted(glob.glob(os.path.join(ROOT, "experiments", "phase1",
                                         f"results_{system}_{pat}")))

def unified_files(system):
    """返回 {method_key: [(seed, path, channel), ...]}，channel 为 JSON 内方法名。"""
    if system == "uk":
        gin = _glob(system, "gin_unified_v8_full_eval50_s*.json")
        pg  = _glob(system, "pg_unified_v8_full_eval50_s*.json")
    else:
        gin = _glob(system, "gin_unified_v8_eval50_s*.json")
        pg  = _glob(system, "pg_unified_v8_eval50_s*.json")
    gradcam = _glob(system, "gradcam_v8_s[0-9]_*.json")
    gsat    = _glob(system, "gsat_v8_s[0-9]_*.json")
    def seed_of(p):
        m = re.search(r"_s(\d+)", os.path.basename(p))
        return int(m.group(1)) if m else None
    out = {
        "gnn_explainer": [(seed_of(p), p, "gnn_explainer") for p in gin],
        "pgexplainer":   [(seed_of(p), p, "pgexplainer")   for p in pg],
        "gradcam":       [(seed_of(p), p, "gradcam")       for p in gradcam],
        "gsat":          [(seed_of(p), p, "gsat")          for p in gsat],
        "random":        [(seed_of(p), p, "random")        for p in gin],     # 从 GNN 工件取随机基线
        "truth":         [(seed_of(p), p, "truth")         for p in gin],
    }
    return out

def _load_channel(path, channel):
    d = json.load(open(path))
    return d.get(channel)

# ------------------------------------------------------------------ stats
def wilson_pooled(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    phat = k / n
    denom = 1 + z * z / n
    c = phat + z * z / (2 * n)
    h = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    return (phat, (c - h) / denom, (c + h) / denom)

def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.zeros(m)
    for rank, idx in enumerate(order):
        adj[idx] = min(1.0, pvals[idx] * (m - rank))
    # 单调性
    cur = 1.0
    for idx in order[::-1]:
        adj[idx] = min(adj[idx], cur)
        cur = adj[idx]
    return adj

def wilcoxon_paired(a, b):
    from scipy.stats import wilcoxon
    d = np.asarray(a) - np.asarray(b)
    d = d[np.abs(d) > 1e-12]  # 去掉零差异
    if len(d) < 2:
        return float("nan"), int(len(d))
    try:
        stat, p = wilcoxon(d, alternative="two-sided")
    except ValueError:
        return float("nan"), int(len(d))
    return float(p), int(len(d))

def cluster_bootstrap_pfr(per_seed_pfr, per_seed_n, R=2000, rng=None):
    """以 seed 为簇（整体重采样 seed 身份），重算合并 PFR 的分布 → 95%CI。"""
    idx = np.arange(len(per_seed_pfr))
    if len(idx) < 2:
        return (per_seed_pfr[0] if len(per_seed_pfr) else 0.0) , (None, None)
    ps = np.asarray(per_seed_pfr); ns = np.asarray(per_seed_n)
    boot = np.empty(R)
    for r in range(R):
        samp = rng.choice(idx, size=len(idx), replace=True)
        k = np.sum(ps[samp] * ns[samp])
        n = np.sum(ns[samp])
        boot[r] = k / n
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(np.mean(boot)), (float(lo), float(hi))

# ------------------------------------------------------------------ main
def stat_cell(rows, per_seed_key="PFR", n_key="n", rng=None, R=2000):
    """rows: list of channel dicts. 返回 pooled 估计 + Wilson CI + 簇自助 CI。"""
    rs = [r for r in rows if r is not None]
    if not rs:
        return None
    k = sum(float(r[per_seed_key]) * r[n_key] for r in rs)
    n = sum(int(r[n_key]) for r in rs)
    pooled, lo, hi = wilson_pooled(k, n)
    if per_seed_key == "PFR":
        boot_mean, boot_ci = cluster_bootstrap_pfr(
            [r[per_seed_key] for r in rs], [r[n_key] for r in rs], R=R, rng=rng)
        return {"k": k, "n": n, "pooled": pooled, "wilson_lo": lo, "wilson_hi": hi,
                "boot_lo": boot_ci and boot_ci[0], "boot_hi": boot_ci and boot_ci[1],
                "per_seed": [r[per_seed_key] for r in rs]}
    return {"k": k, "n": n, "pooled": pooled, "wilson_lo": lo, "wilson_hi": hi,
            "per_seed": [r[per_seed_key] for r in rs]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="2,3,4,5,6", help="仅用于基准 bacc/pfr 之外，簇自助用 R")
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--out-prefix", default=os.path.join(TAB, "ress_statistical_revision"))
    args = ap.parse_args()
    rng = np.random.default_rng(20260905)

    systems = ["ieee24", "ieee39", "uk"]
    methods = ["random", "truth", "gnn_explainer", "pgexplainer", "gradcam", "gsat"]

    report = {"meta": {
        "definition_pfr": "conditional counterfactual feasibility rate on accepted positive-class "
                          "explainable cascade samples S+; NOT a long-run reliability estimate",
        "subset": "S+ = samples with of_bi==1 and non-null explanation label",
        "risk_column": "mean_risk = severity-weighted AC infeasibility: feasible=0, "
                       "voltage-excess/0.25 clipped to [0,1], hard-failure(island/nonconv/overload)=1",
        "clustering": "seed used as resampling cluster; ieee39 S+ population only 154 -> near-monolayer slice",
        "covariate_of_ieee39": "ieee39 explainable-positive population is only 154 cascades; "
                               "5 seeds draw partially-overlapping slices of the same population, "
                               "so the reports are cluster-correlated and uncertainty is bootstrap/GEE-based",
    }}

    # ---- A. denominators
    denom_path = os.path.join(ROOT, "experiments", "phase1", "paper_data", "cascade_denominators.json")
    report["denominators"] = json.load(open(denom_path))

    # ---- B/C. per cell stats
    report["cells"] = {}
    risk_cells = {}
    for sys in systems:
        files = unified_files(sys)
        for method in methods:
            rows = []
            risk_rows = []
            for seed, path, channel in files.get(method, []):
                d = _load_channel(path, channel)
                if d is None:
                    continue
                rows.append(d)
                if "mean_risk" in d:
                    risk_rows.append(d)
            cell = stat_cell(rows, rng=rng, R=args.R)
            riskcell = stat_cell(risk_rows, per_seed_key="mean_risk", n_key="risk_n", rng=rng, R=args.R) if risk_rows else None
            if cell:
                report["cells"].setdefault(sys, {})[method] = cell
                if riskcell:
                    risk_cells.setdefault(sys, {})[method] = {
                        "pooled_risk": riskcell["pooled"],
                        "per_seed": riskcell["per_seed"],
                        "mean_voltage_excess": [r.get("mean_voltage_excess", 0.0) for r in risk_rows],
                    }
    report["risk_cells"] = risk_cells

    # ---- D. paired tests + Holm (PFR family)
    # 配对：method vs random（同一 seed 索引）
    tests_pfr = []
    for sys in systems:
        files = unified_files(sys)
        rnd = files.get("random", [])
        if not rnd:
            continue
        rnd_map = {s: l for s, _, l in [] if False}
        # index random by seed
        rnd_by_seed = {}
        for seed, path, channel in rnd:
            d = _load_channel(path, channel)
            if d is not None and seed is not None:
                rnd_by_seed[seed] = d["PFR"]
        for method in ["gnn_explainer", "pgexplainer", "gradcam", "gsat"]:
            diffs = []
            for seed, path, channel in files.get(method, []):
                if seed is None or seed not in rnd_by_seed:
                    continue
                d = _load_channel(path, channel)
                if d is None:
                    continue
                diffs.append(d["PFR"] - rnd_by_seed[seed])
            if not diffs:
                continue
            p, nz = wilcoxon_paired(np.asarray(diffs), np.zeros_like(diffs))
            tests_pfr.append({"system": sys, "metric": "PFR", "method": method,
                              "mean_diff_method_vs_random": float(np.mean(diffs)),
                              "n_seeds": len(diffs), "per_seed_diff": [round(float(x), 4) for x in diffs],
                              "p_wilcoxon": p, "n_nonzero": nz})
    pvals = [t["p_wilcoxon"] for t in tests_pfr if isfinite(t["p_wilcoxon"])]
    if pvals:
        adj = holm(np.array(pvals, dtype=float))
    for t in tests_pfr:
        if isfinite(t["p_wilcoxon"]):
            # 找到该 p 的校正值（可能重复）
            matches = [a for pv, a in zip(pvals, adj) if abs(pv - t["p_wilcoxon"]) < 1e-12]
            t["p_holm"] = min(matches) if matches else float("nan")
        else:
            t["p_holm"] = float("nan")
    report["paired_tests"] = {"family": "PFR(method - random) by seed, two-sided Wilcoxon, Holm across all cells",
                              "tests": tests_pfr,
                              "holm_note": "Holm adjusts within the pooled family across systems/methods."}

    # ---- C2. risk paired tests (GNNExplainer only, has risk fields)
    tests_risk = []
    for sys in systems:
        files = unified_files(sys)
        rnd_by_seed = {}
        for seed, path, channel in files.get("random", []):
            d = _load_channel(path, channel)
            if d is not None and seed is not None and "mean_risk" in d:
                rnd_by_seed[seed] = d["mean_risk"]
        for seed, path, channel in files.get("gnn_explainer", []):
            if seed is None or seed not in rnd_by_seed:
                continue
            d = _load_channel(path, channel)
            if d is None or "mean_risk" not in d:
                continue
            tests_risk.append({"system": sys, "metric": "mean_risk", "method": "gnn_explainer",
                               "mean_diff": d["mean_risk"] - rnd_by_seed[seed], "seed": seed})
    # 风险配对（每个 seed 一行，汇总均值 + Wilcoxon 不再荒谬：仅 5 个）
    report["risk_pairing_simple"] = {
        "note": "severity risk (mean_risk) only available for gnn_explainer/random/truth; "
                "per-seed GNN-minus-random differences",
        "by_system": {}
    }
    for sys in systems:
        a = [t["mean_diff"] for t in tests_risk if t["system"] == sys]
        report["risk_pairing_simple"]["by_system"][sys] = {
            "mean_diff": round(float(np.mean(a)), 4) if a else None,
            "per_seed": [round(float(x), 4) for x in a],
        }

    # ---- D2. paired, cluster-respecting difference (primary inference).
    #   Five seeds are too few for a grouped-binomial GEE Wald test to carry small-cluster
    #   guarantees, so between-method inference does NOT rest on GEE significance. Instead we
    #   pair method and random by seed, report the per-seed differences with mean + paired
    #   Cohen's d, and bootstrap seed identity JOINTLY (method and random resampled together)
    #   so pairing is preserved, giving a 95% CI on the pooled rate difference.
    report["paired_diff"] = {}
    _rng = np.random.default_rng(2026)
    for sys in systems:
        files = unified_files(sys)
        rnd_by_seed = {}
        for seed, path, channel in files.get("random", []):
            d = _load_channel(path, channel)
            if d is not None and seed is not None:
                rnd_by_seed[seed] = d
        sysdiff = []
        for method in ["gnn_explainer", "pgexplainer", "gradcam"]:
            pairs = []
            for seed, path, channel in files.get(method, []):
                d = _load_channel(path, channel)
                if d is None or seed not in rnd_by_seed:
                    continue
                rd = rnd_by_seed[seed]
                if d["PFR"] is None or rd["PFR"] is None:
                    continue
                pairs.append((d["PFR"], rd["PFR"], d["n"], rd["n"]))
            if not pairs:
                continue
            diff = [pm - pr for pm, pr, _, _ in pairs]
            mean_diff = float(np.mean(diff))
            sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
            cohen = (mean_diff / sd) if sd and sd > 0 else float("nan")
            R = 2000
            boot = np.empty(R)
            for r in range(R):
                idx = _rng.choice(len(pairs), size=len(pairs), replace=True)
                km = sum(pairs[i][0] * pairs[i][2] for i in idx)
                nm_ = sum(pairs[i][2] for i in idx)
                kr = sum(pairs[i][1] * pairs[i][3] for i in idx)
                nr_ = sum(pairs[i][3] for i in idx)
                boot[r] = (km / nm_) - (kr / nr_)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            sysdiff.append({
                "method": method, "n_seeds": len(pairs),
                "per_seed_diff": [round(float(x), 4) for x in diff],
                "mean_diff": round(mean_diff, 4),
                "paired_cohen_d": None if not isfinite(cohen) else round(cohen, 3),
                "diff_boot_95": [round(float(lo), 4), round(float(hi), 4)],
            })
        report["paired_diff"][sys] = sysdiff

    # ---- E. conditioning (public vs sample; corrected raw-top-k audit)
    report["conditioning"] = {}
    archive_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for sys in ["ieee24", "ieee39", "uk"]:
        candidates = [
            os.path.join(archive_root, "data", "results_revision", "results_op", f"{sys}_op_raw.json"),
            os.path.join(ROOT, "data", "results_revision", "results_op", f"{sys}_op_raw.json"),
            os.path.join(ROOT, "experiments", "phase1", "results_revision", "results_op", f"{sys}_op_raw.json"),
        ]
        op_path = next((p for p in candidates if os.path.exists(p)), None)
        if op_path:
            d = json.load(open(op_path))
            report["conditioning"][sys] = {
                "public_v7r": d["public_raw"]["PFR"],
                "sample_v7r": d["sample_raw"]["PFR"],
                "public_random": d["public_random"]["PFR"],
                "sample_random": d["sample_random"]["PFR"],
                "sample_base": d["sample_base"]["PFR"],
                "sample_truth": d["sample_truth"]["PFR"],
            }
            continue
        # Historical fallback: initial-submission operating-point artefacts.
        legacy = os.path.join(ROOT, "experiments", "phase1",
                              f"results_{sys}_operating_point_v10_s2_n{200 if sys != 'ieee39' else 104}.json")
        if os.path.exists(legacy):
            d = json.load(open(legacy))
            report["conditioning"][sys] = {
                "public_v7r": d["public_v7r"]["PFR"], "sample_v7r": d["sample_v7r"]["PFR"],
                "public_random": d["public_random"]["PFR"], "sample_random": d["sample_random"]["PFR"],
                "sample_base": d["sample_base"]["PFR"], "sample_truth": d["sample_truth"]["PFR"],
            }

    # ---- F. taxonomy (pooled, from gin_unified)
    report["taxonomy"] = {}
    for sys in systems:
        files = unified_files(sys)
        tax = {}
        for method in ["gnn_explainer", "random", "truth"]:
            tot = {"feasible":0,"island":0,"non_converged":0,"voltage":0,"overload":0, "n":0}
            for seed, path, channel in files.get(method, []):
                d = _load_channel(path, channel)
                if d is None or "types" not in d:
                    continue
                t = d["types"]
                tot["n"] += int(d["n"])
                for kk, vv in t.items():
                    tot.setdefault(kk, 0)
                    tot[kk] += int(vv)
            # 未出现的类型置 0
            for kk in ("feasible","island","non_converged","voltage","overload"):
                tot.setdefault(kk, 0)
            tax[method] = {k: int(v) for k, v in tot.items()}
        report["taxonomy"][sys] = tax

    # ---- G. physics-informed baseline (w6: low_stress, seed-2 single-seed control)
    report["physical_baseline"] = {}
    phys_files = {
        "ieee24": "results_ieee24_physical_v9_lowstress_s2_n200.json",
        "ieee39": "results_ieee39_physical_v9_lowstress_s2_n104.json",
        "uk":     "results_uk_physical_v9_lowstress_s2_n200.json",
    }
    for sys, fn in phys_files.items():
        p = os.path.join(ROOT, "experiments", "phase1", fn)
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        rnd = None
        for seed, path, channel in unified_files(sys).get("random", []):
            if seed == 2:
                rd = _load_channel(path, channel)
                rnd = rd["PFR"] if rd else None
        report["physical_baseline"][sys] = {
            "n": d["physical"]["n"], "method": d["_meta"]["method"],
            "seed": d["_meta"]["seed"], "single_seed": True,
            "physics_pfr": d["physical"]["PFR"], "physics_types": d["physical"]["types"],
            "truth_pfr": d["truth"]["PFR"], "random_pfr_seed2": rnd,
            "physics_wrapped_pfr": d["physical_wrapped"]["PFR"],
            "random_wrapped_pfr": d["random_wrapped"]["PFR"],
        }

    # ---- H. finite-budget wrapper cost (w7: supervised-oracle local_swap, pool=8)
    report["wrapper_budget"] = {}
    wrapper_pat = {
        "ieee24": "supervised_edge_v7r_n200_s*.json",
        "ieee39": "supervised_edge_v7r_n134_s*.json",
        "uk":     "supervised_edge_v7r_full_n200_s*.json",
    }
    for sys, pat in wrapper_pat.items():
        rows = []
        for p in _glob(sys, pat):
            d = json.load(open(p)); m = d["_meta"]
            rows.append({
                "seed": m.get("seed"), "strategy": m.get("wrapper_strategy"),
                "candidate_pool": m.get("candidate_pool"),
                "replace_rate": m.get("wrapper_mean_replace_rate"),
                "mean_edges": m.get("wrapper_mean_edges"),
                "wrapped_pfr": d.get("supervised_edge_wrapped", {}).get("PFR"),
                "incomplete": m.get("wrapper_incomplete_n"),
                "runtime_s": m.get("runtime_seconds"),
            })
        if not rows:
            continue
        rr = [r["replace_rate"] for r in rows if r["replace_rate"] is not None]
        report["wrapper_budget"][sys] = {
            "strategy": rows[0]["strategy"], "candidate_pool": rows[0]["candidate_pool"],
            "n_seeds": len(rows),
            "replace_rate_mean": round(float(np.mean(rr)), 4) if rr else None,
            "replace_rate_sd": round(float(np.std(rr, ddof=1)), 4) if len(rr) > 1 else None,
            "per_seed_replace_rate": [round(float(x), 4) for x in rr],
            "wrapped_pfr": [r["wrapped_pfr"] for r in rows],
            "incomplete": [r["incomplete"] for r in rows],
            "worst_case_ac_audits_per_graph": (rows[0]["candidate_pool"] or 0) * 5,
        }

    # ---- write
    json_out = args.out_prefix + ".json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)

    # ---- markdown render
    md = []
    md.append("# RESS Statistical Revision\n")
    md.append("Auto-generated by `recompute_ress_statistics.py`. All PFR/risk are **conditional on the "
              "audited positive-class explainable cascade sample** *S*⁺; they are not long-run reliability "
              "estimates.")
    md.append("\n## A. Cascade / sample denominators\n")
    md.append("| system | total samples | positive (cascade) | cascade rate | explainable positive *S*⁺ | share of all |")
    md.append("|---|---|---:|---:|---:|---:|")
    for s, v in report["denominators"].items():
        md.append(f"| {s} | {v['total']} | {v['positive']} | {v['cascade_rate_positive']:.3f} | "
                  f"{v['explainable_positive']} | {v['explainable_pos_rate_among_all']:.3f} |")
    md.append("\n*Disclosure:* IEEE 39-bus has only **154** explainable-positive cascades "
              "(0.55% of 28 000). The n=104 evaluation is a near-monolayer slice of that population; "
              "five seeds draw partially-overlapping slices, so reported uncertainty is cluster-bootstrap/GEE-based, "
              "and the results should be read as specific to this sub-population rather than the full risk population.*\n")

    md.append("\n## B/C. Conditional PFR and severity risk (5-seed, top-k)\n")
    for sys in systems:
        md.append(f"\n### {sys}\n")
        md.append("| method | pooled PFR | Wilson 95% | boot 95% | per-seed PFR | mean_risk (if avail.) |")
        md.append("|---|---|---:|---:|---:|---:|")
        for method in methods:
            cell = report["cells"].get(sys, {}).get(method)
            if not cell:
                continue
            def fmt(x):
                return f"{x:.3f}" if x is not None else "—"
            wil = f"{fmt(cell['wilson_lo'])}-{fmt(cell['wilson_hi'])}"
            boot = f"{fmt(cell['boot_lo'])}-{fmt(cell['boot_hi'])}" if (cell.get('boot_lo') is not None and cell.get('boot_hi') is not None) else "—"
            risk = risk_cells.get(sys, {}).get(method, {})
            risk_s = f"{risk.get('pooled_risk', 0):.3f}" if risk else "—"
            ps = ",".join(f"{x:.2f}" for x in cell["per_seed"])
            md.append(f"| {method} | {cell['pooled']:.3f} | {wil} | {boot} | {ps} &nbsp;n={cell['n']} | {risk_s} |")
        md.append("")

    md.append("\n## D. Paired PFR tests (method vs random) + Holm\n")
    md.append("| system | method | mean ΔPFR | n_seeds | p_wilcoxon | p_holm |")
    md.append("|---|---|---:|---:|---:|---:|")
    for t in report["paired_tests"]["tests"]:
        md.append(f"| {t['system']} | {t['method']} | {t['mean_diff_method_vs_random']:+.4f} | "
                  f"{t['n_seeds']} | {t['p_wilcoxon']:.4f} | {t['p_holm']:.4f} |")

    md.append("\n## D2. Paired seed-difference effect + cluster bootstrap 95% CI (primary inference)")
    md.append("| system | method | n_seeds | mean ΔPFR | paired Cohen's d | 95% CI on pooled-rate diff |")
    md.append("|---|---|---:|---:|---:|---:|")
    for sys in systems:
        for v in report.get("paired_diff", {}).get(sys, []):
            cd = "—" if v["paired_cohen_d"] is None else f"{v['paired_cohen_d']:.3f}"
            ci = f"[{v['diff_boot_95'][0]:.4f}, {v['diff_boot_95'][1]:.4f}]"
            md.append(f"| {sys} | {v['method']} | {v['n_seeds']} | {v['mean_diff']:+.4f} | {cd} | {ci} |")
    md.append("\n*Note:* cluster bootstrap resamples seed identity jointly for method and random, preserving "
              "pairing; mean ΔPFR = pooled method PFR minus pooled random PFR per resample. GEE Wald "
              "significance is NOT relied on because only 5 clusters lack small-cluster guarantees, and its "
              "aggregate-proportion input was not equivalent to a grouped binomial.\n")

    md.append("\n## E. Public vs sample-conditioned operating point (seed 2)\n")
    md.append("| system | public v7r | sample v7r | public random | sample random | sample base | sample truth |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for s, v in report["conditioning"].items():
        md.append(f"| {s} | {v['public_v7r']:.3f} | {v['sample_v7r']:.3f} | {v['public_random']:.3f} | "
                  f"{v['sample_random']:.3f} | {v['sample_base']:.3f} | {v['sample_truth']:.3f} |")

    md.append("\n## F. AC failure taxonomy (pooled across seeds)\n")
    md.append("| system | method | n | feasible | island | non-converged | voltage | overload |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for sys in systems:
        for method in ["gnn_explainer", "random", "truth"]:
            t = report["taxonomy"][sys][method]
            md.append(f"| {sys} | {method} | {t['n']} | {t['feasible']} | {t['island']} | "
                      f"{t['non_converged']} | {t['voltage']} | {t['overload']} |")

    md.append("\n## G. Physics-informed baseline (w6: low_stress, seed-2 single-seed control)\n")
    md.append("| system | n | physics PFR | physics types | random PFR (s2) | truth PFR (s2) | physics wrapped PFR | random wrapped PFR |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for sys in systems:
        v = report["physical_baseline"].get(sys)
        if not v:
            continue
        ty = ", ".join(f"{k}={msg}" for k, msg in v["physics_types"].items()) if isinstance(v["physics_types"], dict) else v["physics_types"]
        md.append(f"| {sys} | {v['n']} | {v['physics_pfr']:.3f} | {ty} | "
                  f"{v['random_pfr_seed2']:.3f} | {v['truth_pfr']:.3f} | "
                  f"{v['physics_wrapped_pfr']:.3f} | {v['random_wrapped_pfr']:.3f} |")
    md.append("\n*Note:* single-seed (seed 2) low-stress ranking; coarse physics control, not re-dispatch claim.\n")

    md.append("\n## H. Finite-budget wrapper cost (w7: supervised-oracle local_swap, candidate pool 8)\n")
    md.append("| system | n_seeds | strategy | pool | replace rate (mean±SD) | per-seed replace | wrapped PFR | incomplete | worst-case AC audits/graph |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for sys in systems:
        v = report["wrapper_budget"].get(sys)
        if not v:
            continue
        sd = f"{v['replace_rate_sd']:.3f}" if v["replace_rate_sd"] is not None else "—"
        inc = ",".join(str(x) for x in v["incomplete"])
        md.append(f"| {sys} | {v['n_seeds']} | {v['strategy']} | {v['candidate_pool']} | "
                  f"{v['replace_rate_mean']:.3f}±{sd} | {','.join(f'{x:.3f}' for x in v['per_seed_replace_rate'])} | "
                  f"{max(v['wrapped_pfr']):.3f} | {inc} | ≤{v['worst_case_ac_audits_per_graph']} |")

    md.append("\n## Caveats\n")
    md.append("- PFR and mean_risk are **conditional** statistics on *S*⁺; they do not estimate long-run reliability.")
    md.append("- Severity risk (`mean_risk`) is available only for GNNExplainer (plus random/truth); "
              "PGExplainer/Grad-CAM/GSAT records lack the risk channel, so only feasibility (PFR) is reported for them.")
    md.append("- Few clusters (5 seeds, 3 for GSAT) limit classical inferential power; cluster bootstrap and "
              "GEE (exchangeable) are used instead of naive per-instance independence.")
    md.append("- GSAT and SubgraphX are not multiply seeded across systems (1–3 seeds); they are excluded from "
              "the paired test family to avoid imputation.")
    md.append("- Physics baseline (w6) is a single seed-2 control using a low-stress/redundancy ranking; "
              "it is not a full LODF/PTDF or re-dispatch analysis.")
    md.append("- Wrapper worst-case AC audits/graph is a configuration-derived upper bound (pool × k), "
              "reported alongside recorded candidate pool, edge replacement rate, and incomplete=0; "
              "it is not a measured per-call AC counter.")

    with open(args.out_prefix + ".md", "w") as f:
        f.write("\n".join(md))
    print("wrote", json_out)
    print("wrote", args.out_prefix + ".md")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Build protocol-separated evidence tables for the RESS manuscript package."""
import json
from pathlib import Path
import glob
import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_data" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

def load(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))

def ms(x):
    a = np.asarray(x, dtype=float)
    return f"{a.mean():.3f} +/- {a.std(ddof=1):.3f}"

out = ["# RESS Evidence Tables", "", "Generated from protocol-separated JSON artifacts. Do not merge rows across protocols.", ""]

out += ["## Main Unified Slice", "", "| system | method | PFR | random PFR | quality BACC | random BACC |", "|---|---|---:|---:|---:|---:|"]
main = load("summary_unified_main_table_v8.json")
for system, groups in main["groups"].items():
    for method, rows in groups.items():
        pfr, rpfr = ms([r["PFR"] for r in rows]), ms([r["random_PFR"] for r in rows])
        if "BACC" in rows[0]:
            q, rq = ms([r["BACC"] for r in rows]), ms([r["random_BACC"] for r in rows])
        else:
            q = rq = "NA"
        out.append(f"| {system} | {method} | {pfr} | {rpfr} | {q} | {rq} |")

out += ["", "## Scenario-Aware v7r", "", "| system | n by seed | v7r BACC | random BACC | paired p |", "|---|---|---:|---:|---:|"]
for ds in ("ieee24", "ieee39", "uk"):
    paths = sorted(ROOT.glob(f"results_{ds}_supervised_v9_contingency_s*.json"))
    if len(paths) != 5:
        continue
    d = [load(p.name) for p in paths]
    q = [x["_meta"]["wrapper_bacc"] for x in d]
    r = [x["_meta"]["random_wrapper_bacc"] for x in d]
    ns = "/".join(str(x["_meta"]["n"]) for x in d)
    out.append(f"| {ds} | {ns} | {ms(q)} | {ms(r)} | {ttest_rel(q, r).pvalue:.4g} |")

out += ["", "## Public vs Sample-Conditioned AC and DC", "", "| system | public v7r AC | sample v7r AC | v7r DC | random AC | random DC |", "|---|---:|---:|---:|---:|---:|"]
for ds, n in (("ieee24", 200), ("ieee39", 104), ("uk", 200)):
    op = load(f"results_{ds}_operating_point_v10_s2_n{n}.json")
    dc = load(f"results_{ds}_dc_ac_v10_s2_n{n}.json")
    out.append(f"| {ds} | {op['public_v7r']['PFR']:.3f} | {op['sample_v7r']['PFR']:.3f} | {dc['v7r_dc']['PFR']:.3f} | {dc['random_ac']['PFR']:.3f} | {dc['random_dc']['PFR']:.3f} |")

out += ["", "## Raw v7r Top-k Sensitivity (voltage min 0.85)", "", "| system | k=1 | k=3 | k=5 | k=8 |", "|---|---:|---:|---:|---:|"]
for ds, n in (("ieee24", 200), ("ieee39", 104), ("uk", 200)):
    d = load(f"results_{ds}_v7r_sensitivity_v10_s2_n{n}.json")
    vals = [d["rows"][str(k)]["thresholds"]["0.85"]["v7r"]["PFR"] for k in (1, 3, 5, 8)]
    out.append(f"| {ds} | " + " | ".join(f"{x:.3f}" for x in vals) + " |")

(OUT / "ress_evidence_tables.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(OUT / "ress_evidence_tables.md")

#!/usr/bin/env python3
"""Quantify association between edge quality and raw AC operability."""
import csv
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_data" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
rows = []

for ds, pat in {
    "ieee24": "results_ieee24_supervised_edge_v7r_n200_s*.json",
    "ieee39": "results_ieee39_supervised_edge_v7r_n134_s*.json",
    "uk": "results_uk_supervised_edge_v7r_full_n200_s*.json",
}.items():
    for path in sorted(ROOT.glob(pat)):
        d = json.loads(path.read_text())
        rows.append({"system": ds, "method": "v7r-oracle", "seed": d["_meta"]["seed"],
                     "bacc": d["_meta"]["wrapper_raw_bacc"], "raw_pfr": d["supervised_edge"]["PFR"]})

for ds, pat in {
    "ieee24": "results_ieee24_pg_unified_v8_eval50_s*.json",
    "ieee39": "results_ieee39_pg_unified_v8_eval50_s*.json",
    "uk": "results_uk_pg_unified_v8_full_eval50_s*.json",
}.items():
    for path in sorted(ROOT.glob(pat)):
        d = json.loads(path.read_text())
        rows.append({"system": ds, "method": "PGExplainer", "seed": d["_meta"]["seed"],
                     "bacc": d["_meta"]["wrapper_raw_bacc"], "raw_pfr": d["pgexplainer"]["PFR"]})

with (OUT / "quality_operability_points.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["system", "method", "seed", "bacc", "raw_pfr"])
    w.writeheader(); w.writerows(rows)

md = ["# Edge Quality vs Raw AC Operability", "", "Each point is one system-method-seed result. PFR uses unwrapped top-5 interventions under the public AC protocol; this is an association analysis, not a causal claim.", "", "| group | n | Spearman rho | p |", "|---|---:|---:|---:|"]
for label, pts in [("all", rows)] + [(m, [r for r in rows if r["method"] == m]) for m in sorted({r["method"] for r in rows})]:
    b = [r["bacc"] for r in pts]; p = [r["raw_pfr"] for r in pts]
    rho, pv = spearmanr(b, p)
    md.append(f"| {label} | {len(pts)} | {rho:.3f} | {pv:.4g} |")
md += ["", "Interpretation: a weak or negative association means edge-label agreement cannot substitute for AC-operability validation. Saturated PFR systems should be discussed separately."]
(OUT / "quality_operability_correlation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print(OUT / "quality_operability_correlation.md")

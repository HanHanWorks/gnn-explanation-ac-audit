#!/usr/bin/env python3
"""Aggregate raw AC failure taxonomy without mixing wrapper and raw protocols."""
import glob
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_data" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
configs = {
    "v7r oracle raw": {
        "ieee24": ("results_ieee24_supervised_edge_v7r_n200_s*.json", "supervised_edge"),
        "ieee39": ("results_ieee39_supervised_edge_v7r_n134_s*.json", "supervised_edge"),
        "uk": ("results_uk_supervised_edge_v7r_full_n200_s*.json", "supervised_edge"),
    },
    "GNNExplainer raw": {
        "ieee24": ("results_ieee24_gin_unified_v8_eval50_s*.json", "gnn_explainer"),
        "ieee39": ("results_ieee39_gin_unified_v8_eval50_s*.json", "gnn_explainer"),
        "uk": ("results_uk_gin_unified_v8_full_eval50_s*.json", "gnn_explainer"),
    },
    "PGExplainer raw": {
        "ieee24": ("results_ieee24_pg_unified_v8_eval50_s*.json", "pgexplainer"),
        "ieee39": ("results_ieee39_pg_unified_v8_eval50_s*.json", "pgexplainer"),
        "uk": ("results_uk_pg_unified_v8_full_eval50_s*.json", "pgexplainer"),
    },
    "Grad-CAM raw": {
        "ieee24": ("results_ieee24_gradcam_v8_s*_n200.json", "gradcam"),
        "ieee39": ("results_ieee39_gradcam_v8_s*_n104.json", "gradcam"),
        "uk": ("results_uk_gradcam_v8_s*_n200.json", "gradcam"),
    },
}
lines = ["# Raw AC Failure Taxonomy", "", "Counts aggregate only like-for-like raw top-5 public AC results. Wrapper results are intentionally excluded.", "", "| method | system | n | feasible | island | non-converged | voltage | overload |", "|---|---|---:|---:|---:|---:|---:|---:|"]
for method, systems in configs.items():
    for system, (pattern, key) in systems.items():
        total, c = 0, Counter()
        for p in sorted(ROOT.glob(pattern)):
            d = json.loads(p.read_text())
            item = d[key]
            total += item["n"]
            c.update(item.get("types", {}))
        if total:
            lines.append(f"| {method} | {system} | {total} | {c['feasible']} | {c['island']} | {c['non_converged']} | {c['voltage']} | {c['overload']} |")
(OUT / "raw_ac_failure_taxonomy.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT / "raw_ac_failure_taxonomy.md")

#!/usr/bin/env python3
"""Index reproducible Phase 1 JSON results for the RESS paper package."""
import glob
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_data"
patterns = [
    "results_*_unified_v8_*.json", "results_*_pg_unified_v8_*.json",
    "results_*_supervised_edge_v7r_*.json", "results_*_gsat_v8_*.json",
    "results_*_supervised_edge_v7r_eval50_*.json",
    "results_*_supervised_v9_contingency_*.json",
    "results_*_operating_point_v10_*.json", "results_*_v7r_sensitivity_v10_*.json",
    "results_*_dc_ac_v10_*.json",
    "results_*_gradcam_v8_*.json", "results_*_subgraphx_v8_*.json",
    "results_*_physical_*.json",
]
rows = []
seen = set()
for pattern in patterns:
    for raw_path in sorted(ROOT.glob(pattern)):
        if raw_path in seen:
            continue
        seen.add(raw_path)
        try:
            data = json.loads(raw_path.read_text(encoding="utf-8"))
            meta = data.get("_meta", {})
            rows.append({
                "source": str(raw_path.relative_to(ROOT)),
                "system": meta.get("system"),
                "method": meta.get("method", meta.get("explainer")),
                "protocol": meta.get("protocol"),
                "seed": meta.get("seed"),
                "n": meta.get("n"),
                "eval_start": meta.get("eval_start"),
                "ranking_uses_truth": meta.get("ranking_uses_truth"),
                "keys": sorted(k for k in data if k != "_meta"),
            })
        except Exception as exc:
            rows.append({"source": str(raw_path.relative_to(ROOT)), "error": repr(exc)})

# Phase 0 scenario-aware backbone results are outside the Phase 1 directory,
# but are part of the RESS evidence package and must remain traceable.
for raw_path in sorted((ROOT.parent / "phase0" / "ckpt").glob("*/gin_3l_32h_contingency_s*_result.json")):
    try:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        split = data.get("split", {})
        rows.append({
            "source": "../" + str(raw_path.relative_to(ROOT.parent)),
            "system": data.get("dataset"),
            "method": "GIN_backbone",
            "protocol": "phase0-scenario-aware-contingency",
            "seed": data.get("seed"),
            "n": split.get("n", [None, None, None])[-1],
            "eval_start": None,
            "ranking_uses_truth": False,
            "keys": sorted(data.keys()),
        })
    except Exception as exc:
        rows.append({"source": "../" + str(raw_path.relative_to(ROOT.parent)), "error": repr(exc)})

PAPER.mkdir(exist_ok=True)
(PAPER / "manifest.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
with (PAPER / "manifest.csv").open("w", encoding="utf-8") as f:
    f.write("source,system,method,protocol,seed,n,eval_start,ranking_uses_truth\n")
    for r in rows:
        vals = [r.get(k, "") for k in ("source", "system", "method", "protocol", "seed", "n", "eval_start", "ranking_uses_truth")]
        f.write(",".join('"' + str(v).replace('"', '""') + '"' for v in vals) + "\n")
print(f"indexed={len(rows)} manifest={PAPER / 'manifest.json'}")

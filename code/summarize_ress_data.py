#!/usr/bin/env python3
"""Create a compact RESS-facing snapshot from the current JSON artifacts."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper_data" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))

def ms(xs):
    a = np.asarray(xs, dtype=float)
    return f"{a.mean():.3f} +/- {a.std(ddof=1):.3f}"

rows = []
for ds, n in (("ieee24", 200), ("ieee39", 104), ("uk", 200)):
    gsat = ROOT / f"results_{ds}_gsat_v8_s2_n{n}.json"
    phys = ROOT / f"results_{ds}_physical_v9_lowstress_s2_n{n}.json"
    if gsat.exists():
        d = json.loads(gsat.read_text())
        rows.append((ds, "GSAT s2", d["gsat"]["PFR"], d["random"]["PFR"], "n=1 seed"))
    subgraphx = ROOT / f"results_{ds}_subgraphx_v8_s2_n{n}.json"
    subgraphx_pilot = ROOT / f"results_{ds}_subgraphx_v8_pilot50_s2.json"
    sx = subgraphx if subgraphx.exists() else subgraphx_pilot
    if sx.exists():
        d = json.loads(sx.read_text())
        rows.append((ds, "SubgraphX s2", d["subgraphx"]["PFR"], d["random"]["PFR"], f"n={d['_meta']['n']}"))
    if phys.exists():
        d = json.loads(phys.read_text())
        rows.append((ds, "physical low-stress s2", d["physical_wrapped"]["PFR"], d["random_wrapped"]["PFR"], "n=1 seed; unsupervised"))

for ds, n in (("ieee24", 200), ("ieee39", 104), ("uk", 200)):
    paths = [ROOT / f"results_{ds}_gradcam_v8_s{s}_n{n}.json" for s in range(2, 7)]
    if all(p.exists() for p in paths):
        vals = [json.loads(p.read_text())["gradcam"]["PFR"] for p in paths]
        rnd = [json.loads(p.read_text())["random"]["PFR"] for p in paths]
        rows.append((ds, "Grad-CAM s2..6", ms(vals), ms(rnd), "five seeds"))

for ds in ("ieee24", "ieee39", "uk"):
    paths = sorted(ROOT.glob(f"results_{ds}_supervised_v9_contingency_s*.json"))
    if len(paths) == 5:
        vals, rnd = [], []
        for p in paths:
            d = json.loads(p.read_text())
            vals.append(d["_meta"]["wrapper_bacc"])
            rnd.append(d["_meta"]["random_wrapper_bacc"])
        rows.append((ds, "v7r contingency s2..6", ms(vals), ms(rnd), "five seeds; disjoint scenarios"))

for ds, n in (("ieee24", 200), ("ieee39", 104), ("uk", 200)):
    op = ROOT / f"results_{ds}_operating_point_v10_s2_n{n}.json"
    dc = ROOT / f"results_{ds}_dc_ac_v10_s2_n{n}.json"
    if op.exists():
        d = json.loads(op.read_text())
        rows.append((ds, "v7r sample-conditioned", d["sample_v7r"]["PFR"], d["sample_random"]["PFR"], "seed 2; reconstructed injections"))
    if dc.exists():
        d = json.loads(dc.read_text())
        rows.append((ds, "v7r DC", d["v7r_dc"]["PFR"], d["random_dc"]["PFR"], "seed 2; DC connectivity/flow"))

out = ["# RESS 当前数据快照", "", "该表由 `summarize_ress_data.py` 生成；pilot 和单 seed 结果不进入正式均值。", "", "| system | method | PFR | random PFR | evidence |", "|---|---|---:|---:|---|"]
for row in rows:
    out.append("| " + " | ".join(str(x) for x in row) + " |")
out += ["", "当前门：`NOT_SOTA_PENDING_FINAL_AUDIT`。低应力物理排序跨系统方向不一致，v7r 仍为监督 oracle；需要 scenario-aware 外部测试和完整 POE 设计后才能决定方法 SOTA。"]
(OUT / "ress_current_data_snapshot.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print(OUT / "ress_current_data_snapshot.md")

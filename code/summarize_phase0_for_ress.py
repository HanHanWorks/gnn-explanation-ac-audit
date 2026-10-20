#!/usr/bin/env python3
"""Summarize the existing Phase 0 random-split predictive backbones for RESS."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
CKPT = ROOT.parent / "phase0" / "ckpt"
OUT = ROOT / "paper_data" / "tables"
OUT.mkdir(parents=True, exist_ok=True)

def ms(xs):
    x = np.asarray(xs, dtype=float)
    return f"{x.mean():.3f} +/- {x.std(ddof=1):.3f}"

lines = ["# Phase 0 Predictive Backbones", "", "All values use the existing random graph split. They establish the predictive models being explained, not scenario-aware generalization.", "", "| system | model | test BACC | test F1 | test accuracy | n seeds |", "|---|---|---:|---:|---:|---:|"]
for ds in ("ieee24", "ieee39", "uk"):
    for model, suffix in (("GIN", "gin_3l_32h_s*_scores.json"), ("Transformer", "transformer_3l_32h_s*_scores.json")):
        paths = sorted((CKPT / ds).glob(suffix))
        if model == "GIN":
            paths = [p for p in paths if "_full_b512_" not in p.name]
        if len(paths) != 5:
            continue
        vals = [json.loads(p.read_text()) for p in paths]
        lines.append(f"| {ds} | {model} | {ms([v['test_balanced_acc'] for v in vals])} | {ms([v['test_f1_score'] for v in vals])} | {ms([v['test_acc'] for v in vals])} | 5 |")
    if ds == "uk":
        paths = sorted((CKPT / ds).glob("gin_3l_32h_s*_full_b512_scores.json"))
        if len(paths) == 5:
            vals = [json.loads(p.read_text()) for p in paths]
            lines.append(f"| {ds} | GIN full-batch (separate protocol) | {ms([v['test_balanced_acc'] for v in vals])} | {ms([v['test_f1_score'] for v in vals])} | {ms([v['test_acc'] for v in vals])} | 5 |")
(OUT / "phase0_predictive_backbones.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(OUT / "phase0_predictive_backbones.md")

#!/usr/bin/env python3
"""Build the RESS table for scenario-aware Phase 0 backbone sensitivity."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "phase0" / "ckpt"
OUT = ROOT / "phase1" / "paper_data" / "tables" / "phase0_scenario_aware_backbones.md"


def main():
    rows = []
    for system in ("ieee24", "ieee39", "uk"):
        for path in sorted((CKPT / system).glob("gin_3l_32h_contingency_s*_result.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            split = data["split"]
            rows.append((system, data, split))
    lines = [
        "# Scenario-Aware Phase 0 GIN Sensitivity",
        "",
        "Initial-contingency signatures are disjoint across train/validation/test. "
        "These are five-seed sensitivity results, not part of the random-split predictive main table.",
        "",
        "| system | seed | batch | test BACC | test F1 | train / val / test n | train / val / test positive n | train / val / test scenarios | train seconds |",
        "|---|---:|---:|---:|---:|---|---|---|---:|",
    ]
    for system, data, split in rows:
        lines.append(
            f"| {system} | {data['seed']} | {data['batch']} | {data['test_bacc']:.3f} | {data['test_f1']:.3f} | "
            f"{' / '.join(map(str, split['n']))} | {' / '.join(map(str, split['positive_n']))} | "
            f"{' / '.join(map(str, split['scenario_n']))} | {data['train_seconds']:.1f} |"
        )
    lines += ["", "## Five-seed summary", "", "| system | seeds | BACC mean +/- SD | 95% CI (normal approximation) | F1 mean +/- SD | test scenario range |", "|---|---|---:|---:|---:|---:|"]
    for system in ("ieee24", "ieee39", "uk"):
        subset = [data for name, data, _ in rows if name == system]
        vals = [float(x["test_bacc"]) for x in subset]
        f1s = [float(x["test_f1"]) for x in subset]
        mean = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        half = 1.96 * sd / math.sqrt(len(vals))
        fmean = sum(f1s) / len(f1s)
        fsd = math.sqrt(sum((x - fmean) ** 2 for x in f1s) / (len(f1s) - 1)) if len(f1s) > 1 else 0.0
        scenarios = [x["split"]["scenario_n"][2] for x in subset]
        lines.append(f"| {system} | {','.join(str(x['seed']) for x in subset)} | {mean:.3f} +/- {sd:.3f} | [{mean-half:.3f}, {mean+half:.3f}] | {fmean:.3f} +/- {fsd:.3f} | {min(scenarios)}--{max(scenarios)} |")
    lines += [
        "",
        "Interpretation boundary: IEEE39 has high variance despite five seeds, so high random-split accuracy cannot be treated as scenario-generalization evidence. "
        "Results are descriptive and are not a claim of cross-system superiority. Batch size is 64 for IEEE24/IEEE39 and 512 for UK due to the existing UK resource protocol.",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run a fixed-budget end-to-end runtime benchmark for the RESS supplement.

Each method receives the same IEEE24 seed-2, eval_start=50, n=20, top-5
workload. Wall time includes data loading, method setup, explanation, and AC
audit. Training-based setup is recorded separately in the method note.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "paper_data"
LOGDIR = PAPER / "raw_results" / "runtime"
PY = sys.executable

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--eval_start", type=int, default=50)
    args = ap.parse_args()
    LOGDIR.mkdir(parents=True, exist_ok=True)
    common = ["--dataset", "ieee24", "--num", str(args.num), "--eval_start", str(args.eval_start), "--seed", str(args.seed)]
    ckpt = str(ROOT.parent / "phase0" / "ckpt" / "ieee24" / "gin_3l_32h_s2_best.pth")
    tasks = [
        ("v7r", ["run_supervised_edge_audit.py", *common, "--train_num", "20", "--output", str(ROOT / "results_ieee24_runtime_v7r_s2_n20.json")], "LR fit + AC wrapper"),
        ("gnnexplainer", ["run_audit.py", *common, "--model", "gin", "--checkpoint", ckpt, "--output", str(ROOT / "results_ieee24_runtime_gnn_s2_n20.json")], "200 optimization epochs per graph + AC audit"),
        ("pgexplainer", ["run_pgexplainer_audit.py", *common, "--model", "gin", "--checkpoint", ckpt, "--train_num", "20", "--epochs", "20", "--physical_wrapper", "--wrapper_strategy", "local_swap", "--output", str(ROOT / "results_ieee24_runtime_pg_s2_n20.json")], "includes 20-epoch PGExplainer training + AC wrapper"),
        ("gradcam", ["run_gradcam_audit.py", *common, "--model", "gin", "--checkpoint", ckpt, "--output", str(ROOT / "results_ieee24_runtime_gradcam_s2_n20.json")], "gradient attribution + AC audit"),
        ("gsat", ["run_gsat_audit.py", *common, "--model", "gin", "--checkpoint", ckpt, "--train_size", "500", "--output", str(ROOT / "results_ieee24_runtime_gsat_s2_n20.json")], "uses cached GSAT checkpoint + AC audit"),
        ("subgraphx", ["run_subgraphx_audit.py", *common, "--model", "gin", "--checkpoint", ckpt, "--rollout", "5", "--sample_num", "20", "--min_atoms", "4", "--output", str(ROOT / "results_ieee24_runtime_subgraphx_s2_n20.json")], "MCTS/Shapley budget rollout=5,sample=20 + AC audit"),
    ]
    rows = []
    for name, argv, note in tasks:
        log = LOGDIR / f"{name}_ieee24_s{args.seed}_n{args.num}.log"
        t0 = time.perf_counter()
        with log.open("w", encoding="utf-8") as f:
            proc = subprocess.run([PY, str(ROOT / argv[0]), *argv[1:]], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - t0
        rows.append({"method": name, "system": "ieee24", "seed": args.seed, "n": args.num,
                     "wall_seconds": elapsed, "per_graph_seconds": elapsed / args.num,
                     "returncode": proc.returncode, "note": note, "log": str(log.relative_to(ROOT))})
        print(f"{name}: rc={proc.returncode} wall={elapsed:.2f}s")
        if proc.returncode != 0:
            break
    output = PAPER / "tables" / "runtime_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# Fixed-Budget Runtime Benchmark", "", "IEEE24, seed=2, eval_start=50, n=20, top-5; wall time includes data loading, explanation and AC audit.", "", "| method | wall seconds | seconds/graph | return code | note |", "|---|---:|---:|---:|---|"]
    for r in rows:
        md.append(f"| {r['method']} | {r['wall_seconds']:.2f} | {r['per_graph_seconds']:.3f} | {r['returncode']} | {r['note']} |")
    (PAPER / "tables" / "runtime_benchmark.md").write_text("\n".join(md) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()

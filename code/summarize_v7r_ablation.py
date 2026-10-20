#!/usr/bin/env python3
"""Summarize fixed-evaluation train-size and shuffled-label controls."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CASES = {
    "ieee24": {
        "train5": "results_ieee24_supervised_edge_v7r_eval50_train5_s4.json",
        "train20": "results_ieee24_supervised_edge_v7r_eval50_train20_s4.json",
        "train50": "results_ieee24_supervised_edge_v7r_eval50_train50_s4.json",
        "shuffle20": "results_ieee24_supervised_edge_v7r_eval50_shuffle_s4.json",
    },
    "ieee39": {
        "train5": "results_ieee39_supervised_edge_v7r_eval50_train5_s4.json",
        "train50": "results_ieee39_supervised_edge_v7r_eval50_train50_s4.json",
        "shuffle20": "results_ieee39_supervised_edge_v7r_eval50_shuffle_s4.json",
    },
    "uk_full": {
        "train5": "results_uk_supervised_edge_v7r_eval50_train5_s4.json",
        "train50": "results_uk_supervised_edge_v7r_eval50_train50_s4.json",
        "shuffle20": "results_uk_supervised_edge_v7r_eval50_shuffle_s4.json",
    },
}


def main():
    out = {"protocol": "v7r_fixed_eval_start50_ablation", "groups": {}}
    for system, cases in CASES.items():
        out["groups"][system] = {}
        for label, filename in cases.items():
            with open(ROOT / filename, encoding="utf-8") as f:
                d = json.load(f)
            m = d["_meta"]
            out["groups"][system][label] = {
                "train_num": m["train_num"], "n": m["n"],
                "shuffle": m["shuffle_train_labels"],
                "raw_bacc": m["wrapper_raw_bacc"],
                "wrapped_bacc": m["wrapper_bacc"],
                "random_bacc": m["random_wrapper_bacc"],
                "wrapped_pfr": d["supervised_edge_wrapped"]["PFR"],
                "random_pfr": d["random_wrapped"]["PFR"],
            }
    prefix = ROOT / "summary_v7r_ablation_eval50"
    with open(f"{prefix}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(f"{prefix}.md", "w", encoding="utf-8") as f:
        f.write("# v7r 固定评估集消融\n\n")
        f.write("seed=4，eval_start=50；除 train_num/label shuffle 外，协议固定。\n\n")
        f.write("| system | case | train n | raw BACC | wrapped BACC | random BACC | wrapped PFR | random PFR |\n|---|---|---:|---:|---:|---:|---:|---:|\n")
        for system, cases in out["groups"].items():
            for label, d in cases.items():
                f.write(f"| {system} | {label} | {d['train_num']} | {d['raw_bacc']:.4f} | {d['wrapped_bacc']:.4f} | {d['random_bacc']:.4f} | {d['wrapped_pfr']:.3f} | {d['random_pfr']:.3f} |\n")
        f.write("\n消融结论：train=5 在 IEEE24 接近随机，但 IEEE39/UK 仍有质量信号；train=50 在三系统均增强。shuffle-label 通常显著降低质量，UK shuffle 仍高于随机，需多 seed 负对照后再定论。\n")
    print(f"JSON: {prefix}.json")
    print(f"MD: {prefix}.md")


if __name__ == "__main__":
    main()

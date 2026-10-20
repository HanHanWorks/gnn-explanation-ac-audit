#!/usr/bin/env python3
"""Export the CSV files used to draw Figures 2-5 and Table 5."""
import csv, glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(DATA, "figure_csv")
os.makedirs(OUT, exist_ok=True)
SYS = ["ieee24", "ieee39", "uk"]
SYS_LABEL = {"ieee24": "IEEE 24", "ieee39": "IEEE 39", "uk": "UK"}


def load(p):
    with open(p) as f:
        return json.load(f)


def archived(method, system):
    base = os.path.join(DATA, "results_json")
    pats = {
        "GNNExplainer": {
            "ieee24": "results_ieee24_gin_unified_v8_eval50_s{}.json",
            "ieee39": "results_ieee39_gin_unified_v8_eval50_s{}.json",
            "uk": "results_uk_gin_unified_v8_full_eval50_s{}.json"},
        "PGExplainer": {
            "ieee24": "results_ieee24_pg_unified_v8_eval50_s{}.json",
            "ieee39": "results_ieee39_pg_unified_v8_eval50_s{}.json",
            "uk": "results_uk_pg_unified_v8_full_eval50_s{}.json"},
        "Grad-CAM": {
            "ieee24": "results_ieee24_gradcam_v8_s{}_n200.json",
            "ieee39": "results_ieee39_gradcam_v8_s{}_n104.json",
            "uk": "results_uk_gradcam_v8_s{}_n200.json"},
        "GSAT": {
            "ieee24": "results_ieee24_gsat_v8_s{}_n200.json",
            "ieee39": "results_ieee39_gsat_v8_s{}_n104.json",
            "uk": "results_uk_gsat_v8_s2_n200.json"},
        "SubgraphX": {
            "ieee24": "results_ieee24_subgraphx_v8_s2_n200.json",
            "ieee39": "results_ieee39_subgraphx_v8_s2_n104.json",
            "uk": "results_uk_subgraphx_v8_pilot50_s2.json"},
    }
    keys = {"GNNExplainer": "gnn_explainer", "PGExplainer": "pgexplainer",
            "Grad-CAM": "gradcam", "GSAT": "gsat", "SubgraphX": "subgraphx"}
    seeds = [2, 3, 4, 5, 6]
    if method == "GSAT":
        seeds = [2, 3, 4] if system != "uk" else [2]
    if method == "SubgraphX":
        seeds = [2]
    rows = []
    for s in seeds:
        p = os.path.join(base, pats[method][system].format(s))
        if os.path.exists(p):
            d = load(p)
            v = d[keys[method]]
            rows.append((s, v["n"], v["types"], v["PFR"]))
    return rows


# Figure 2 / Table 5
with open(os.path.join(OUT, "figure_2_outcomes.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "method", "seeds", "n", "feasible", "island",
                "non_converged", "voltage", "pfr"])
    for system in SYS:
        q = load(os.path.join(DATA, "results_revision", "results_quality", f"{system}_quality.json"))
        for method, key in [("SEP", "SEP"), ("|Q|", "absQ"), ("Truth", "truth"),
                            ("Random", "random")]:
            n = sum(v["methods"][key]["pfr"]["n"] for v in q["seeds"].values())
            c = {}
            for v in q["seeds"].values():
                for k, x in v["methods"][key]["pfr"]["types"].items():
                    c[k] = c.get(k, 0) + x
            w.writerow([SYS_LABEL[system], method, len(q["seeds"]), n,
                        c.get("feasible", 0), c.get("island", 0),
                        c.get("non_converged", 0), c.get("voltage", 0),
                        c.get("feasible", 0) / n])
        for method in ["GNNExplainer", "PGExplainer", "Grad-CAM", "GSAT", "SubgraphX"]:
            rows = archived(method, system)
            if not rows:
                continue
            n = sum(r[1] for r in rows)
            c = {}
            for _, _, t, _ in rows:
                for k, x in t.items():
                    c[k] = c.get(k, 0) + x
            w.writerow([SYS_LABEL[system], method, len(rows), n,
                        c.get("feasible", 0), c.get("island", 0),
                        c.get("non_converged", 0), c.get("voltage", 0),
                        c.get("feasible", 0) / n])

# Figure 3
with open(os.path.join(OUT, "figure_3_quality.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "ranking", "seed", "bacc", "pfr"])
    for system in SYS:
        q = load(os.path.join(DATA, "results_revision", "results_quality", f"{system}_quality.json"))
        for method, key in [("SEP", "SEP"), ("|Q|", "absQ"), ("|P|+|Q|", "absPQ"),
                            ("Random", "random"), ("Truth", "truth")]:
            for seed, v in q["seeds"].items():
                w.writerow([SYS_LABEL[system], method, seed,
                            v["methods"][key]["bacc"], v["methods"][key]["pfr"]["PFR"]])
        for fp in sorted(glob.glob(os.path.join(DATA, "results_revision", "results_pg_quality",
                                                f"{system}_pg_s*.json"))):
            d = load(fp)
            w.writerow([SYS_LABEL[system], "PGExplainer", d["_meta"]["seed"],
                        d["edge_label_agreement"]["pooled_bacc"], d["pgexplainer"]["PFR"]])

with open(os.path.join(OUT, "figure_3_null.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "seed", "shuffled_model_index", "bacc"])
    for system in SYS:
        q = load(os.path.join(DATA, "results_revision", "results_quality", f"{system}_quality.json"))
        for seed, v in q["seeds"].items():
            for k, b in enumerate(v["shuffled_label_bacc"]["values"]):
                w.writerow([SYS_LABEL[system], seed, k, b])

# Figure 4
transfer = {
    "ieee24": ([0.934, 0.928, 0.730, 0.859, 0.729], 0.970, [9, 6, 6, 6, 8]),
    "ieee39": ([0.540, 0.900, 0.598, 0.691, 0.927], 0.926, [9, 6, 9, 9, 9]),
    "uk": ([0.999, 0.966, 0.989, 0.999, 0.996], 0.971, [27, 15, 13, 27, 27]),
}
with open(os.path.join(OUT, "figure_4_transfer.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["system", "seed", "random_split_bacc", "contingency_bacc", "test_scenarios"])
    for system, (vals, rnd, scen) in transfer.items():
        for s, v, sc in zip([2, 3, 4, 5, 6], vals, scen):
            w.writerow([SYS_LABEL[system], s, rnd, v, sc])

# Figure 5
with open(os.path.join(OUT, "figure_5_protocol.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["panel", "system", "item", "value"])
    sep_k = {"ieee24": [0, 0, 0, 0], "ieee39": [1.0, 0.1923, 0.0096, 0.0],
             "uk": [1.0, 1.0, 0.700, 0.055]}
    rand_k = {"ieee24": [0.975, 0.805, 0.520, 0.170],
              "ieee39": [0.769, 0.279, 0.087, 0.0],
              "uk": [1.0, 1.0, 1.0, 1.0]}
    for system in SYS:
        for k, v in zip([1, 3, 5, 8], sep_k[system]):
            w.writerow(["a", SYS_LABEL[system], f"SEP_k{k}", v])
        for k, v in zip([1, 3, 5, 8], rand_k[system]):
            w.writerow(["a", SYS_LABEL[system], f"random_k{k}", v])
    for system in SYS:
        op = load(os.path.join(DATA, "results_revision", "results_op", f"{system}_op_raw.json"))
        for key in ["public_raw", "sample_raw", "sample_base"]:
            w.writerow(["c", SYS_LABEL[system], key, op[key]["PFR"]])

print("wrote", OUT)

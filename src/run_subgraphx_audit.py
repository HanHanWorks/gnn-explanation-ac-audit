#!/usr/bin/env python3
"""SubgraphX MCTS edge-mask baseline under the unified_protocol physical audit protocol."""
import argparse, json, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mat73
import numpy as np
import torch
from gnn.model import get_gnnNets
from explainer.subgraphx import SubgraphX
from audit_core import build_net, audit_baseline, pfr_summary
from run_audit import load_sample

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--model", default="gin")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--eval_start", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--rollout", type=int, default=5)
    ap.add_argument("--sample_num", type=int, default=20)
    ap.add_argument("--min_atoms", type=int, default=4)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    raw = os.path.join(DATA_ROOT, args.dataset, "raw")
    blist = np.asarray(mat73.loadmat(os.path.join(raw, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(raw, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(raw, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(raw, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(raw, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)
    model = get_gnnNets(3, 2, {"model_name": args.model, "num_layers": 3,
        "hidden_dim": 32, "dropout": 0.1, "readout": "max", "edge_dim": 4},
        "binary_classification")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["net"])
    model.eval()
    subgraphx = SubgraphX(model, 2, torch.device("cpu"), num_hops=3,
        rollout=args.rollout, min_atoms=args.min_atoms, expand_atoms=8,
        high2low=True, sample_num=args.sample_num, reward_method="mc_shapley",
        subgraph_building_method="zero_filling", vis=False)

    pos_idx = [i for i in range(len(of_bi))
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    results = {"base": [], "subgraphx": [], "random": [], "truth": []}
    elapsed = 0.0
    sample_idx = pos_idx[args.eval_start:args.eval_start + args.num]
    if len(sample_idx) < args.num:
        raise ValueError(f"not enough evaluation samples: requested {args.num}, got {len(sample_idx)}")
    for n, i in enumerate(sample_idx):
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        results["base"].append(audit_baseline(net, []))
        t0 = time.time()
        edge_mask = np.asarray(subgraphx.explain(data.x, data.edge_index, data.edge_attr,
            max_nodes=args.topk, label=data.y))
        elapsed += time.time() - t0
        half = len(keep_rows)
        pair = np.maximum(edge_mask[:half], edge_mask[half:2 * half]) if len(edge_mask) >= 2 * half else edge_mask[:half]
        top = np.argsort(pair)[::-1][:min(args.topk, half)]
        results["subgraphx"].append(audit_baseline(net, [mapping[keep_rows[j]] for j in top]))
        rng = random.Random(100000 * args.seed + n)
        rnd = rng.sample(range(len(mapping)), min(args.topk, len(mapping)))
        results["random"].append(audit_baseline(net, [mapping[j] for j in rnd]))
        truth_rows = [int(j) - 1 for j in np.asarray(exp[i][0], dtype=int).ravel()]
        results["truth"].append(audit_baseline(net, [mapping[j] for j in truth_rows]))
        print(f"  {n + 1}/{min(args.num, len(pos_idx))} completed")

    summary = {name: pfr_summary(values) for name, values in results.items()}
    summary["_meta"] = {"system": args.dataset, "model": args.model,
        "method": "subgraphx", "protocol": "unified-eval-start", "n": len(results["subgraphx"]),
        "topk": args.topk, "seed": args.seed, "rollout": args.rollout,
        "sample_num": args.sample_num, "min_atoms": args.min_atoms,
        "eval_start": args.eval_start,
        "runtime_seconds": elapsed}
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    for name in ("subgraphx", "random", "truth"):
        print(f"{name}: PFR={summary[name].get('PFR', 0):.3f}")
 print(f" : {args.output}")

if __name__ == "__main__":
    main()

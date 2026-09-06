#!/usr/bin/env python3
"""Gradient x Input edge baseline under the unified_protocol physical audit protocol."""
import argparse, json, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mat73
import numpy as np
import torch
from gnn.model import get_gnnNets
from audit_core import build_net, audit_baseline, pfr_summary
from run_audit import load_sample

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--model", default="gin")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=4)
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

    pos_idx = [i for i in range(len(of_bi))
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    sample_idx = pos_idx[:args.num]
    results = {"base": [], "gradient_x_input": [], "random": [], "truth": []}
    elapsed = 0.0
    for n, i in enumerate(sample_idx):
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        results["base"].append(audit_baseline(net, []))

        edge_attr = data.edge_attr.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        t0 = time.time()
        score = model(x=data.x, edge_index=data.edge_index, edge_attr=edge_attr)[0, 1]
        score.backward()
        elapsed += time.time() - t0
        edge_score = (edge_attr.grad * edge_attr).abs().sum(dim=1).detach().numpy()
        half = len(keep_rows)
        pair = np.maximum(edge_score[:half], edge_score[half:2 * half])
        top = np.argsort(pair)[::-1][:min(args.topk, half)]
        results["gradient_x_input"].append(
            audit_baseline(net, [mapping[keep_rows[j]] for j in top]))

        rnd = random.sample(range(len(mapping)), min(args.topk, len(mapping)))
        results["random"].append(audit_baseline(net, [mapping[j] for j in rnd]))
        truth_rows = [int(j) - 1 for j in np.asarray(exp[i][0], dtype=int).ravel()]
        results["truth"].append(audit_baseline(net, [mapping[j] for j in truth_rows]))
        if (n + 1) % 10 == 0:
 print(f" {n + 1}/{len(sample_idx)} ")

    summary = {name: pfr_summary(values) for name, values in results.items()}
    summary["_meta"] = {"system": args.dataset, "model": args.model,
        "method": "gradient_x_input", "protocol": "unified_protocol",
        "n": len(sample_idx), "topk": args.topk, "seed": args.seed,
        "runtime_seconds": elapsed}
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    for name in ("gradient_x_input", "random", "truth"):
        print(f"{name}: PFR={summary[name].get('PFR', 0):.3f}")
 print(f" : {args.output}")

if __name__ == "__main__":
    main()

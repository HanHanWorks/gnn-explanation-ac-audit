#!/usr/bin/env python3
"""Layer Grad-CAM edge baseline under the unified_protocol physical audit protocol."""
import argparse, json, os, random, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mat73
import numpy as np
import torch
from gnn.model import get_gnnNets, GCNConv, GATConv, GINEConv, TransformerConv
from explainer.gradcam import GraphLayerGradCam
from audit_core import build_net, audit_baseline, pfr_summary
from run_audit import load_sample

DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")

def forward_graph(x, model, edge_index, edge_attr):
    return model(x, edge_index, edge_attr)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--model", default="gin")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--eval_start", type=int, default=0)
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
    layers = [layer for layer in model.modules()
              if isinstance(layer, (GCNConv, GATConv, GINEConv, TransformerConv))]
    pos_idx = [i for i in range(len(of_bi))
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    results = {"base": [], "gradcam": [], "random": [], "truth": []}
    elapsed = 0.0
    sample_idx = pos_idx[args.eval_start:args.eval_start + args.num]
    if len(sample_idx) < args.num:
        raise ValueError(f"not enough evaluation samples: requested {args.num}, got {len(sample_idx)}")
    for n, i in enumerate(sample_idx):
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        results["base"].append(audit_baseline(net, []))
        t0 = time.time()
        node_scores = []
        for layer in layers:
            attr = GraphLayerGradCam(forward_graph, layer).attribute(
                data.x.clone().requires_grad_(True), target=1,
                additional_forward_args=(model, data.edge_index, data.edge_attr),
                relu_attributions=True)
            node_scores.append(attr.squeeze().detach().cpu().numpy())
        elapsed += time.time() - t0
        node_score = np.asarray(node_scores).mean(axis=0)
        edge_score = node_score[data.edge_index[0].cpu().numpy()] + node_score[data.edge_index[1].cpu().numpy()]
        half = len(keep_rows)
        pair = np.maximum(edge_score[:half], edge_score[half:2 * half]) if len(edge_score) >= 2 * half else edge_score[:half]
        top = np.argsort(pair)[::-1][:min(args.topk, half)]
        results["gradcam"].append(audit_baseline(net, [mapping[keep_rows[j]] for j in top]))
        rng = random.Random(100000 * args.seed + n)
        rnd = rng.sample(range(len(mapping)), min(args.topk, len(mapping)))
        results["random"].append(audit_baseline(net, [mapping[j] for j in rnd]))
        truth_rows = [int(j) - 1 for j in np.asarray(exp[i][0], dtype=int).ravel()]
        results["truth"].append(audit_baseline(net, [mapping[j] for j in truth_rows]))
        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{min(args.num, len(pos_idx))} completed")

    summary = {name: pfr_summary(values) for name, values in results.items()}
    summary["_meta"] = {"system": args.dataset, "model": args.model,
        "method": "gradcam", "protocol": "unified-eval-start", "n": len(results["gradcam"]),
        "topk": args.topk, "seed": args.seed, "layers": len(layers),
        "eval_start": args.eval_start,
        "runtime_seconds": elapsed}
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    for name in ("gradcam", "random", "truth"):
        print(f"{name}: PFR={summary[name].get('PFR', 0):.3f}")
 print(f" : {args.output}")

if __name__ == "__main__":
    main()

"""Audit experiment for the IEEE 24-bus system: explanation --> disconnection --> AC feasibility.

Candidate sets: (A) GNNExplainer top-k edges, (B) random k edges (control),
(C) ground-truth cascade-path edges, (D) empty set (baseline).
All candidates are superimposed on the sample's initial disconnected edges,
keeping the operating point fixed.

Usage: python run_audit_ieee24.py [--num 50] [--topk 5] [--seed 2]
"""
import sys, os, json, time, argparse, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mat73
import torch
from dataset.powergrid import PowerGrid
from gnn.model import get_gnnNets
from audit_core import build_net, derive_injections, audit, pfr_summary

DATA_ROOT = os.path.join(ROOT, "datasets/PowerGraph/data_cascades")
RAW_DIR = os.path.join(DATA_ROOT, "ieee24", "raw")
CKPT = os.path.join(ROOT, "checkpoints", "ieee24", "transformer_3l_32h_best.pth")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_ieee24.json")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    t0 = time.time()
    dataset = PowerGrid(root=DATA_ROOT, name="ieee24", task="binary_classification")
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    blist = np.asarray(mat73.loadmat(os.path.join(RAW_DIR, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(RAW_DIR, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(RAW_DIR, "exp.mat"))["explainations"]
    net, mapping = build_net("ieee24", root=ROOT)
 print(f"[{time.time()-t0:.0f}s] + ")

    model = get_gnnNets(3, 2, {"model_name": "transformer", "num_layers": 3, "hidden_dim": 32,
                               "dropout": 0.1, "readout": "max", "edge_dim": 4}, "binary_classification")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["net"])
    model.eval()
    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )

    pos_idx = [i for i in range(len(dataset)) if dataset[i].y.item() == 1]
    random.shuffle(pos_idx)
    sample_idx = pos_idx[: args.num]
 print(f" {len(sample_idx)} ")

    def cont_rows(i):
        f = np.asarray(Ef[i][0], dtype=float).reshape(-1, 4)
        return [j for j in range(len(f)) if np.all(np.abs(f[j]) < 1e-12)]

    def exp_rows(i):
        e = exp[i][0]
        return [int(j) - 1 for j in np.asarray(e, dtype=int).ravel()] if e is not None else []

    results = {"base": [], "gnn_explainer": [], "random": [], "truth": []}
    t_explain = 0.0
    for n, i in enumerate(sample_idx):
        data = dataset[i]
        cont = cont_rows(i)
        cont_elems = [mapping[j] for j in cont]
        P, Q = derive_injections(np.asarray(Ef[i][0]).reshape(-1, 4), blist, nbus=24)

        results["base"].append(audit(net, (P, Q), [], cont_edges=cont_elems))

        t1 = time.time()
        expl = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr, target=int(data.y.item()))
        em = np.asarray(expl.edge_mask.detach().cpu()) if hasattr(expl.edge_mask, "detach") else np.asarray(expl.edge_mask)
        t_explain += time.time() - t1
        keep_rows = [j for j in range(len(blist)) if j not in set(cont)]
        E_keep = len(keep_rows)
        pair = np.maximum(em[:E_keep], em[E_keep:2 * E_keep]) if len(em) >= 2 * E_keep else em[:E_keep]
        topk = np.argsort(pair)[::-1][: min(args.topk, E_keep)]
        disc_gnn = [mapping[keep_rows[j]] for j in topk]
        results["gnn_explainer"].append(audit(net, (P, Q), disc_gnn, cont_edges=cont_elems))

        noncont = [j for j in range(len(blist)) if j not in set(cont)]
        rnd = random.sample(noncont, min(args.topk, len(noncont)))
        results["random"].append(audit(net, (P, Q), [mapping[j] for j in rnd], cont_edges=cont_elems))

        truth = [mapping[j] for j in exp_rows(i) if j not in set(cont)]
        results["truth"].append(audit(net, (P, Q), truth, cont_edges=cont_elems))
        if (n + 1) % 10 == 0:
 print(f" {n+1}/{len(sample_idx)} ")

    print("\n" + "=" * 66)
 print(f"PFR (top-{args.topk}, n={len(sample_idx)}, [0.9,1.1], ≤110%):")
    summary = {}
    for name in ["base", "gnn_explainer", "random", "truth"]:
        s = pfr_summary(results[name])
        summary[name] = s
        print(f"  {name:<14} PFR={s.get('PFR', 0):.3f}  {s.get('types', {})}")
 print(f" : {t_explain/60:.1f} min")
    print("=" * 66)
 print(" : ≈0.8/ ≈0.93 → GNNExplainer PFR , ;")
 print(" PFR → ' ' ")

    summary["_meta"] = {"system": "ieee24", "n": len(sample_idx), "topk": args.topk, "seed": args.seed,
                        "model": "transformer_3l_32h", "voltage": "[0.9,1.1]", "load_max": 110.0}
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=float)
 print(f" : {OUT}")

if __name__ == "__main__":
    main()

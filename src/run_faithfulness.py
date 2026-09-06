"""Simulator-flip model-centric faithfulness metrics (fid+/fid-).

  fid+ (fidelity+): fraction of samples whose prediction is unchanged when only
      the explained top-k edges are kept (necessity of explained edges).
  fid- (fidelity-): fraction whose prediction flips when the top-k edges are
      removed (sufficiency of explained edges).

Usage: python run_faithfulness.py --dataset ieee24 --num 50
"""
import sys, os, json, time, argparse, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mat73
import torch
import torch_sparse_compat
sys.modules["torch_sparse"] = torch_sparse_compat

from gnn.model import get_gnnNets
from run_audit import load_sample
from run_gsat_audit import gsat_explain_graph, SubsetDataset

DATA_ROOT = os.path.join(ROOT, "datasets/PowerGraph/data_cascades")
MODEL = "transformer"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--skip_gsat", action="store_true")
    args = ap.parse_args()

    RAW_DIR = os.path.join(DATA_ROOT, args.dataset, "raw")
    model_name = args.model
    CKPT = args.checkpoint or os.path.join(
        ROOT, "checkpoints", args.dataset, f"{model_name}_3l_32h_best.pth")
    OUT = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_{args.dataset}_{model_name}_faith_v3_s{args.seed}.json")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    blist = np.asarray(mat73.loadmat(os.path.join(RAW_DIR, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(RAW_DIR, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(RAW_DIR, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(RAW_DIR, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(RAW_DIR, "Bf.mat"))["B_f_tot"]

    model = get_gnnNets(3, 2, {"model_name": model_name, "num_layers": 3, "hidden_dim": 32,
                               "dropout": 0.1, "readout": "max", "edge_dim": 4}, "binary_classification")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["net"])
    model.eval()

    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    gnn_explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )

    pool = list(range(len(of_bi))); random.seed(args.seed); random.shuffle(pool)
    tds = [load_sample(i, blist, Ef, exp, of_bi, Bf)[0] for i in pool[:500]]
    for d in tds:
        d.edge_mask = d.edge_mask.view(-1)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
    gsat_kwargs = {"dataset": SubsetDataset(tds), "dataset_name": args.dataset, "seed": args.seed,
                   "num_classes": 2, "hidden_dim": 32, "batch_size": 8,
                   "model_save_dir": os.path.join(ROOT, "model_checkpoints"),
                   "model_name": model_name, "gsat_epochs": 50}

    pos_idx = [i for i in range(len(of_bi))
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    sample_idx = pos_idx[: args.num]
 print(f" : {len(sample_idx)}")

    def predict_subgraph(data, keep_idx):
 """ keep_idx ， """
        x, ei, ea = data.x, data.edge_index, data.edge_attr
        Eh = ei.shape[1] // 2
        keep = set(int(j) for j in keep_idx)
        keep_all = sorted(list(keep) + [Eh + j for j in keep])
        with torch.no_grad():
            logits = model(x=x, edge_index=ei[:, keep_all], edge_attr=ea[keep_all])
        return int(logits.argmax(dim=-1).item())

    res = {"gnn_explainer": {"fid_plus": [], "fid_minus": []}}
    if not args.skip_gsat:
        res["gsat"] = {"fid_plus": [], "fid_minus": []}
    for n, i in enumerate(sample_idx):
        # Match run_audit.py: explanation optimization is deterministic per instance.
        torch.manual_seed(1000 * args.seed + n)
        np.random.seed(1000 * args.seed + n)
        random.seed(1000 * args.seed + n)
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        data.edge_mask = data.edge_mask.view(-1)
        Eh = len(keep_rows)
        y = int(data.y.item())

        masks = {}
        expl = gnn_explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        em = np.asarray(expl.edge_mask.detach().cpu()) if hasattr(expl.edge_mask, "detach") else np.asarray(expl.edge_mask)
        masks["gnn_explainer"] = np.maximum(em[:Eh], em[Eh:2 * Eh]) if len(em) >= 2 * Eh else em[:Eh]
        if not args.skip_gsat:
            em_g, _ = gsat_explain_graph(model, data, y, device, **gsat_kwargs)
            emg = np.asarray(em_g.detach().cpu()) if hasattr(em_g, "detach") else np.asarray(em_g)
            masks["gsat"] = np.maximum(emg[:Eh], emg[Eh:2 * Eh]) if len(emg) >= 2 * Eh else emg[:Eh]

        for method, pair in masks.items():
            topk = np.argsort(pair)[::-1][: min(args.topk, Eh)]
            rest = [j for j in range(Eh) if j not in set(topk)]

            p_keep = predict_subgraph(data, topk)

            p_del = predict_subgraph(data, rest)
            res[method]["fid_plus"].append(int(p_keep == y))
            res[method]["fid_minus"].append(int(p_del != y))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(sample_idx)}")

    summary = {}
    for method in res:
        fp = np.mean(res[method]["fid_plus"]); fm = np.mean(res[method]["fid_minus"])
        summary[method] = {"fid_plus": float(fp), "fid_minus": float(fm),
                           "fid_plus_n": len(res[method]["fid_plus"])}
        print(f"  {method:<14} fid+={fp:.3f}  fid-={fm:.3f}")
    summary["_meta"] = {"system": args.dataset, "n": len(sample_idx), "topk": args.topk,
                         "seed": args.seed, "per_instance_seed": "1000*seed+n"}
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=float)
 print(f" : {OUT}")

if __name__ == "__main__":
    main()

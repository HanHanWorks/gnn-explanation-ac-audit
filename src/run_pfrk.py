"""PFR@k curve (k=1/3/5/8) plus official balanced acc and X-AUROC fidelity.

Explains once, audits at multiple thresholds. Methods: GNNExplainer (PyG
framework), GSAT (official implementation), random, ground truth.
Output: results_{sys}_pfrk.json.

Usage: python run_pfrk.py --dataset ieee24 --num 50 [--ks 1,3,5,8]
"""
import sys, os, json, time, argparse, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mat73
import torch
from torch_geometric.data import Data
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
import torch_sparse_compat
sys.modules["torch_sparse"] = torch_sparse_compat

from gnn.model import get_gnnNets
from audit_core import build_net, audit_baseline, pfr_summary
from run_audit import load_sample
from run_gsat_audit import gsat_explain_graph, SubsetDataset

DATA_ROOT = os.path.join(ROOT, "datasets/PowerGraph/data_cascades")
MODEL = "transformer"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--ks", default="1,3,5,8")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None)
 ap.add_argument("--skip_gsat", action="store_true", help=" GSAT checkpoint ")
    args = ap.parse_args()
    ks = [int(x) for x in args.ks.split(",")]

    RAW_DIR = os.path.join(DATA_ROOT, args.dataset, "raw")
    CKPT = args.checkpoint or os.path.join(ROOT, "checkpoints", args.dataset, f"{args.model}_3l_32h_best.pth")
    OUT = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{args.dataset}_{args.model}_pfrk_unified_protocol_s{args.seed}.json")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    t0 = time.time()
    blist = np.asarray(mat73.loadmat(os.path.join(RAW_DIR, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(RAW_DIR, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(RAW_DIR, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(RAW_DIR, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(RAW_DIR, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)
 print(f"[{time.time()-t0:.0f}s] + ({args.dataset}, {len(mapping)} )")

    model = get_gnnNets(3, 2, {"model_name": args.model, "num_layers": 3, "hidden_dim": 32,
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

    pool = list(range(len(of_bi))); random.shuffle(pool)
    tds = [load_sample(i, blist, Ef, exp, of_bi, Bf)[0] for i in pool[:500]]
    for d in tds:
        d.edge_mask = d.edge_mask.view(-1)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
    train_ds = SubsetDataset(tds)
    gsat_kwargs = {"dataset": train_ds, "dataset_name": args.dataset, "seed": args.seed,
                   "num_classes": 2, "hidden_dim": 32, "batch_size": 8,
                   "model_save_dir": os.path.join(ROOT, "model_checkpoints"),
                   "model_name": MODEL, "gsat_epochs": 50}

    pos_idx = [i for i in range(len(of_bi))
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    sample_idx = pos_idx[: args.num]
 print(f" {len(sample_idx)} ")

    def exp_rows(i):
        e = exp[i][0]
        return [int(j) - 1 for j in np.asarray(e, dtype=int).ravel()] if e is not None else []

    k_star = max(len(exp_rows(i)) for i in pos_idx)

    rows_meta, pair_gnn, pair_gsat = [], [], []
    truth_audits = []
    for n, i in enumerate(sample_idx):
        torch.manual_seed(1000 * args.seed + n)
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        data.edge_mask = data.edge_mask.view(-1)
        rows_meta.append((i, keep_rows))
        Eh = len(keep_rows)

        t1 = time.time()
        expl = gnn_explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        em = np.asarray(expl.edge_mask.detach().cpu()) if hasattr(expl.edge_mask, "detach") else np.asarray(expl.edge_mask)
        pair_gnn.append(np.maximum(em[:Eh], em[Eh:2 * Eh]) if len(em) >= 2 * Eh else em[:Eh])

        if not args.skip_gsat:
            em_g, _ = gsat_explain_graph(model, data, int(data.y.item()), device, **gsat_kwargs)
            emg = np.asarray(em_g.detach().cpu()) if hasattr(em_g, "detach") else np.asarray(em_g)
            pair_gsat.append(np.maximum(emg[:Eh], emg[Eh:2 * Eh]) if len(emg) >= 2 * Eh else emg[:Eh])

        truth = [mapping[j] for j in exp_rows(i) if j in set(keep_rows)]
        truth_audits.append(audit_baseline(net, truth))
        if (n + 1) % 10 == 0:
 print(f" {n+1}/{len(sample_idx)}")

    # ---- PFR@k ----
    def pfr_for(pairs, k, mode="topk"):
        res = []
        for (i, keep_rows), pair in zip(rows_meta, pairs):
            idx = np.argsort(pair)[::-1][: min(k, len(pair))]
            if mode == "topk":
                disc = [mapping[keep_rows[j]] for j in idx]
            res.append(audit_baseline(net, disc))
        return pfr_summary(res)["PFR"]

    results = {"k": {}, "truth": pfr_summary(truth_audits)["PFR"]}
    for k in ks:
        row = {"gnn_explainer": pfr_for(pair_gnn, k)}
        if not args.skip_gsat:
            row["gsat"] = pfr_for(pair_gsat, k)
        row["random"] = pfr_for([np.random.rand(len(p)) for p in pair_gnn], k)
        results["k"][str(k)] = row
        print(f"  PFR@{k}: {row}")

    def fidelity(pairs):
        baccs, aurocs = [], []
        for (i, keep_rows), pair in zip(rows_meta, pairs):
            e = exp[i][0]
            gt = np.zeros(len(keep_rows))
            exp_set = set(int(x) for x in np.asarray(e, dtype=int).ravel()) if e is not None else set()
            for k_, r in enumerate(keep_rows):
                if (r + 1) in exp_set:
                    gt[k_] = 1
            pred = np.zeros(len(keep_rows))
            pred[np.argsort(pair)[::-1][: min(k_star, len(pair))]] = 1
            baccs.append(balanced_accuracy_score(gt, pred))
            if gt.sum() > 0 and gt.sum() < len(gt):
                try:
                    aurocs.append(roc_auc_score(gt, pair))
                except Exception:
                    pass
        return {"balanced_acc_topkstar": float(np.mean(baccs)), "n": len(baccs),
                "x_auroc": float(np.mean(aurocs)) if aurocs else None}

    results["fidelity"] = {"gnn_explainer": fidelity(pair_gnn)}
    if not args.skip_gsat:
        results["fidelity"]["gsat"] = fidelity(pair_gsat)
    results["_meta"] = {"system": args.dataset, "n": len(sample_idx), "ks": ks, "seed": args.seed,
                   "k_star": k_star, "model": args.model, "protocol": "unified_protocol"}
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=float)
 print(f" : {OUT} | k*={k_star}")

if __name__ == "__main__":
    main()

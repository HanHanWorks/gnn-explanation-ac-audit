#!/usr/bin/env python3
"""Held-out-backbone audit on the exact GIN random-test split.

The random split used by ``gendata.get_dataloader`` is
``torch.randperm(N, generator=manual_seed(seed))`` with 85/5/10 partitions.
We reconstruct that split, keep only test graphs that are positive and
explainable, and evaluate explanations with the checkpoint trained on the
remaining graphs.  SEP and PGExplainer are trained only on non-test graphs.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gnn.model import get_gnnNets, GCNConv, GATConv, GINEConv, TransformerConv  # noqa: E402
from explainer.gradcam import GraphLayerGradCam  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

from audit_core import build_net, audit_baseline, pfr_summary  # noqa: E402
from run_audit import load_sample  # noqa: E402
from run_supervised_edge_audit import edge_features, truth_rows  # noqa: E402


def split_test_indices(n, seed):
    nt = int(0.85 * n)
    nv = int(0.05 * n)
    idx = torch.randperm(n, generator=torch.default_generator.manual_seed(seed)).tolist()
    return set(idx[nt + nv:])


def pooled_bacc(pred, gt):
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")
    tnr = tn / (tn + fp) if (tn + fp) else float("nan")
    return 0.5 * (tpr + tnr), tp, fp, tn, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--train_num", type=int, default=20)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    raw = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades", args.dataset, "raw")
    import mat73
    blist = np.asarray(mat73.loadmat(os.path.join(raw, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(raw, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(raw, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(raw, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(raw, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)

    n_all = len(of_bi)
    test = split_test_indices(n_all, args.seed)
    pos = [i for i in range(n_all)
           if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    eval_pool = [i for i in pos if i in test]
    random.Random(1000 + args.seed).shuffle(eval_pool)
    eval_idx = eval_pool[: min(args.num, len(eval_pool))]
    train_pool = [i for i in pos if i not in test]
    random.Random(2000 + args.seed).shuffle(train_pool)
    train_idx = train_pool[: min(args.train_num, len(train_pool))]

    model = get_gnnNets(3, 2, {"model_name": "gin", "num_layers": 3, "hidden_dim": 32,
                               "dropout": 0.1, "readout": "max", "edge_dim": 4},
                        "binary_classification")
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=False)["net"])
    model.eval()

    from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig
    from torch_geometric.explain.algorithm import PGExplainer
    gnn_explainer = Explainer(
        model=model, algorithm=GNNExplainer(epochs=200, lr=0.01),
        explanation_type="model", node_mask_type="object", edge_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification", task_level="graph",
                                 return_type="raw"),
    )
    pg_explainer = Explainer(
        model=model, algorithm=PGExplainer(epochs=20, lr=0.003),
        explanation_type="phenomenon", node_mask_type=None, edge_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification", task_level="graph",
                                 return_type="raw"),
    )
    train_data = [load_sample(i, blist, Ef, exp, of_bi, Bf)[0] for i in train_idx]
    for epoch in range(20):
        for data in train_data:
            pg_explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                         target=data.y, edge_attr=data.edge_attr)

    # SEP trained only on non-test graphs
    X, y = [], []
    for i in train_idx:
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        X.append(edge_features(data, keep))
        truth = truth_rows(exp, i)
        y.append(np.asarray([int(r in truth) for r in keep]))
    clf = LogisticRegression(class_weight="balanced", max_iter=1000,
                             random_state=args.seed, solver="liblinear")
    clf.fit(np.concatenate(X), np.concatenate(y))

    layers = [layer for layer in model.modules()
              if isinstance(layer, (GCNConv, GATConv, GINEConv, TransformerConv))]

    results = {k: [] for k in ["gnn_explainer", "pgexplainer", "gradcam", "sep", "random", "truth"]}
    counts = {k: [0, 0, 0, 0] for k in ["gnn_explainer", "pgexplainer", "gradcam", "sep"]}

    def audit_set(selected, keep):
        return audit_baseline(net, [mapping[keep[j]] for j in selected])

    def pred_bacc(selected, keep, i):
        truth = truth_rows(exp, i)
        gt = np.asarray([1 if int(r) in truth else 0 for r in keep], dtype=int)
        pred = np.zeros(len(keep), dtype=int)
        pred[selected] = 1
        return pooled_bacc(pred, gt)

    for n, i in enumerate(eval_idx):
        data, keep = load_sample(i, blist, Ef, exp, of_bi, Bf)
        half = len(keep)
        results["random"].append(
            audit_set(random.Random(100000 * args.seed + n).sample(range(half), min(args.topk, half)), keep))
        results["truth"].append(
            audit_set([j for j in range(half) if int(keep[j]) in truth_rows(exp, i)], keep))

        # GNNExplainer
        torch.manual_seed(1000 * args.seed + n)
        np.random.seed(1000 * args.seed + n)
        random.seed(1000 * args.seed + n)
        expl = gnn_explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        em = np.asarray(expl.edge_mask.detach().cpu())
        pair = np.maximum(em[:half], em[half:2 * half]) if len(em) >= 2 * half else em[:half]
        top = list(np.argsort(pair)[::-1][: min(args.topk, half)])
        results["gnn_explainer"].append(audit_set(top, keep))
        b, tp, fp, tn, fn = pred_bacc(top, keep, i)
        counts["gnn_explainer"][0] += tp; counts["gnn_explainer"][1] += fp
        counts["gnn_explainer"][2] += tn; counts["gnn_explainer"][3] += fn

        # PGExplainer
        expl_pg = pg_explainer(x=data.x, edge_index=data.edge_index, target=data.y,
                               edge_attr=data.edge_attr)
        emp = np.asarray(expl_pg.edge_mask.detach().cpu())
        pairp = np.maximum(emp[:half], emp[half:2 * half]) if len(emp) >= 2 * half else emp[:half]
        topp = list(np.argsort(pairp)[::-1][: min(args.topk, half)])
        results["pgexplainer"].append(audit_set(topp, keep))
        b, tp, fp, tn, fn = pred_bacc(topp, keep, i)
        counts["pgexplainer"][0] += tp; counts["pgexplainer"][1] += fp
        counts["pgexplainer"][2] += tn; counts["pgexplainer"][3] += fn

        # Grad-CAM
        node_scores = []
        for layer in layers:
            attr = GraphLayerGradCam(lambda x, m, ei, ea: m(x, ei, ea), layer).attribute(
                data.x.clone().requires_grad_(True), target=1,
                additional_forward_args=(model, data.edge_index, data.edge_attr),
                relu_attributions=True)
            node_scores.append(attr.squeeze().detach().cpu().numpy())
        node_score = np.asarray(node_scores).mean(axis=0)
        edge_score = node_score[data.edge_index[0].cpu().numpy()] + node_score[data.edge_index[1].cpu().numpy()]
        pairc = np.maximum(edge_score[:half], edge_score[half:2 * half]) if len(edge_score) >= 2 * half else edge_score[:half]
        topc = list(np.argsort(pairc)[::-1][: min(args.topk, half)])
        results["gradcam"].append(audit_set(topc, keep))
        b, tp, fp, tn, fn = pred_bacc(topc, keep, i)
        counts["gradcam"][0] += tp; counts["gradcam"][1] += fp
        counts["gradcam"][2] += tn; counts["gradcam"][3] += fn

        # SEP
        scores = clf.predict_proba(edge_features(data, keep))[:, 1]
        tops = list(np.argsort(scores)[::-1][: min(args.topk, half)])
        results["sep"].append(audit_set(tops, keep))
        b, tp, fp, tn, fn = pred_bacc(tops, keep, i)
        counts["sep"][0] += tp; counts["sep"][1] += fp
        counts["sep"][2] += tn; counts["sep"][3] += fn

        if (n + 1) % 10 == 0:
            print(f"  {n + 1}/{len(eval_idx)}")

    summary = {k: pfr_summary(v) for k, v in results.items()}
    for k, (tp, fp, tn, fn) in counts.items():
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        tnr = tn / (tn + fp) if (tn + fp) else float("nan")
        summary[k]["edge_label_agreement"] = {
            "pooled_bacc": 0.5 * (tpr + tnr), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "tpr": tpr, "tnr": tnr,
        }
    summary["_meta"] = {
        "system": args.dataset, "seed": args.seed, "n": len(eval_idx),
        "n_test_explainable_positive": len(eval_pool), "train_num": len(train_idx),
        "topk": args.topk, "split": "random 85/5/10 test partition",
        "checkpoint": os.path.basename(args.checkpoint),
    }
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print({k: summary[k].get("PFR") for k in results})
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

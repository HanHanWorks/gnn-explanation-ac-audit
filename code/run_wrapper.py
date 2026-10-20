#!/usr/bin/env python3
"""Phase 2 原型：PFR-constrained explanation 包装器（可行性约束的解释修复）。

机制（explainer-agnostic 包装器）：解释器给出 top-k 候选边 → 按分数贪心加入断边集，
每步用 AC 潮流审计组合可行性；不可行的候选边被跳过并尝试次优边 → 输出"可行化"的 k 条边。

评价：
  - PFR_wrapper：可行化后断边集的物理可行率（对比原始 PFR）
  - 替换率 = 1 − |可行集 ∩ 原始 top-k| / k（保真度损失上界）
  - 官方口径保真度（balanced acc @ top-k）在包装前后对比（目标 ≤5% 损失）

用法: python run_wrapper.py --dataset uk --num 50 [--topk 5] [--seed 2]
"""
import sys, os, json, time, argparse, random
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import mat73
import torch
from sklearn.metrics import balanced_accuracy_score
import torch_sparse_compat
sys.modules["torch_sparse"] = torch_sparse_compat

from gnn.model import get_gnnNets
from audit_core import build_net, audit_baseline
from run_audit import load_sample
from run_gsat_audit import gsat_explain_graph, SubsetDataset

DATA_ROOT = os.path.join(ROOT, "datasets/PowerGraph/data_cascades")
MODEL = "transformer"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="uk")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    RAW_DIR = os.path.join(DATA_ROOT, args.dataset, "raw")
    CKPT = os.path.join(ROOT, "experiments/phase0/ckpt", args.dataset, f"{MODEL}_3l_32h_best.pth")
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{args.dataset}_wrapper.json")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    blist = np.asarray(mat73.loadmat(os.path.join(RAW_DIR, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(RAW_DIR, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(RAW_DIR, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(RAW_DIR, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(RAW_DIR, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)

    model = get_gnnNets(3, 2, {"model_name": MODEL, "num_layers": 3, "hidden_dim": 32,
                               "dropout": 0.1, "readout": "max", "edge_dim": 4}, "binary_classification")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["net"])
    model.eval()

    pool = list(range(len(of_bi))); random.seed(args.seed); random.shuffle(pool)
    tds = [load_sample(i, blist, Ef, exp, of_bi, Bf)[0] for i in pool[:500]]
    for d in tds:
        d.edge_mask = d.edge_mask.view(-1)
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
    gsat_kwargs = {"dataset": SubsetDataset(tds), "dataset_name": args.dataset, "seed": args.seed,
                   "num_classes": 2, "hidden_dim": 32, "batch_size": 8,
                   "model_save_dir": os.path.join(ROOT, "experiments/phase1/gsat_ckpt"),
                   "model_name": MODEL, "gsat_epochs": 50}

    pos_idx = [i for i in range(len(of_bi))
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    sample_idx = pos_idx[: args.num]
    print(f"样本: {len(sample_idx)}")

    k = args.topk
    stats = {"pfr_orig": [], "pfr_wrapped": [], "replace_rate": [], "n_feasible": [],
             "bacc_orig": [], "bacc_wrapped": []}

    for n, i in enumerate(sample_idx):
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        data.edge_mask = data.edge_mask.view(-1)
        Eh = len(keep_rows)
        y = int(data.y.item())

        em_g, _ = gsat_explain_graph(model, data, y, device, **gsat_kwargs)
        emg = np.asarray(em_g.detach().cpu()) if hasattr(em_g, "detach") else np.asarray(em_g)
        pair = np.maximum(emg[:Eh], emg[Eh:2 * Eh]) if len(emg) >= 2 * Eh else emg[:Eh]
        order = np.argsort(pair)[::-1]                     # 分数降序候选
        topk = order[:k]
        orig_disc = [mapping[keep_rows[j]] for j in topk]

        # 原始 PFR（直接断 top-k）
        ok_orig, st_orig, _ = audit_baseline(net, orig_disc)
        stats["pfr_orig"].append(int(ok_orig))

        # 贪心可行化：按分数顺序逐个尝试，组合可行才保留
        chosen = []                                        # 选中的 keep_rows 索引
        for j in order:
            if len(chosen) >= k:
                break
            cand = chosen + [int(j)]
            disc = [mapping[keep_rows[c]] for c in cand]
            ok, _, _ = audit_baseline(net, disc)
            if ok:
                chosen = cand
        wrapped_disc = [mapping[keep_rows[c]] for c in chosen]
        ok_w, st_w, _ = audit_baseline(net, wrapped_disc)
        stats["pfr_wrapped"].append(int(ok_w))
        stats["n_feasible"].append(len(chosen))
        stats["replace_rate"].append(1 - len(set(chosen) & set(topk)) / k)

        # 官方口径保真度（balanced acc @ top-k）前后对比
        e = exp[i][0]
        exp_set = set(int(x) for x in np.asarray(e, dtype=int).ravel()) if e is not None else set()
        gt = np.array([1 if (r + 1) in exp_set else 0 for r in keep_rows])
        def bacc_of(sel):
            pred = np.zeros(Eh); pred[sel] = 1
            return balanced_accuracy_score(gt, pred)
        stats["bacc_orig"].append(bacc_of(topk))
        stats["bacc_wrapped"].append(bacc_of(chosen) if chosen else 0.5)
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(sample_idx)}")

    r = {key: float(np.mean(v)) for key, v in stats.items()}
    print("\n" + "=" * 60)
    print(f"修复包装器 [{args.dataset}] (top-{k}, n={len(sample_idx)}):")
    print(f"  原始 PFR (GSAT top-k):        {r['pfr_orig']:.3f}")
    print(f"  包装后 PFR (可行化 top-k):    {r['pfr_wrapped']:.3f}")
    print(f"  平均保留可行边数:             {r['n_feasible']:.2f}/{k}")
    print(f"  平均替换率 (保真度损失上界):  {r['replace_rate']*100:.1f}%")
    print(f"  官方保真度 balanced acc: 原 {r['bacc_orig']:.3f} → 包装后 {r['bacc_wrapped']:.3f}"
          f" (Δ={r['bacc_wrapped']-r['bacc_orig']:+.3f})")
    print("=" * 60)
    r["_meta"] = {"system": args.dataset, "n": len(sample_idx), "topk": k, "seed": args.seed}
    with open(OUT, "w") as f:
        json.dump(r, f, indent=2, default=float)
    print(f"结果已保存: {OUT}")


if __name__ == "__main__":
    main()

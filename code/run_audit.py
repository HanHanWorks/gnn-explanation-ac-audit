#!/usr/bin/env python3
"""Phase 1 主实验（v2 公共工况协议）：解释 → 候选断开 → AC 潮流审计 → PFR 指标族。

协议（评审报告 2.3 + 3.5）：
  - 审计工况 = System.m 原始运行点（pandapower 精确复现，杜绝重构误差）
  - 候选修改 = 解释器 top-k 边在原始拓扑上断开（预防性切除语义）
  - 对照：base(空集, sanity) / random(k 条) / truth(真值级联路径 exp 边)
  - 判据：AC-NR 收敛 + 电压[0.85,1.1] + 负载≤110%，类型学: feasible/non_converged/island/voltage/overload

用法: python run_audit.py --dataset ieee24 --num 50 --topk 5 [--model transformer] [--seed 2]
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
from gnn.model import get_gnnNets
from audit_core import build_net, audit_baseline, pfr_summary

DATA_ROOT = os.path.join(ROOT, "datasets/PowerGraph/data_cascades")


def physical_risk_summary(values):
    """Continuous-ish physical risk: voltage violation magnitude, hard failure=1."""
    scores = []
    voltage_excess = []
    for ok, status, detail in values:
        if ok or status == "feasible":
            scores.append(0.0)
        elif status == "voltage" and detail:
            excess = max(0.0, 0.85 - float(detail.get("vmin", 0.85)),
                         float(detail.get("vmax", 1.10)) - 1.10)
            scores.append(float(min(1.0, excess / 0.25)))
            voltage_excess.append(excess)
        else:
            scores.append(1.0)
    return {"mean_risk": float(np.mean(scores)) if scores else None,
            "risk_n": len(scores),
            "mean_voltage_excess": float(np.mean(voltage_excess)) if voltage_excess else 0.0}


def load_sample(i, blist, Ef, exp, of_bi, Bf):
    """按需构建单个样本的 PyG Data（与官方 powergrid.process 单样本逻辑一致，不加载全量 data.pt）。"""
    f = np.asarray(Ef[i][0], dtype=float).reshape(-1, 4)
    cont = set(j for j in range(len(f)) if np.all(np.abs(f[j]) < 1e-12))
    keep_rows = [j for j in range(len(blist)) if j not in cont]
    x = torch.tensor(np.asarray(Bf[i][0], dtype=float).reshape(-1, 3), dtype=torch.float32)
    ei = torch.tensor(blist[keep_rows] - 1, dtype=torch.long).t().contiguous()
    ei = torch.cat([ei, ei.flip(0)], dim=1)                      # 双向边
    ea = torch.tensor(f[keep_rows], dtype=torch.float32)
    ea = torch.cat([ea, ea], dim=0)
    y = torch.tensor(int(np.asarray(of_bi[i][0]).ravel()[0]), dtype=torch.long).view(1)
    e = exp[i][0]
    exp_set = set(int(j) - 1 for j in np.asarray(e, dtype=int).ravel()) if e is not None else set()
    mask = torch.zeros(len(keep_rows), 1)
    for k, r in enumerate(keep_rows):
        if r in exp_set:
            mask[k] = 1.0
    edge_mask = torch.cat([mask, mask], dim=0)
    return Data(x=x, edge_index=ei, edge_attr=ea, y=y, edge_mask=edge_mask), keep_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--eval_start", type=int, default=0,
                    help="Offset into the deterministic positive-sample list")
    ap.add_argument("--max_graphs", type=int, default=0,
                    help="Restrict source graph indices; 0 means all graphs")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    RAW_DIR = os.path.join(DATA_ROOT, args.dataset, "raw")
    CKPT = args.checkpoint or os.path.join(
        ROOT, "experiments/phase0/ckpt", args.dataset, f"{args.model}_3l_32h_best.pth")
    OUT = args.output or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"results_{args.dataset}_{args.model}_v3_s{args.seed}.json")

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    t0 = time.time()
    blist = np.asarray(mat73.loadmat(os.path.join(RAW_DIR, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(RAW_DIR, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(RAW_DIR, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(RAW_DIR, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(RAW_DIR, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)
    print(f"[{time.time()-t0:.0f}s] 数据(.mat 按需)+网络就绪 ({args.dataset}, {len(mapping)} 支路)")

    model = get_gnnNets(3, 2, {"model_name": args.model, "num_layers": 3, "hidden_dim": 32,
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

    # 只用"有解释标签"的正类样本（truth 对照需要 ground-truth 路径；论文解释评估同此口径）
    upper = min(len(of_bi), args.max_graphs) if args.max_graphs > 0 else len(of_bi)
    pos_idx = [i for i in range(upper)
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    sample_idx = pos_idx[args.eval_start:args.eval_start + args.num]
    if len(sample_idx) < args.num:
        raise ValueError(f"not enough evaluation samples: requested {args.num}, got {len(sample_idx)}")
    print(f"采样 {len(sample_idx)} 个有解释标签的正类样本 (可解释正类总数={len(pos_idx)})")

    def exp_rows(i):
        e = exp[i][0]
        return [int(j) - 1 for j in np.asarray(e, dtype=int).ravel()] if e is not None else []

    results = {"base": [], "gnn_explainer": [], "random": [], "truth": []}
    t_explain = 0.0
    for n, i in enumerate(sample_idx):
        # Freeze per-instance explainer randomness so seed effects are reproducible.
        torch.manual_seed(1000 * args.seed + n)
        np.random.seed(1000 * args.seed + n)
        random.seed(1000 * args.seed + n)
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)

        results["base"].append(audit_baseline(net, []))   # sanity: 原始 case 必可行

        t1 = time.time()
        expl = explainer(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)
        em = np.asarray(expl.edge_mask.detach().cpu()) if hasattr(expl.edge_mask, "detach") else np.asarray(expl.edge_mask)
        t_explain += time.time() - t1
        # 双向边掩码取每对物理边 max，再取 top-k（解释对象 = 该样本保留边，对应 blist 非 cont 行）
        E_half = len(keep_rows)
        pair = np.maximum(em[:E_half], em[E_half:2 * E_half]) if len(em) >= 2 * E_half else em[:E_half]
        topk = np.argsort(pair)[::-1][: min(args.topk, E_half)]
        disc_gnn = [mapping[keep_rows[j]] for j in topk]
        results["gnn_explainer"].append(audit_baseline(net, disc_gnn))

        noncont_all = list(range(len(mapping)))
        raw_rng = random.Random(100000 * args.seed + n)
        rnd = raw_rng.sample(noncont_all, min(args.topk, len(noncont_all)))
        results["random"].append(audit_baseline(net, [mapping[j] for j in rnd]))

        truth = [mapping[j] for j in exp_rows(i)]
        results["truth"].append(audit_baseline(net, truth))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(sample_idx)} 完成")

    print("\n" + "=" * 66)
    print(f"PFR 指标族 [{args.dataset}] (top-{args.topk}, n={len(sample_idx)}, 公共工况, 电压[0.85,1.1], 负载≤110%):")
    summary = {}
    for name in ["base", "gnn_explainer", "random", "truth"]:
        s = pfr_summary(results[name])
        s.update(physical_risk_summary(results[name]))
        summary[name] = s
        print(f"  {name:<14} PFR={s.get('PFR', 0):.3f}  {s.get('types', {})}")
    print(f"  解释耗时: {t_explain/60:.1f} min")
    print("=" * 66)

    summary["_meta"] = {"system": args.dataset, "n": len(sample_idx), "eval_start": args.eval_start,
                        "max_graphs": args.max_graphs, "topk": args.topk, "seed": args.seed,
                        "model": args.model, "protocol": "baseline-common-operating-point",
                        "voltage": "[0.85,1.1]", "load_max": 110.0,
                        "per_instance_seed": "1000*seed+n"}
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"结果已保存: {OUT}")


if __name__ == "__main__":
    main()

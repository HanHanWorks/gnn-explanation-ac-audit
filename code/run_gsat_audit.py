#!/usr/bin/env python3
"""Phase 1 决胜实验：GSAT（官方 0.90 保真度）→ 候选断开 → AC 潮流审计 → PFR 对照。

协议与 run_audit.py 相同（公共工况审计），候选来自 GSAT 内在注意力掩码 top-k。
GSAT 训练使用官方实现（500 样本子集，联合优化，首次运行自动训练并存 checkpoint）。

用法: python run_gsat_audit.py --dataset ieee24 --num 50 --topk 5 [--seed 2]
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

# 注入 torch_sparse 兼容模块（macOS 无 wheel；GSAT 仅用 transpose）
import torch_sparse_compat
sys.modules["torch_sparse"] = torch_sparse_compat

from gnn.model import get_gnnNets
from audit_core import build_net, audit_baseline, pfr_summary
from run_audit import load_sample   # 复用按需加载

# GSAT 相关（只导入 gsat 模块链，避免 graph_explainer 全链的 torch_scatter/gym/wandb 依赖）
from explainer.gsat import GSAT, ExtractorMLP, gsat_get_config
from explainer.explainer_utils.gsat import init_metric_dict, save_checkpoint, load_checkpoint
from gendata import get_dataloader
from torch.optim.lr_scheduler import ReduceLROnPlateau


def gsat_explain_graph(model, data, target, device, **kwargs):
    """内联官方 explain_gsat_graph 核心（GSAT 训练/加载 + 单样本注意力掩码）。"""
    from copy import deepcopy
    dataset_name = kwargs["dataset_name"]; seed = kwargs["seed"]; num_class = kwargs["num_classes"]
    subdir = os.path.join(kwargs["model_save_dir"], "gsat")
    os.makedirs(subdir, exist_ok=True)
    gsat_saving_path = os.path.join(subdir, f"gsat_{dataset_name}_{str(device)}_{seed}.pt")

    shared_config, method_config = gsat_get_config()
    if kwargs.get("gsat_epochs") is not None:
        method_config["epochs"] = kwargs["gsat_epochs"]   # 缩短训练（控制后台任务时长，防环境回收）
    extractor = ExtractorMLP(kwargs["hidden_dim"], shared_config).to(device)
    lr = method_config["lr"]; wd = method_config.get("weight_decay", 0)
    optimizer = torch.optim.Adam(list(extractor.parameters()) + list(model.parameters()), lr=lr, weight_decay=wd)
    sched_cfg = method_config.get("scheduler", {})
    scheduler = None if sched_cfg == {} else ReduceLROnPlateau(optimizer, mode="max", **sched_cfg)

    train_size = min(len(kwargs["dataset"]), 500)
    explain_dataset_idx = random.sample(range(len(kwargs["dataset"])), k=train_size)
    explain_dataset = kwargs["dataset"][explain_dataset_idx]
    dataloader_params = {"batch_size": kwargs["batch_size"], "random_split_flag": "True",
                         "data_split_ratio": [0.85, 0.05, 0.1], "seed": seed}
    loader = None

    if os.path.isfile(gsat_saving_path):
        print("Load saved GSAT model...")
        load_checkpoint(extractor, subdir, model_name=f"gsat_{dataset_name}_{str(device)}_{seed}")
        extractor = extractor.to(device)
        gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name,
                    num_class, False, seed, method_config, shared_config)
    else:
        # 仅训练时才构建 loader（避免消耗 torch 随机流、污染其他方法的确定性）
        loader, _, _, test_dataset = get_dataloader(explain_dataset, **dataloader_params)
        print("[INFO] Training GSAT...")
        gsat = GSAT(model, extractor, optimizer, scheduler, device, subdir, dataset_name,
                    num_class, False, seed, method_config, shared_config)
        metric_dict = deepcopy(init_metric_dict)
        gsat.train(loader, test_dataset, metric_dict, use_edge_attr=True)
        save_checkpoint(gsat.extractor, subdir, model_name=f"gsat_{dataset_name}_{str(device)}_{seed}")

    data.batch = torch.zeros(data.num_nodes, dtype=torch.long)
    gsat.extractor = gsat.extractor.to(device)
    data = data.to(device)
    edge_att, _, _ = gsat.eval_one_batch(data, epoch=method_config["epochs"])
    return edge_att, None

DATA_ROOT = os.path.join(ROOT, "datasets/PowerGraph/data_cascades")


class SubsetDataset:
    """轻量 PyG 数据集包装：支持 len / int 索引 / list 索引（GSAT 官方代码用）。"""
    def __init__(self, datas):
        self.datas = datas
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, idx):
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            return SubsetDataset([self.datas[i] for i in idx])   # 返回新包装（可继续 fancy 索引）
        return self.datas[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--model", default="transformer")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--eval_start", type=int, default=0)
    ap.add_argument("--max_graphs", type=int, default=0)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--train_size", type=int, default=500)
    ap.add_argument("--gsat_epochs", type=int, default=None, help="GSAT 训练 epochs（默认用官方配置 100；缩短以控制时长）")
    ap.add_argument("--output", default=None, help="结果 JSON 输出路径；不指定时使用默认结果文件")
    args = ap.parse_args()

    RAW_DIR = os.path.join(DATA_ROOT, args.dataset, "raw")
    CKPT = args.checkpoint or os.path.join(ROOT, "experiments/phase0/ckpt", args.dataset,
                                           f"{args.model}_3l_32h_best.pth")
    OUT = args.output or os.path.join(os.path.dirname(os.path.abspath(__file__)), f"results_{args.dataset}_gsat.json")
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cpu")

    t0 = time.time()
    blist = np.asarray(mat73.loadmat(os.path.join(RAW_DIR, "blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(RAW_DIR, "Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(RAW_DIR, "exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(RAW_DIR, "of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(RAW_DIR, "Bf.mat"))["B_f_tot"]
    net, mapping = build_net(args.dataset, root=ROOT)
    print(f"[{time.time()-t0:.0f}s] 数据+网络就绪 ({args.dataset}, {len(mapping)} 支路)")

    model = get_gnnNets(3, 2, {"model_name": args.model, "num_layers": 3, "hidden_dim": 32,
                               "dropout": 0.1, "readout": "max", "edge_dim": 4}, "binary_classification")
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["net"])
    model.eval()

    # GSAT 训练子集（有解释标签正类 + 负类混合，官方 500 样本）
    upper = min(len(of_bi), args.max_graphs) if args.max_graphs > 0 else len(of_bi)
    pos_idx = [i for i in range(upper)
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    neg_idx = [i for i in range(len(of_bi)) if int(np.asarray(of_bi[i][0]).ravel()[0]) == 0]
    random.shuffle(pos_idx); random.shuffle(neg_idx)
    sample_idx = pos_idx[args.eval_start:args.eval_start + args.num]
    if len(sample_idx) < args.num:
        raise ValueError(f"not enough evaluation samples: requested {args.num}, got {len(sample_idx)}")
    eval_set = set(sample_idx)
    train_pool = [i for i in (pos_idx[:args.eval_start] + neg_idx) if i not in eval_set][:args.train_size]
    train_datas = [load_sample(i, blist, Ef, exp, of_bi, Bf)[0] for i in train_pool]
    for d in train_datas:
        d.edge_mask = d.edge_mask.view(-1)   # GSAT 期望一维边标签
        d.batch = torch.zeros(d.num_nodes, dtype=torch.long)
    train_ds = SubsetDataset(train_datas)
    print(f"GSAT 训练子集: {len(train_ds)} 样本")

    # 测试样本：使用首次 shuffle 后的固定序列，不再被训练集流程污染。
    print(f"审计样本: {len(sample_idx)} 个有解释标签正类")

    from copy import deepcopy
    kwargs = {
        "dataset": train_ds, "dataset_name": args.dataset, "seed": args.seed,
        "num_classes": 2, "hidden_dim": 32, "batch_size": 8,
        "model_save_dir": os.path.join(ROOT, "experiments/phase1/gsat_ckpt"),
        "model_name": args.model, "gsat_epochs": args.gsat_epochs,
    }

    def exp_rows(i):
        e = exp[i][0]
        return [int(j) - 1 for j in np.asarray(e, dtype=int).ravel()] if e is not None else []

    results = {"base": [], "gsat": [], "random": [], "truth": []}
    t_explain = 0.0
    for n, i in enumerate(sample_idx):
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        data.edge_mask = data.edge_mask.view(-1)   # GSAT 一维
        results["base"].append(audit_baseline(net, []))

        t1 = time.time()
        em_raw, _ = gsat_explain_graph(model, data, int(data.y.item()), device, **kwargs)
        em = np.asarray(em_raw.detach().cpu()) if hasattr(em_raw, "detach") else np.asarray(em_raw)
        t_explain += time.time() - t1
        E_half = len(keep_rows)
        pair = np.maximum(em[:E_half], em[E_half:2 * E_half]) if len(em) >= 2 * E_half else em[:E_half]
        topk = np.argsort(pair)[::-1][: min(args.topk, E_half)]
        results["gsat"].append(audit_baseline(net, [mapping[keep_rows[j]] for j in topk]))

        rnd_rng = random.Random(100000 * args.seed + n)
        rnd = rnd_rng.sample(list(range(len(mapping))), min(args.topk, len(mapping)))
        results["random"].append(audit_baseline(net, [mapping[j] for j in rnd]))
        truth = [mapping[j] for j in exp_rows(i)]
        results["truth"].append(audit_baseline(net, truth))
        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(sample_idx)} 完成")

    print("\n" + "=" * 66)
    print(f"PFR 指标族 [{args.dataset}][GSAT] (top-{args.topk}, n={len(sample_idx)}, 电压[0.85,1.1], 负载≤110%):")
    summary = {}
    for name in ["base", "gsat", "random", "truth"]:
        s = pfr_summary(results[name])
        summary[name] = s
        print(f"  {name:<14} PFR={s.get('PFR', 0):.3f}  {s.get('types', {})}")
    print(f"  解释耗时: {t_explain/60:.1f} min")
    print("=" * 66)

    summary["_meta"] = {"system": args.dataset, "n": len(sample_idx), "topk": args.topk, "seed": args.seed,
        "model": f"{args.model}_3l_32h", "explainer": "gsat", "protocol": "unified-eval-start",
        "eval_start": args.eval_start, "max_graphs": args.max_graphs,
                        "voltage": "[0.85,1.1]", "load_max": 110.0}
    with open(OUT, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"结果已保存: {OUT}")


if __name__ == "__main__":
    main()

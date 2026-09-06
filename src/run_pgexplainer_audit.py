#!/usr/bin/env python3
"""Train PyG PGExplainer and evaluate it with the unified_protocol physical audit."""
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

def select_feasible(order, scores, mapping, keep_rows, net, topk, strategy="greedy",
                    beam_width=3, candidate_pool=8):
    """Select a feasible top-k set while keeping the raw explainer ranking fixed.

    ``greedy`` is the original protocol. ``beam`` changes only the combinatorial
    search: it retains the highest cumulative explainer-score feasible prefixes.
    """
    if strategy in ("greedy", "local_swap"):
        chosen = []
        for j in order:
            if len(chosen) >= topk:
                break
            candidate = chosen + [int(j)]
            ok, _, _ = audit_baseline(net, [mapping[keep_rows[k]] for k in candidate])
            if ok:
                chosen = candidate
        if strategy == "greedy" or len(chosen) != topk:
            return chosen

        # Bounded one-edge local search: preserve exactly-k and only accept a
        # feasible replacement that improves the explainer-score sum.
        pool = [int(j) for j in order] if candidate_pool <= 0 else [
            int(j) for j in order[:min(candidate_pool, len(order))]
        ]
        current = list(chosen)
        current_score = float(sum(scores[j] for j in current))
        best = current
        best_score = current_score
        for old in list(current):
            for j in pool:
                if j in current:
                    continue
                candidate = [x for x in current if x != old] + [j]
                ok, _, _ = audit_baseline(
                    net, [mapping[keep_rows[k]] for k in candidate])
                score = float(sum(scores[k] for k in candidate))
                if ok and score > best_score + 1e-12:
                    best, best_score = candidate, score
        return best

    pool = [int(j) for j in order] if candidate_pool <= 0 else [
        int(j) for j in order[:min(candidate_pool, len(order))]
    ]
    # Each state is (selected indices, cumulative raw explainer score).
    states = [(tuple(), 0.0)]
    for _ in range(topk):
        expanded = []
        for selected, total_score in states:
            for j in pool:
                if j in selected:
                    continue
                candidate = list(selected) + [j]
                ok, _, _ = audit_baseline(net, [mapping[keep_rows[k]] for k in candidate])
                if ok:
                    expanded.append((tuple(candidate), total_score + float(scores[j])))
        if not expanded:
            break
        expanded.sort(key=lambda item: item[1], reverse=True)
        # Deduplicate by set while retaining the score-ranked ordering.
        seen = set()
        states = []
        for selected, score in expanded:
            key = frozenset(selected)
            if key in seen:
                continue
            seen.add(key)
            states.append((selected, score))
            if len(states) >= beam_width:
                break
    if not states:
        return []
    return list(max(states, key=lambda item: (len(item[0]), item[1]))[0])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ieee24")
    ap.add_argument("--model", default="gin")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num", type=int, default=50)
    ap.add_argument("--train_num", type=int, default=50)
    ap.add_argument("--eval_start", type=int, default=-1,
                    help="Fixed evaluation offset; -1 uses train_num")
    ap.add_argument("--max_graphs", type=int, default=0,
                    help="Restrict source graph indices; 0 means all graphs")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--physical_wrapper", action="store_true",
                    help="Greedily retain only AC-feasible additions from the PGExplainer ranking")
    ap.add_argument("--wrapper_strategy", choices=["greedy", "local_swap", "beam"], default="greedy",
                    help="Feasible wrapper search strategy; beam changes only combinatorial selection")
    ap.add_argument("--beam_width", type=int, default=3)
    ap.add_argument("--candidate_pool", type=int, default=8,
                    help="Beam candidate pool; 0 means all mapped edges")
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

    upper = min(len(of_bi), args.max_graphs) if args.max_graphs > 0 else len(of_bi)
    pos_idx = [i for i in range(upper)
               if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    random.shuffle(pos_idx)
    train_idx = pos_idx[:min(args.train_num, len(pos_idx))]
    eval_start = args.train_num if args.eval_start < 0 else args.eval_start
    if eval_start < len(train_idx):
        raise ValueError("eval_start must be >= train_num to prevent train/eval overlap")
    eval_idx = pos_idx[eval_start:min(eval_start + args.num, len(pos_idx))]
    if len(eval_idx) < args.num:
        raise ValueError(f"not enough disjoint eval samples: requested {args.num}, got {len(eval_idx)}")
    train_data = [load_sample(i, blist, Ef, exp, of_bi, Bf)[0] for i in train_idx]

    from torch_geometric.explain import Explainer, ModelConfig
    from torch_geometric.explain.algorithm import PGExplainer
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=args.epochs, lr=0.003),
        explanation_type="phenomenon",
        node_mask_type=None,
        edge_mask_type="object",
        model_config=ModelConfig(mode="multiclass_classification", task_level="graph", return_type="raw"),
    )
    losses = []
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for data in train_data:
            loss = explainer.algorithm.train(
                epoch, model, data.x, data.edge_index, target=data.y,
                edge_attr=data.edge_attr)
            epoch_loss += loss
        losses.append(epoch_loss / max(1, len(train_data)))
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"PGExplainer epoch {epoch + 1}/{args.epochs} loss={losses[-1]:.6f}")

    results = {"base": [], "pgexplainer": [], "random": [], "truth": []}
    wrapper_size, wrapper_replace, random_wrapper_size = [], [], []
    raw_bacc, wrapped_bacc, random_wrapped_bacc = [], [], []
    if args.physical_wrapper:
        results["pgexplainer_wrapped"] = []
        results["random_wrapped"] = []
    elapsed = 0.0
    for n, i in enumerate(eval_idx):
        data, keep_rows = load_sample(i, blist, Ef, exp, of_bi, Bf)
        results["base"].append(audit_baseline(net, []))
        t0 = time.time()
        explanation = explainer(x=data.x, edge_index=data.edge_index,
                                target=data.y, edge_attr=data.edge_attr)
        elapsed += time.time() - t0
        em = explanation.edge_mask.detach().cpu().numpy()
        half = len(keep_rows)
        pair = np.maximum(em[:half], em[half:2 * half]) if len(em) >= 2 * half else em[:half]
        order = np.argsort(pair)[::-1]
        top = order[:min(args.topk, half)]
        results["pgexplainer"].append(audit_baseline(net, [mapping[keep_rows[j]] for j in top]))
        if args.physical_wrapper:
            truth_rows = [int(j) - 1 for j in np.asarray(exp[i][0], dtype=int).ravel()]
            truth_set = set(truth_rows)
            gt = np.asarray([1 if (row + 1) in truth_set else 0 for row in keep_rows])
            def bacc(selected):
                if np.unique(gt).size < 2:
                    return None
                pred = np.zeros(half, dtype=int)
                pred[selected] = 1
                from sklearn.metrics import balanced_accuracy_score
                return float(balanced_accuracy_score(gt, pred))
            chosen = select_feasible(
                order, pair, mapping, keep_rows, net, args.topk,
                strategy=args.wrapper_strategy, beam_width=args.beam_width,
                candidate_pool=args.candidate_pool)
            if len(chosen) == args.topk:
                wrapped_result = audit_baseline(net, [mapping[keep_rows[j]] for j in chosen])
            else:
                # An incomplete set is not a valid top-k intervention and must
                # count as failure instead of receiving the empty-set baseline.
                wrapped_result = (False, "incomplete", {})
            results["pgexplainer_wrapped"].append(wrapped_result)
            wrapper_rng = random.Random(200000 * args.seed + n)
            random_order = list(range(half))
            wrapper_rng.shuffle(random_order)
            random_chosen = []
            for j in random_order:
                if len(random_chosen) >= args.topk:
                    break
                candidate = random_chosen + [j]
                ok, _, _ = audit_baseline(net, [mapping[keep_rows[k]] for k in candidate])
                if ok:
                    random_chosen = candidate
            if len(random_chosen) == args.topk:
                random_wrapped_result = audit_baseline(
                    net, [mapping[keep_rows[j]] for j in random_chosen])
            else:
                random_wrapped_result = (False, "incomplete", {})
            results["random_wrapped"].append(random_wrapped_result)
            wrapper_size.append(len(chosen))
            random_wrapper_size.append(len(random_chosen))
            wrapper_replace.append(1 - len(set(chosen) & set(top)) / max(1, len(top)))
            raw_score = bacc(top)
            wrapped_score = bacc(chosen) if chosen else bacc([])
            if raw_score is not None and wrapped_score is not None:
                raw_bacc.append(raw_score)
                wrapped_bacc.append(wrapped_score)
                random_wrapped_bacc.append(bacc(random_chosen))
        raw_rng = random.Random(100000 * args.seed + n)
        rnd = raw_rng.sample(range(len(mapping)), min(args.topk, len(mapping)))
        results["random"].append(audit_baseline(net, [mapping[j] for j in rnd]))
        truth_rows = [int(j) - 1 for j in np.asarray(exp[i][0], dtype=int).ravel()]
        results["truth"].append(audit_baseline(net, [mapping[j] for j in truth_rows]))
        if (n + 1) % 10 == 0:
 print(f" {n + 1}/{len(eval_idx)} ")

    summary = {name: pfr_summary(values) for name, values in results.items()}
    summary["_meta"] = {"system": args.dataset, "model": args.model,
        "method": "pgexplainer", "protocol": "unified_protocol", "explanation_type": "phenomenon",
        "train_num": len(train_data),
        "n": len(eval_idx), "eval_start": eval_start, "max_graphs": args.max_graphs,
        "topk": args.topk, "seed": args.seed,
        "epochs": args.epochs, "losses": losses, "runtime_seconds": elapsed,
        "physical_wrapper": args.physical_wrapper,
        "wrapper_strategy": args.wrapper_strategy,
        "beam_width": args.beam_width,
        "candidate_pool": args.candidate_pool,
        "wrapper_mean_edges": float(np.mean(wrapper_size)) if wrapper_size else None,
        "random_wrapper_mean_edges": float(np.mean(random_wrapper_size)) if random_wrapper_size else None,
        "wrapper_mean_replace_rate": float(np.mean(wrapper_replace)) if wrapper_replace else None,
        "wrapper_raw_bacc": float(np.mean(raw_bacc)) if raw_bacc else None,
        "wrapper_bacc": float(np.mean(wrapped_bacc)) if wrapped_bacc else None,
        "random_wrapper_bacc": float(np.mean(random_wrapped_bacc)) if random_wrapped_bacc else None,
        "wrapper_bacc_valid_n": len(raw_bacc)}
    if args.physical_wrapper:
        summary["_meta"]["wrapper_incomplete_n"] = int(sum(
            1 for r in results["pgexplainer_wrapped"] if r[1] == "incomplete"))
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    for name in ("pgexplainer", "random", "truth"):
        print(f"{name}: PFR={summary[name].get('PFR', 0):.3f}")
    if args.physical_wrapper:
        print(f"pgexplainer_wrapped: PFR={summary['pgexplainer_wrapped'].get('PFR', 0):.3f} "
              f"mean_edges={np.mean(wrapper_size):.2f}/{args.topk} "
              f"replace_rate={np.mean(wrapper_replace):.3f} "
              f"bacc_valid={len(raw_bacc)} "
              f"bacc={np.mean(raw_bacc) if raw_bacc else float('nan'):.3f}->"
              f"{np.mean(wrapped_bacc) if wrapped_bacc else float('nan'):.3f}")
        print(f"random_wrapped: PFR={summary['random_wrapped'].get('PFR', 0):.3f} "
              f"mean_edges={np.mean(random_wrapper_size):.2f}/{args.topk} "
              f"bacc={np.mean(random_wrapped_bacc) if random_wrapped_bacc else float('nan'):.3f}")
 print(f" : {args.output}")

if __name__ == "__main__":
    main()

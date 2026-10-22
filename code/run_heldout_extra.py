import argparse, json, os, random, sys, time
import numpy as np
import torch
ROOT = os.environ.get("RESS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "datasets", "PowerGraph-XAI", "src")
sys.path.insert(0, SRC); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gnn.model import get_gnnNets
from audit_core import build_net, audit_baseline, pfr_summary
from run_audit import load_sample
DATA_ROOT = os.path.join(ROOT, "datasets", "PowerGraph", "data_cascades")

def split_test_indices(n, seed):
    nt, nv = int(0.85*n), int(0.05*n)
    idx = torch.randperm(n, generator=torch.default_generator.manual_seed(seed)).tolist()
    return set(idx[nt+nv:])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["ieee24","ieee39","uk"])
    ap.add_argument("--method", required=True, choices=["gsat","subgraphx"])
    ap.add_argument("--checkpoint", required=True); ap.add_argument("--num", type=int, required=True)
    ap.add_argument("--topk", type=int, default=5); ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--train_size", type=int, default=500); ap.add_argument("--model_save_dir", default=None)
    ap.add_argument("--output", required=True); a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(1)
    import mat73
    raw = os.path.join(DATA_ROOT, a.dataset, "raw")
    blist = np.asarray(mat73.loadmat(os.path.join(raw,"blist.mat"))["bList"], dtype=int)
    Ef = mat73.loadmat(os.path.join(raw,"Ef.mat"))["E_f_post"]
    exp = mat73.loadmat(os.path.join(raw,"exp.mat"))["explainations"]
    of_bi = mat73.loadmat(os.path.join(raw,"of_bi.mat"))["output_features"]
    Bf = mat73.loadmat(os.path.join(raw,"Bf.mat"))["B_f_tot"]
    net, mapping = build_net(a.dataset, root=ROOT)
    model = get_gnnNets(3,2,{"model_name":"gin","num_layers":3,"hidden_dim":32,"dropout":0.1,"readout":"max","edge_dim":4},"binary_classification")
    model.load_state_dict(torch.load(a.checkpoint, map_location="cpu", weights_only=False)["net"]); model.eval()
    n_all = len(of_bi); test = split_test_indices(n_all, a.seed)
    pos = [i for i in range(n_all) if int(np.asarray(of_bi[i][0]).ravel()[0]) == 1 and exp[i][0] is not None]
    eval_pool = [i for i in pos if i in test]; random.Random(1000+a.seed).shuffle(eval_pool)
    eval_idx = eval_pool[:min(a.num, len(eval_pool))]
    if len(eval_idx) < a.num: print("warning: only", len(eval_idx), "test positives")
    train_pos = [i for i in pos if i not in test]
    neg_pool = [i for i in range(n_all) if int(np.asarray(of_bi[i][0]).ravel()[0]) == 0 and i not in test]
    random.Random(2000+a.seed).shuffle(train_pos); random.Random(3000+a.seed).shuffle(neg_pool)
    train_idx = (train_pos + neg_pool)[:min(a.train_size, len(train_pos)+len(neg_pool))]
    print(f"[{a.dataset}/{a.method}] test positives={len(eval_pool)}, eval={len(eval_idx)}, train={len(train_idx)}", flush=True)
    res = {"method": [], "random": [], "truth": []}; t_explain = 0.0
    def audit_set(sel, keep): return audit_baseline(net, [mapping[keep[j]] for j in sel])
    def controls(n, i, keep):
        rng = random.Random(100000*a.seed+n); rnd = rng.sample(range(len(keep)), min(a.topk,len(keep)))
        truth = [j for j in range(len(keep)) if int(keep[j]) in set(int(x)-1 for x in np.asarray(exp[i][0],dtype=int).ravel())]
        return audit_set(rnd,keep), audit_set(truth,keep)
    if a.method == "gsat":
        from run_gsat_audit import gsat_explain_graph, SubsetDataset
        train_datas = []
        for i in train_idx:
            d,_ = load_sample(i,blist,Ef,exp,of_bi,Bf); d.edge_mask=d.edge_mask.view(-1)
            d.batch=torch.zeros(d.num_nodes,dtype=torch.long); train_datas.append(d)
        train_ds = SubsetDataset(train_datas)
        save_dir = a.model_save_dir or os.path.join(ROOT,"outputs","gsat_heldout_ckpt"); os.makedirs(save_dir,exist_ok=True)
        kwargs = {"dataset":train_ds,"dataset_name":a.dataset,"seed":a.seed,"num_classes":2,"hidden_dim":32,
                  "batch_size":8,"model_save_dir":save_dir,"model_name":"gin","gsat_epochs":None}
        for n,i in enumerate(eval_idx):
            data,keep=load_sample(i,blist,Ef,exp,of_bi,Bf); data.edge_mask=data.edge_mask.view(-1)
            rr,tt=controls(n,i,keep); res["random"].append(rr); res["truth"].append(tt)
            t0=time.time(); em_raw,_=gsat_explain_graph(model,data,int(data.y.item()),torch.device("cpu"),**kwargs)
            t_explain += time.time()-t0
            em=np.asarray(em_raw.detach().cpu()) if hasattr(em_raw,"detach") else np.asarray(em_raw)
            half=len(keep); pair=np.maximum(em[:half],em[half:2*half]) if len(em)>=2*half else em[:half]
            res["method"].append(audit_set(list(np.argsort(pair)[::-1][:min(a.topk,half)]),keep))
            if (n+1)%10==0: print(f"  {n+1}/{len(eval_idx)}",flush=True)
    else:
        from explainer.subgraphx import SubgraphX
        sub = SubgraphX(model,2,torch.device("cpu"),num_hops=3,rollout=5,min_atoms=4,expand_atoms=8,
                        high2low=True,sample_num=20,reward_method="mc_shapley",subgraph_building_method="zero_filling",vis=False)
        for n,i in enumerate(eval_idx):
            data,keep=load_sample(i,blist,Ef,exp,of_bi,Bf)
            rr,tt=controls(n,i,keep); res["random"].append(rr); res["truth"].append(tt)
            t0=time.time(); em=np.asarray(sub.explain(data.x,data.edge_index,data.edge_attr,max_nodes=a.topk,label=data.y)); t_explain += time.time()-t0
            half=len(keep); pair=np.maximum(em[:half],em[half:2*half]) if len(em)>=2*half else em[:half]
            res["method"].append(audit_set(list(np.argsort(pair)[::-1][:min(a.topk,half)]),keep))
            if (n+1)%5==0: print(f"  {n+1}/{len(eval_idx)}",flush=True)
    summary={k:pfr_summary(v) for k,v in res.items()}
    summary["_meta"]={"system":a.dataset,"method":a.method,"seed":a.seed,"n":len(eval_idx),
        "n_test_explainable_positive":len(eval_pool),"train_num":len(train_idx),"topk":a.topk,
        "split":"random 85/5/10 test partition","checkpoint":os.path.basename(a.checkpoint),
        "explanation_seconds":t_explain,"note":"GSAT trained only on non-test graphs" if a.method=="gsat" else "training-free on test graphs"}
    with open(a.output,"w") as f: json.dump(summary,f,indent=2,default=float)
    print(json.dumps({k:summary[k].get("PFR") for k in res},indent=2)); print("saved",a.output)
if __name__ == "__main__": main()

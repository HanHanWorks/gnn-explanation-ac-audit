# Interpretable GNN auditing for power-grid fault cascades

Method code for auditing edge-level explanations produced by graph neural
networks (GNNs) on power-grid fault-cascade data. Given an explainer's
top-k edge ranking, the audit interprets the candidate set as a set of
branches to disconnect and evaluates the physical feasibility of that
intervention under an AC power-flow check.

## Repository layout

```
env/   Conda / pip environment lock files for environment reproduction
src/   Method scripts (audit core + per-explainer entry points)
```

## Data dependency

The scripts expect the public **PowerGraph** dataset layout (node-level
cascade data with raw `.mat` files: `blist.mat`, `Ef.mat`, `exp.mat`,
`of_bi.mat`, `Bf.mat`) under `datasets/PowerGraph/data_cascades/<system>/raw`,
and the GNN/explainer source tree under `datasets/PowerGraph-XAI/src`.
Set these up before running; no result outputs are produced by this repo.

Supported operating points are the IEEE 24 / 39 / 118 and UK systems
(Matpower `System.m` files).

## Quick start

```bash
conda env create -f env/environment_powergraph.yml
conda activate powergraph

python src/run_audit.py --dataset ieee24 --num 50 --topk 5
```

Per-explainer entry points follow the same pattern
(`run_subgraphx_audit.py`, `run_gsat_audit.py`, `run_gradient_audit.py`,
`run_supervised_edge_audit.py`, etc.). Checkpoints are loaded from
`checkpoints/<system>/`; provide them explicitly or train your own.

## Notes

- `run_threshold_sensitivity.py` sweeps the top-k size and voltage
  constraint to test explanation-to-intervention stress responses.
- `torch_sparse_compat.py` provides a fallback path for GNN message
  passing on systems where `torch_sparse` is unavailable.
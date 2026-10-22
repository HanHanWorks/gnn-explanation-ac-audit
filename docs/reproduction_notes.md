# Reproduction notes

The archive is organised so that the review can be performed without the GitHub
repository. The raw PowerGraph data and the MATPOWER case files are public and are
not redistributed; set `ROOT` to a directory containing

```
datasets/PowerGraph/data_cascades/{ieee24,ieee39,uk}/raw/{Bf,Ef,exp,of_bi,blist}.mat
datasets/PowerGraph-Node/13_Power_system/{IEEE24,IEEE39,UK}/System.m
```

## Table mapping

| Manuscript table | Source |
|---|---|
| Table 1 | conceptual; no data |
| Table 2 | conceptual; no data |
| Table 3 | `code/audit_core.py`; `mapping/*.csv` |
| Table 4 | `data/results_revision/results_quality/*.json`, `results_pg_quality/*.json` |
| Table 5 | `data/results_json/results_*_unified_v8_*`, `results_*_gradcam_v8_*`, `results_*_gsat_v8_*`, `results_*_subgraphx_v8_*`, `results_quality/*.json` |
| Table 6 | `data/results_revision/results_conn5/*.json` (five seeds); `results_conn/*.json` (seed 2) |
| Table 7 | `data/results_revision/results_matched/*.json`, archived SEP runs |
| Table 8 | `data/figure_csv/figure_4_transfer.csv`; contingency-disjoint GIN runs |
| Table 9 | `results_*_v7r_sensitivity_v10_*`, `results_*_dc_ac_v10_*`, `data/results_revision/results_op/*.json` |
| Table 10 | `data/results_revision/results_wrapper/*.json`, `results_matched/*.json` |
| Appendix C | `data/results_json/results_*_supervised_v9_contingency_*.json` |
| Appendix D | `data/results_revision/results_heldout/*.json` (four original methods plus `*_heldout_gsat_s2.json` and `*_heldout_subgraphx_s2.json`) |
| Figures 2-5 | `data/figure_csv/*.csv` |
| Section 3.8 (Wilcoxon / effect sizes) | `data/results_revision/results_statistics/ress_statistical_revision.json` |

## Commands

```bash
python code/verify_archive.py
python code/export_figure_csv.py
python code/build_revision_summary.py     # writes revision_summary.md/json
python code/make_figures.py
python code/run_heldout_extra.py --help
```

`build_revision_summary.py` and `make_figures.py` read the archive root from
`RESS_ROOT` and default to the archive root; the commands above run from a clean
extraction.

## Statistical notes

- Edge balanced accuracy is pooled over all evaluated edges: TP/FP/TN/FN are summed
  across graphs, then BACC = 0.5*(TPR+TNR). Per-graph averaging is not used because a
  graph with no positive edge has an undefined per-graph BACC.
- PFR intervals are seed-cluster bootstrap intervals (2000 replicates). Table 8 uses a
  t interval with four degrees of freedom for the five observed seeds.
- The permutation null retrains the SEP on shuffled training labels with the same
  split, features, class balance, and solver; ten permutations per seed are reported.
- The connectivity control checks the joint connectivity of the selected set, not the
  bridge status of individual edges.
- The wrapper AC-call count includes the greedy phase and the local-swap phase. The
  `kP = 40` bound covers only the local-swap phase; the correct worst-case bound is
  `N_avail + kP`.

## Legacy operating-point artefacts

Files named `data/results_json/results_*_operating_point_v10_s2_n*.json` are initial-submission artefacts retained only to preserve the 137-artefact provenance. They used an earlier sample-conditioned implementation and are superseded by `data/results_revision/results_op/*_op_raw.json`. Do not use the legacy files for Table 9, Figure 5, or Section 5.7.

The corrected Table 9 operating-point block and the `conditioning` block of `data/results_revision/results_statistics/ress_statistical_revision.json` are generated from the corrected `results_op/*_op_raw.json` files.

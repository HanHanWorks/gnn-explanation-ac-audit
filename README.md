# RESS revision source-data and code archive

Manuscript: JRESS-D-26-06831 (major revision)
Title: From Explanation Agreement to Operability: A Restricted AC Screening Audit of GNN Explanations for Cascading-Failure Risk Assessment

This archive is the authoritative source for every number, table, and figure in the revision. Verification and figure regeneration do not depend on the GitHub repository being reachable during review.

## Contents

```
data/results_json/        285 result JSON files (137 formal indexed artefacts + 148 pilot/auxiliary and phase-0 checkpoint records)
data/results_revision/    new results generated for this revision
  results_quality/        matched-slice SEP, label-free, random, and shuffled-label controls
  results_pg_quality/     PGExplainer per-seed pooled edge agreement and PFR
  results_matched/        IEEE 39 matched 104-graph SEP runs (seeds 2-6)
  results_conn/           joint-connectivity-preserving control (seed 2)
  results_conn5/          joint-connectivity-preserving control (five seeds, 50 samples/seed)
  results_heldout/        held-out-backbone audit on the exact GIN random-test partition (seed 2); includes all six methods, with GSAT retrained on non-test graphs
  results_wrapper/        wrapper AC-call distributions and threshold checks
  results_op/             identical raw top-k sets under public vs sample-conditioned protocols
  results_statistics/      effect-size and Wilcoxon tables used in Section 3.8
  results_split_indices.json  exact per-seed train/eval indices
data/figure_csv/          CSV files used to draw Figures 2-5
revision_summary.md/json   generated aggregation digest used for the revision
figures/Figure1_source.svg  vector source for Figure 1
code/                     analysis, aggregation, figure, and verification scripts
code/build_figure1.sh       regenerates the Figure 1 PDF from the SVG
mapping/                  branch-mapping tables
docs/                     verification log and reproduction notes
manifest.json             the 137 formal artefacts indexed in the initial submission
manifest.csv              tabular index of the same artefacts
cascade_denominators.json population denominators
MANIFEST.sha256           checksums for every file in this archive
```

Figure 1 is maintained as a vector drawing (`figures/Figure1_source.svg`) and converted with
`code/build_figure1.sh` using `rsvg-convert`. The Python scripts generate Figures 2-5 and the
tables. They read the working tree from the `RESS_ROOT` environment variable (default: the
repository root); set it to a checkout containing `server_data/phase1` and
`LaTeX_Source_Files/figures` before running the aggregation and figure scripts.

## Environment

The reported run used Python 3.11.13, numpy 1.26.4, scipy 1.16.3, pandas 2.3.3,
scikit-learn 1.3.2, pandapower 3.5.4, PyTorch 2.2.2, torch-geometric 2.8.0,
matplotlib 3.11.2, statsmodels 0.15.0, networkx 3.6.1, PyYAML 6.0.1, tensorboard 2.21.0. The pinned environment is in
`requirements.txt` and `environment_powergraph.yml`.

The raw PowerGraph data and the MATPOWER case files are public and are not
redistributed here. Set `ROOT` to a directory containing
`datasets/PowerGraph/data_cascades/{ieee24,ieee39,uk}/raw` and
`datasets/PowerGraph-Node/13_Power_system/{IEEE24,IEEE39,UK}/System.m`.

## Verification

```bash
python code/verify_archive.py
```

The script validates every SHA-256 checksum in `MANIFEST.sha256`, compiles every
Python file, counts the 137 formal artefacts, checks the 38/46/99 branch mappings
and the five figure-level CSV files, and confirms that the population-denominator
file is present.

## Reproduction

The supported check is:

```bash
python code/verify_archive.py
```

The aggregation, table, and figure scripts are under `code/`. `code/run_heldout_extra.py` reruns the GSAT/SubgraphX exact-test-partition audit; `code/make_figures.py` regenerates Figures 2--5. They read the
archive root from the `RESS_ROOT` environment variable and default to the archive
root, so the commands above run from a clean extraction. The mapping from each
manuscript table and figure to its source files, and the commands used in the
revision, are listed in `docs/reproduction_notes.md`.

## Formal versus auxiliary artefacts

The initial submission indexed 137 experiment JSON artefacts. All 137 are present
in `data/results_json/` and listed in `manifest.json`. The directory also contains
148 pilot, smoke-test, phase-0 checkpoint, and unindexed artefacts that were retained for transparency;
they are not used for any formal claim. `verify_archive.py` checks the 137 formal
files separately.

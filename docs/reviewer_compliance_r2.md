# Reviewer compliance check (R2)

Status: all reviewer comments have a corresponding manuscript/response change and, where needed, an implemented analysis change. The only limitation is that the exact-test-partition held-out audit in Appendix D is seed-2 (the same design and compute budget as the original Appendix D), not a five-seed held-out replication; this is stated explicitly in the manuscript and response.

| ID | Reviewer requirement | Current status | Evidence / caveat |
|---|---|---|---|
| R1.1 | Repository must contain the indexed JSONs, figure CSVs, and executable scripts; availability statements must match | Addressed | GitHub repository updated from the current archive; `verify_archive.py` passes 137/137 formal artefacts, 438 checksums, and all 47 Python files compile. `revision_summary.md/json` regenerated and included. |
| R1.2 | Remove incorrect PDF hyphenation | Addressed | `main.tex` uses `\hyphenpenalty=10000` and `\exhyphenpenalty=10000`; the final response describes the measure without a misleading numeric rendering. |
| R1.3 | Define BACC and explain its role | Addressed | Eq. (2), `A_edge`; graph-level `A_graph` defined separately; pooled rationale and permutation null in Sections 3.2/5.1; Table 1 states what agreement does and does not support. |
| R1.4 | Make Figure 2 (now Figure 3) intervals visible and text readable | Addressed | Figure 3 redrawn with visible Wilson intervals, larger labels, two-row layout, marker/legend encoding; caption distinguishes display Wilson intervals from seed-cluster bootstrap intervals. |
| R1.5 | Clarify single-backbone limitation | Addressed | Section 4.2/Limitations; Transformer pilot in Appendix A; no backbone-independent claim. |
| R1.6 | Clarify or remove the Overload column | Addressed | Overload column removed from Table 5; Table 3/Figure 3 caption explain thermal loading is recorded but not a rejection criterion. |
| R1.7 | Use consistent risk notation | Addressed | Only `R_sev` is used in the manuscript; `mean_risk` is mentioned once as the archive field mapping. Section 5.3 reports `R_sev` for the rankings with per-instance voltage detail. |
| R2.1 | Separate faithfulness, label agreement, intervention feasibility; reconsider title; report fidelity | Addressed | Title changed; Table 1 separates prediction/agreement/fidelity/screening/burden; Appendix B reports Fid+/Fid-; explicit statement that a failed screen does not establish poor explanation quality. |
| R2.2 | Resolve operating-point and graph-state inconsistencies | Addressed | Table 2 specifies topology, injections, outage treatment and purpose at prediction, explanation, contingency signature, public screen, sample-conditioned screen, and wrapper. Primary public case does not retain the initiating outage; sample-conditioned does. |
| R2.3 | Define restricted physical criterion, solver details, mapping, non-convergence | Addressed | Table 3 gives solver, initialization, tolerance, exact iteration limit, active/reactive limits, slack, voltage, island and thermal rules; branch mapping counts and one-to-one parallel mapping stated; non-convergence separated from infeasibility. |
| R2.4 | Matched evaluation samples, common intersection, matched random controls, justify eval_start=50 | Addressed | Matched slice and purge gap documented; Table 7 reports both IEEE 39 slices; matched random controls in Table 5; exact-test-partition audit in Appendix D now covers all six methods. Caveat: Appendix D is seed-2 (50/16/50 graphs), not five-seed. |
| R2.5 | Evaluate quality-feasibility relationship by system, expand coverage, avoid overclaiming non-significance | Addressed | Figure 2 sample-level plot; by-system descriptive Spearman values -0.05/+0.45/-0.73 with explicit non-inferential caveat; counterexample framing; no p-value claim; six rankings with per-edge scores included. |
| R2.6 | State interval populations, justify Table 2 intervals, supply missing effect-size quantities | Addressed | Table 8 logit-scale t intervals with 4 df and values 0.690-0.943, 0.470-0.928, and 0.971-0.999, all inside [0,1]; Section 3.8 states populations; Table 5 contains method-minus-random differences, paired effect sizes and bootstrap intervals; Wilcoxon limitations stated. |
| R2.7 | Audit identical raw top-k sets under both operating-point protocols; report base states; qualify cap claim; relabel Figure 5c | Addressed | Section 5.7 uses identical raw sets and reports public/sample/base rates; Figure 5c relabelled; cap statement qualified; sample-conditioned `audit()` fixed for impedance and island handling and rerun on all three systems. Corrected raw SEP public/sample: 0.000/0.000, 0.010/0.000, 0.700/0.080; sample base: 0.970/0.885/1.000. |
| R2.8 | Consolidated baseline table, oracle designation, repeated shuffled labels and split checks | Addressed | Table 5 includes random, all explainers, |Q|, |P|+|Q|, SEP and truth anchor, with denominators/seeds/counts/missing entries and method-specific random controls; SEP naming and no-oracle claims; permutation null with ten shuffles per seed; SEP values 0.900/0.760/0.870 exceed null maxima 0.866/0.746/0.707; training/evaluation split archived. |
| R2.9 | Pseudocode, corrected AC-call bound, call distribution plus runtime, connectivity-preserving control | Addressed | Algorithm 1 gives greedy/local-swap/incomplete logic; bound corrected to N_avail + kP; Table 10 now includes mean, median, max, p90 and p95 under the same seed-2 first-20-graph workload as runtime and excludes the initial base audit; joint-connectivity control in Sections 5.4/Table 6. |
| R2.10 | Reconcile internal statements, expose Figure intervals, report threshold results | Addressed | Contradictory "no voltage/convergence records" sentence removed; pooled IEEE 24 SEP taxonomy is 2 passes + 458 islands + 226 non-convergences + 314 voltage violations = 1000; the 0.000 threshold result is separately labelled seed-2 n=200; Figure 3 intervals visible; Table 9 and Section 5.8 report 0.85/0.90/0.95 p.u. results; Table 9/Figure 5 captions state n=200/104/200. |

## Remaining caveats, not blocking requirements

- Appendix D exact-test-partition audit is one seed (2) and 50/16/50 graphs; the five-seed main Table 5 remains a random-split benchmark. The paper states this in Table 5, Appendix D, Limitations, and the response.
- The held-out GSAT/SubgraphX rows in Table D.1 are PFR-only; per-edge held-out agreement for those two methods is not added because their per-edge masks are not in the held-out agreement table.
- No new experiment is required by the reviewer comments beyond the already implemented sample-conditioned and all-method held-out runs.

## Verification commands

```bash
python code/verify_archive.py
cd paper && tectonic main.tex
```

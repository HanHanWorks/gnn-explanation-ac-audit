#!/usr/bin/env python3
"""Generate Figures 2-5 from archived/new result files (Figure 1 is SVG)."""

# Figure 1 is a vector drawing maintained as figures/Figure1_source.svg
# and converted with code/build_figure1.sh; this script generates Figures 2-5.

from __future__ import annotations

import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

ROOT = os.environ.get("RESS_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

SYS = ["ieee24", "ieee39", "uk"]
SYS_LABEL = {"ieee24": "IEEE 24", "ieee39": "IEEE 39", "uk": "UK"}
METHODS = ["Random", "GNNExplainer", "PGExplainer", "Grad-CAM", "GSAT",
           "SubgraphX", "SEP", "Truth"]
COLORS = {
    "Random": "#9e9e9e", "GNNExplainer": "#2a6f9e", "PGExplainer": "#c0504d",
    "Grad-CAM": "#8064a2", "GSAT": "#e69f00", "SubgraphX": "#4d9221",
    "SEP": "#000000", "Truth": "#7f3b08", "|Q|": "#1b9e77",
    "|P|+|Q|": "#66c2a5",
}


def load(path):
    with open(path) as f:
        return json.load(f)


def archive_method(method, system):
    base = os.path.join(ROOT, "data", "results_json")
    pats = {
        "GNNExplainer": {
            "ieee24": "results_ieee24_gin_unified_v8_eval50_s{}.json",
            "ieee39": "results_ieee39_gin_unified_v8_eval50_s{}.json",
            "uk": "results_uk_gin_unified_v8_full_eval50_s{}.json",
        },
        "PGExplainer": {
            "ieee24": "results_ieee24_pg_unified_v8_eval50_s{}.json",
            "ieee39": "results_ieee39_pg_unified_v8_eval50_s{}.json",
            "uk": "results_uk_pg_unified_v8_full_eval50_s{}.json",
        },
        "Grad-CAM": {
            "ieee24": "results_ieee24_gradcam_v8_s{}_n200.json",
            "ieee39": "results_ieee39_gradcam_v8_s{}_n104.json",
            "uk": "results_uk_gradcam_v8_s{}_n200.json",
        },
        "GSAT": {
            "ieee24": "results_ieee24_gsat_v8_s{}_n200.json",
            "ieee39": "results_ieee39_gsat_v8_s{}_n104.json",
            "uk": "results_uk_gsat_v8_s2_n200.json",
        },
        "SubgraphX": {
            "ieee24": "results_ieee24_subgraphx_v8_s2_n200.json",
            "ieee39": "results_ieee39_subgraphx_v8_s2_n104.json",
            "uk": "results_uk_subgraphx_v8_pilot50_s2.json",
        },
    }
    keys = {"GNNExplainer": "gnn_explainer", "PGExplainer": "pgexplainer",
            "Grad-CAM": "gradcam", "GSAT": "gsat", "SubgraphX": "subgraphx"}
    seeds = [2, 3, 4, 5, 6]
    if method == "GSAT":
        seeds = [2, 3, 4] if system != "uk" else [2]
    if method == "SubgraphX":
        seeds = [2]
    rows = []
    for s in seeds:
        p = os.path.join(base, pats[method][system].format(s))
        if os.path.exists(p):
            d = load(p)
            v = d[keys[method]]
            rows.append((v["n"], v["types"], v["PFR"]))
    return rows


def pooled_from_rows(rows):
    n = sum(r[0] for r in rows)
    c = {}
    for _, t, _ in rows:
        for k, v in t.items():
            c[k] = c.get(k, 0) + v
    return n, c


def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0, centre - half), min(1, centre + half)


def quality(system):
    return load(os.path.join(ROOT, "data", "results_revision", "results_quality", f"{system}_quality.json"))


def fig_outcomes():
    methods = ["Random", "GNNExplainer", "PGExplainer", "Grad-CAM", "GSAT",
               "SubgraphX", "SEP", "Truth"]
    mpl.rcParams.update({"font.size": 11.0, "axes.titlesize": 11.0,
                         "axes.labelsize": 10.5, "xtick.labelsize": 10.0,
                         "ytick.labelsize": 10.0, "legend.fontsize": 9.0})
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.2, 1.0], hspace=0.55, wspace=0.32)
    for j, system in enumerate(SYS):
        ax = fig.add_subplot(gs[0, j])
        q = quality(system)
        y = np.arange(len(methods))
        for i, m in enumerate(methods):
            if m == "SEP":
                n = sum(s["methods"]["SEP"]["pfr"]["n"] for s in q["seeds"].values())
                k = sum(s["methods"]["SEP"]["pfr"]["types"].get("feasible", 0) for s in q["seeds"].values())
                p = k / n
                lo, hi = wilson(k, n)
            elif m == "Truth":
                n = sum(s["methods"]["truth"]["pfr"]["n"] for s in q["seeds"].values())
                k = sum(s["methods"]["truth"]["pfr"]["types"].get("feasible", 0) for s in q["seeds"].values())
                p = k / n
                lo, hi = wilson(k, n)
            elif m == "Random":
                n = sum(s["methods"]["random"]["pfr"]["n"] for s in q["seeds"].values())
                k = sum(s["methods"]["random"]["pfr"]["types"].get("feasible", 0) for s in q["seeds"].values())
                p = k / n
                lo, hi = wilson(k, n)
            else:
                rows = archive_method(m, system)
                if not rows:
                    continue
                n, c = pooled_from_rows(rows)
                k = c.get("feasible", 0)
                p = k / n
                if len(rows) == 1:
                    lo = hi = p
                else:
                    lo, hi = wilson(k, n)
            ax.errorbar(p, i, xerr=[[p - lo], [hi - p]], fmt="o", ms=3.8,
                        color=COLORS[m], capsize=1.6, lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(methods, fontsize=9.8)
        ax.invert_yaxis()
        ax.set_xlim(-0.02, 1.04)
        ax.set_xlabel("Screening pass rate")
        ax.set_title(SYS_LABEL[system], fontsize=11.5)
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
    for j, system in enumerate(SYS):
        ax = fig.add_subplot(gs[1, j])
        q = quality(system)
        bottom = np.zeros(len(methods))
        cats = ["feasible", "island", "non_converged", "voltage"]
        cols = {"feasible": "#4d9221", "island": "#c0504d",
                "non_converged": "#e69f00", "voltage": "#2a6f9e"}
        for c in cats:
            vals = []
            for m in methods:
                if m == "SEP":
                    vals.append(sum(s["methods"]["SEP"]["pfr"]["types"].get(c, 0) for s in q["seeds"].values()))
                elif m == "Truth":
                    vals.append(sum(s["methods"]["truth"]["pfr"]["types"].get(c, 0) for s in q["seeds"].values()))
                elif m == "Random":
                    vals.append(sum(s["methods"]["random"]["pfr"]["types"].get(c, 0) for s in q["seeds"].values()))
                else:
                    rows = archive_method(m, system)
                    vals.append(sum(r[1].get(c, 0) for r in rows))
            vals = np.asarray(vals, dtype=float)
            tot = np.array([sum(vals[:i + 1]) for i in range(len(vals))])
            ax.barh(np.arange(len(methods)), vals, left=bottom, color=cols[c],
                    height=0.72, label=c.replace("_", " "))
            bottom += vals
        ax.set_yticks(np.arange(len(methods)))
        ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel("Audited samples by outcome")
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
        ax.set_axisbelow(True)
        if j == 2:
            handles, labels_ = ax.get_legend_handles_labels()
            fig.legend(handles, labels_, loc="lower center", ncol=4, frameon=False, fontsize=9.8,
                       bbox_to_anchor=(0.5, -0.015), columnspacing=1.6, handlelength=1.4)
    fig.text(0.5, 1.01, "(a) Pooled pass rate and 95% Wilson interval", ha="center", fontsize=11.5, weight="bold")
    
    fig.savefig(os.path.join(OUT, "figure_2_outcomes.pdf"))
    plt.close(fig)


def fig_quality():
    mpl.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5,
                         "axes.labelsize": 10.0, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "legend.fontsize": 8.5})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.05), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    markers = {"ieee24": "o", "ieee39": "s", "uk": "^"}
    for system in SYS:
        q = quality(system)
        for m, key in [("SEP", "SEP"), ("|Q|", "absQ"), ("|P|+|Q|", "absPQ"),
                       ("Random", "random"), ("Truth", "truth")]:
            xs = [s["methods"][key]["bacc"] for s in q["seeds"].values()]
            ys = [s["methods"][key]["pfr"]["PFR"] for s in q["seeds"].values()]
            ax.scatter(xs, ys, marker=markers[system], s=16, color=COLORS[m],
                       alpha=0.65, linewidths=0.3, edgecolors="white")
    # PGExplainer pooled points if available
    for system in SYS:
        pg = os.path.join(ROOT, "data", "results_revision", "results_pg_quality")
        xs, ys = [], []
        for s in [2, 3, 4, 5, 6]:
            p = os.path.join(pg, f"{system}_pg_s{s}.json")
            if os.path.exists(p):
                d = load(p)
                xs.append(d["edge_label_agreement"]["pooled_bacc"])
                ys.append(d["pgexplainer"]["PFR"])
        if xs:
            ax.scatter(xs, ys, marker=markers[system], s=18, color=COLORS["PGExplainer"],
                       alpha=0.8, linewidths=0.3, edgecolors="white")
    ax.set_xlabel("Pooled edge-label agreement ($A_{\\mathrm{edge}}$)")
    ax.set_ylabel("Restricted AC screening pass rate")
    ax.set_xlim(0.35, 1.05)
    ax.set_ylim(-0.04, 1.04)
    ax.grid(color="#e5e5e5", lw=0.5)
    ax.set_axisbelow(True)
    handles = [plt.Line2D([], [], marker="o", ls="", color=COLORS[m], label=m)
               for m in ["SEP", "|Q|", "|P|+|Q|", "PGExplainer", "Random", "Truth"]]
    # legend moved below the panel to avoid covering data
    # (added to the figure after the second panel is drawn)
    ax.set_title("(a) Agreement versus screening pass rate", fontsize=10.5)

    ax = axes[1]
    data = []
    labels = []
    for system in SYS:
        q = quality(system)
        vals = []
        for s in q["seeds"].values():
            vals.extend(s["shuffled_label_bacc"]["values"])
        data.append(vals)
        labels.append(SYS_LABEL[system])
    bp = ax.boxplot(data, tick_labels=labels, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", lw=0.8),
                    boxprops=dict(facecolor="#d9d9d9", lw=0.6),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    flierprops=dict(marker=".", ms=2, lw=0))
    for i, system in enumerate(SYS):
        q = quality(system)
        sep = np.mean([s["methods"]["SEP"]["bacc"] for s in q["seeds"].values()])
        abq = np.mean([s["methods"]["absQ"]["bacc"] for s in q["seeds"].values()])
        ax.scatter([i + 1], [sep], marker="D", s=22, color="black", zorder=5, label="SEP" if i == 0 else None)
        ax.scatter([i + 1], [abq], marker="o", s=20, color=COLORS["|Q|"], zorder=5, label="|Q|" if i == 0 else None)
    ax.set_ylabel("Pooled $A_{\\mathrm{edge}}$")
    ax.set_ylim(0.3, 1.02)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=True, facecolor="white", edgecolor="#999999",
              framealpha=1.0, fontsize=7.8, loc="lower right",
              borderpad=0.5, handletextpad=0.5, borderaxespad=0.25, labelspacing=0.35)
    ax.set_title("(b) Shuffled-label permutation null", fontsize=10.5)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8.0,
               bbox_to_anchor=(0.5, -0.035))
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.savefig(os.path.join(OUT, "figure_3_quality.pdf"))
    plt.close(fig)


def fig_transfer():
    seeds = [2, 3, 4, 5, 6]
    data = {
        "ieee24": [0.934, 0.928, 0.730, 0.859, 0.729],
        "ieee39": [0.540, 0.900, 0.598, 0.691, 0.927],
        "uk": [0.999, 0.966, 0.989, 0.999, 0.996],
    }
    scenarios = {"ieee24": [9, 6, 6, 6, 8], "ieee39": [9, 6, 9, 9, 9],
                 "uk": [27, 15, 13, 27, 27]}
    random_mean = {"ieee24": 0.970, "ieee39": 0.926, "uk": 0.971}
    mpl.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5,
                         "axes.labelsize": 10.0, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "legend.fontsize": 8.5})
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0], wspace=0.32)
    ax = fig.add_subplot(gs[0, 0])
    for system in SYS:
        ax.plot([0, 1], [random_mean[system], np.mean(data[system])],
                marker="o", ms=3.5, lw=0.9, color={"ieee24": "#2a6f9e", "ieee39": "#c0504d", "uk": "#4d9221"}[system],
                label=SYS_LABEL[system])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Random\nsplit", "Contingency-\ndisjoint"])
    ax.set_ylabel("Test balanced accuracy")
    ax.set_ylim(0.45, 1.03)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    ax.set_title("(a) Mean transfer")
    ax.legend(frameon=False, fontsize=6.5)
    ax = fig.add_subplot(gs[0, 1])
    for i, system in enumerate(SYS):
        v = np.asarray(data[system])
        ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v,
                   s=14, color={"ieee24": "#2a6f9e", "ieee39": "#c0504d", "uk": "#4d9221"}[system],
                   alpha=0.8, edgecolors="white", linewidths=0.3)
        ax.plot([i - 0.2, i + 0.2], [v.mean(), v.mean()], color="black", lw=0.8)
        ax.plot([i - 0.2, i + 0.2], [v.min(), v.min()], color="black", lw=0.4, ls=":")
        ax.plot([i - 0.2, i + 0.2], [v.max(), v.max()], color="black", lw=0.4, ls=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels([SYS_LABEL[s] for s in SYS], fontsize=7)
    ax.set_ylabel("Seed-level BACC")
    ax.set_ylim(0.45, 1.03)
    ax.grid(axis="y", color="#e5e5e5", lw=0.5)
    ax.set_title("(b) Seed variability")
    ax = fig.add_subplot(gs[0, 2])
    for i, system in enumerate(SYS):
        ax.scatter(scenarios[system], data[system], s=14,
                   color={"ieee24": "#2a6f9e", "ieee39": "#c0504d", "uk": "#4d9221"}[system],
                   alpha=0.8, edgecolors="white", linewidths=0.3)
    ax.set_xlabel("Held-out test scenarios per seed")
    ax.set_ylabel("Test BACC")
    ax.set_ylim(0.45, 1.03)
    ax.grid(color="#e5e5e5", lw=0.5)
    ax.set_title("(c) Held-out evidence")
    fig.savefig(os.path.join(OUT, "figure_4_transfer.pdf"))
    plt.close(fig)


def fig_protocol():
    systems = SYS
    ks = [1, 3, 5, 8]
    sep_k = {"ieee24": [0, 0, 0, 0], "ieee39": [1.0, 0.1923, 0.0096, 0.0],
             "uk": [1.0, 1.0, 0.700, 0.055]}
    rand_k = {"ieee24": [0.975, 0.805, 0.520, 0.170],
              "ieee39": [0.769, 0.279, 0.087, 0.0],
              "uk": [1.0, 1.0, 1.0, 1.0]}
    sep_thr = {"ieee24": [0, 0, 0], "ieee39": [0.0096, 0.0096, 0.0096],
               "uk": [0.700, 0.700, 0.700]}
    rand_thr = {"ieee24": [0.520, 0.320, 0.005], "ieee39": [0.087, 0.067, 0.029],
                "uk": [1.0, 1.0, 0.990]}
    op = {}
    for s_ in systems:
        p = os.path.join(ROOT, "data", "results_revision", "results_op", f"{s_}_op_raw.json")
        if os.path.exists(p):
            op[s_] = load(p)
    mpl.rcParams.update({"font.size": 10.5, "axes.titlesize": 10.5,
                         "axes.labelsize": 10.0, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 9.5, "legend.fontsize": 8.5})
    cols = {"ieee24": "#2a6f9e", "ieee39": "#c0504d", "uk": "#4d9221"}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.8))
    ax = axes[0, 0]
    for s_ in systems:
        ax.plot(ks, sep_k[s_], marker="o", ms=3.5, lw=1.0, color=cols[s_], label=SYS_LABEL[s_] + " SEP")
        ax.plot(ks, rand_k[s_], marker="o", ms=3.0, lw=0.8, ls="--", color=cols[s_], alpha=0.75)
    ax.set_xlabel("Intervention budget $k$")
    ax.set_ylabel("Screening pass rate")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(color="#e5e5e5", lw=0.5)
    ax.set_title("(a) Raw SEP (solid) and random (dashed)")
    sys_handles = [plt.Line2D([], [], color=cols[s_], lw=1.2, label=SYS_LABEL[s_])
                   for s_ in systems]
    ax.legend(handles=sys_handles, frameon=True, facecolor="white",
              edgecolor="#999999", framealpha=1.0, fontsize=8.0,
              loc="upper right", borderpad=0.5, borderaxespad=0.3)
    ax = axes[0, 1]
    thr = [0.85, 0.90, 0.95]
    for s_ in systems:
        ax.plot(thr, sep_thr[s_], marker="o", ms=3.5, lw=1.0, color=cols[s_], label=SYS_LABEL[s_] + " SEP")
        ax.plot(thr, rand_thr[s_], marker="s", ms=2.8, lw=0.8, ls="--", color=cols[s_], alpha=0.75)
    ax.set_xticks(thr)
    ax.set_xlabel("Lower voltage limit (p.u.)")
    ax.set_ylabel("Screening pass rate")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("(b) SEP (solid) and random (dashed), $k=5$")
    ax.grid(color="#e5e5e5", lw=0.5)
    ax.legend(handles=sys_handles, frameon=True, facecolor="white",
              edgecolor="#999999", framealpha=1.0, fontsize=8.0,
              loc="center right", borderpad=0.5, borderaxespad=0.3)
    ax = axes[1, 0]
    if op:
        labels = [SYS_LABEL[s_] for s_ in systems if s_ in op]
        y = np.arange(len(labels))
        ax.scatter([op[s_]["public_raw"]["PFR"] for s_ in systems if s_ in op], y + 0.16,
                   marker="o", s=26, color="#888888", label="Public raw")
        ax.scatter([op[s_]["sample_raw"]["PFR"] for s_ in systems if s_ in op], y,
                   marker="s", s=26, color="#2a6f9e", label="Sample raw")
        ax.scatter([op[s_]["sample_base"]["PFR"] for s_ in systems if s_ in op], y - 0.16,
                   marker="D", s=22, color="#b8cfe0", edgecolor="#2a6f9e", label="Sample base")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlim(-0.02, 1.05)
        ax.set_xlabel("Pass rate")
        ax.set_title("(c) Identical raw top-5 sets")
        ax.grid(axis="x", color="#e5e5e5", lw=0.5)
    ax = axes[1, 1]
    dcs = []
    sps = []
    for s_ in systems:
        if s_ == "ieee24":
            dcs.append(0.510 - 0.0); sps.append(0.710 - 0.0)
        elif s_ == "ieee39":
            dcs.append(0.0096 - 0.0096); sps.append(1.0 - 0.0096)
        else:
            dcs.append(0.710 - 0.700); sps.append(0.915 - 0.700)
    y = np.arange(len(systems))
    ax.scatter(dcs, y + 0.10, marker="o", s=26, c=[cols[s_] for s_ in systems], label="DC - AC")
    ax.scatter(sps, y - 0.10, marker="s", s=24, c=[cols[s_] for s_ in systems], alpha=0.7, label="Sample - public")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([SYS_LABEL[s_] for s_ in systems], fontsize=7)
    ax.set_xlabel("Signed difference in pass rate")
    ax.set_title("(d) Protocol effects (SEP, seed 2)")
    ax.grid(axis="x", color="#e5e5e5", lw=0.5)
    handles, labels_ = axes[1, 0].get_legend_handles_labels()
    handles2, labels2 = axes[1, 1].get_legend_handles_labels()
    fig.legend(handles + handles2, labels_ + labels2, loc="lower center", ncol=4,
               frameon=False, fontsize=8.0, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(OUT, "figure_5_protocol.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    fig_outcomes()
    fig_quality()
    fig_transfer()
    fig_protocol()
    print("figures written to", OUT)

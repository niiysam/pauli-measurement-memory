"""Trajectory-level uncertainty and plots; never fit a critical point."""
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.reproduce import OUT, ROOT, write_csv, dump
from src.ensemble import analytic_index

FIG = ROOT/"figures"/"gate_a_plus"


def estimate(values):
    a = np.asarray(values, float)
    mean = float(np.mean(a))
    half = float(t.ppf(.975, len(a)-1)*np.std(a, ddof=1)/np.sqrt(len(a)))
    return mean, half


def key(s):
    return s["family"], s["L"], s["q"], s["c"], s["initial"], s["boundary"]


def analyze():
    FIG.mkdir(parents=True, exist_ok=True)
    stages = [s for s in ("baseline", "pilot") if (OUT/f"{s}_trajectories.json").exists()]
    runs = [s for stage in stages for s in json.loads(
        (OUT/f"{stage}_trajectories.json").read_text())]
    samples = []
    for stage in stages:
        with (OUT/f"{stage}_samples.csv").open() as f:
            samples.extend(dict(run=r["run"], clock=r["clock"],
                                time=float(r[r["clock"]+"_sweeps"]),
                                entropy=float(r["entropy"])) for r in csv.DictReader(f))
    by_run = {s["run"]: s for s in runs}
    grouped = defaultdict(list)
    for s in runs:
        grouped[key(s)].append(s)
    summary, diagnostics, paired = [], [], []
    for k, group in sorted(grouped.items()):
        family, L, q, c, init, bc = k
        row = dict(family=family, L=L, q=q, c=c, initial=init, boundary=bc,
                   n=len(group))
        for field in ("late1", "late2", "drift", "active_fraction",
                      "same_global_fraction", "same_anchor_fraction"):
            row[field], row[field+"_95half"] = estimate([s[field] for s in group])
        row["I_iid_PBC"] = (analytic_index(L, 3, group[0]["pi"])
                           if bc == "periodic" else "")
        row["repeat_reactivated_count"] = sum(
            s["repeated_anchor_changed_after_intervening"] for s in group)
        summary.append(row)
        for a in range(3):
            values = [s["label_counts"][a]/sum(s["label_counts"]) for s in group]
            m, h = estimate(values)
            diagnostics.append(dict(family=family, L=L, q=q, c=c, initial=init,
                                    metric=f"frequency_{'XYZ'[a]}", observed=m,
                                    CI95half=h, expected=group[0]["pi"][a], n=len(group)))
            if group[0]["pi"][a] > 0:
                for b in range(3):
                    vv = [s["transitions"][a][b]/sum(s["transitions"][a])
                          for s in group if sum(s["transitions"][a]) > 0]
                    m, h = estimate(vv)
                    diagnostics.append(dict(family=family, L=L, q=q, c=c, initial=init,
                                            metric=f"T_{'XYZ'[a]}{'XYZ'[b]}",
                                            observed=m, CI95half=h,
                                            expected=c*(a==b)+(1-c)*group[0]["pi"][b],
                                            n=len(vv)))
        D = 1-np.dot(group[0]["pi"], group[0]["pi"])
        m, h = estimate([s["hist_anti"]/s["hist_pairs"] for s in group])
        diagnostics.append(dict(family=family, L=L, q=q, c=c, initial=init,
                                metric="C_hist_visit_lag1", observed=m, CI95half=h,
                                expected=(1-(1-2*(1-c)*D)**3)/2, n=len(group)))
        if c == .6:
            base = grouped[(family, L, q, 0, init, bc)]
            base = {s["rep"]: s for s in base}
            dd = [s["late2"]-base[s["rep"]]["late2"] for s in group]
            m, h = estimate(dd)
            paired.append(dict(family=family, L=L, q=q, initial=init, boundary=bc,
                               n_pairs=len(dd), difference_c06_minus_c0=m,
                               CI95half=h, lower=m-h, upper=m+h))
    write_csv(OUT/"summary.csv", summary)
    write_csv(OUT/"paired_differences.csv", paired)
    write_csv(OUT/"markov_diagnostics.csv", diagnostics)
    series = defaultdict(list)
    for row in samples:
        s = by_run[row["run"]]
        series[(key(s), row["clock"], row["time"])].append(row["entropy"])
    means = []
    for (k, clock, tt), vv in sorted(series.items()):
        if len(vv) < 2:
            continue
        m, h = estimate(vv)
        means.append(dict(family=k[0], L=k[1], q=k[2], c=k[3], initial=k[4],
                          boundary=k[5], clock=clock, time=tt, n=len(vv), mean=m,
                          CI95half=h))
    write_csv(OUT/"entropy_curves.csv", means)

    plt.rcParams.update({"font.size": 10, "savefig.dpi": 180})
    def curve(ax, family, L, q, c, init="zero", clock="physical", label=None):
        rows = [s for s in means if (s["family"], s["L"], s["q"], s["c"],
                                    s["initial"], s["clock"])
                == (family, L, q, c, init, clock) and s["n"] == 16]
        x = np.array([s["time"] for s in rows])
        y = np.array([s["mean"] for s in rows])
        h = np.array([s["CI95half"] for s in rows])
        line, = ax.plot(x, y, label=label or f"q={q:g}, c={c:g}")
        ax.fill_between(x, y-h, y+h, color=line.get_color(), alpha=.12)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for q in (0, .1, .13, .33):
        curve(axes[0], "symmetric", 16, q, 0, label=f"q={q:g}")
    axes[0].set(xlabel="Measurements / L", ylabel="Half-chain entropy (bits)",
                title="IID baseline, L=16 (16 trajectories)", xlim=(0, 50))
    axes[0].legend()
    for q in (0, .1, .13, .33):
        rows = [s for s in summary if s["family"] == "symmetric" and s["q"] == q
                and s["c"] == 0 and s["initial"] == "zero"]
        axes[1].errorbar([s["L"] for s in rows], [s["late2"] for s in rows],
                         yerr=[s["late2_95half"] for s in rows], marker="o",
                         capsize=3, label=f"q={q:g}")
    axes[1].set(xlabel="L", ylabel="Entropy, sweeps 193-256 (bits)",
                title="Small-size benchmark; no phase fit", xticks=[8, 12, 16])
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(FIG/"baseline.png")
    fig.savefig(FIG/"baseline.pdf")
    plt.close(fig)

    if "pilot" in stages:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for q in (.1, .13, .33):
            rows = [s for s in paired if s["q"] == q and s["initial"] == "zero"]
            axes[0].errorbar([s["L"] for s in rows],
                             [s["difference_c06_minus_c0"] for s in rows],
                             yerr=[s["CI95half"] for s in rows], marker="o",
                             capsize=3, label=f"q={q:g}")
        axes[0].axhline(0, color="black", lw=.7)
        axes[0].set(xlabel="L", ylabel="Paired entropy difference (bits)",
                    title="c=0.6 minus c=0; late window", xticks=[8, 12, 16])
        axes[0].legend()
        for c in (0, .6):
            curve(axes[1], "symmetric", 16, .13, c, label=f"c={c:g}")
            curve(axes[2], "symmetric", 16, .13, c, clock="active", label=f"c={c:g}")
        axes[1].set(xlabel="Measurements / L", ylabel="Entropy (bits)",
                    title="q=0.13, L=16; physical clock", xlim=(0, 40))
        axes[2].set(xlabel="State-changing measurements / L", ylabel="Entropy (bits)",
                    title="Same trajectories; active clock", xlim=(0, 40))
        axes[1].legend()
        axes[2].legend()
        fig.tight_layout()
        fig.savefig(FIG/"memory_and_clocks.png")
        fig.savefig(FIG/"memory_and_clocks.pdf")
        plt.close(fig)

        # Exact matched active-event diagnostic at 64L state changes.
        matched = []
        rows64 = [s for s in samples if s["clock"] == "active" and s["time"] == 64]
        mapping = {s["run"]:s["entropy"] for s in rows64}
        for k, group in sorted(grouped.items()):
            family, L, q, c, init, bc = k
            if c != .6 or q == 0:
                continue
            base = {s["rep"]:s for s in grouped[(family,L,q,0,init,bc)]}
            dd = [mapping[s["run"]]-mapping[base[s["rep"]]["run"]]
                  for s in group if s["run"] in mapping
                  and base[s["rep"]]["run"] in mapping]
            if len(dd) > 1:
                m, h = estimate(dd)
                matched.append(dict(L=L,q=q,initial=init,n=len(dd),
                                    active_sweeps=64,difference=m,CI95half=h))
        write_csv(OUT/"matched_active64.csv", matched)

    control = list(csv.DictReader((OUT/"global_hold_control.csv").open()))
    reset = [r for r in control if r["repeated"] == "0"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.step([int(r["reset"]) for r in reset],
            [int(r["S_base"]) for r in reset], where="post", label="IID: raw events")
    ax.step([int(r["event"]) for r in control],
            [int(r["S_held"]) for r in control], where="post", alpha=.8,
            label="Global hold c=0.6: raw events")
    ax.set(xlabel="Measurement events", ylabel="Entropy (bits)",
           title="Exact negative control: identical states at resets")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG/"global_hold.png")
    plt.close(fig)
    dump(OUT/"analysis_notes.json",
         dict(intervals="Pointwise 95% Student-t, independent trajectory-window means; "
                        "paired Student-t for c differences. No multiplicity correction.",
              active_clock="Conditional jump-chain sampling, NOT stationary physical-time average",
              windows="Each run: means of sweeps129-192 and193-256, correlated within run",
              trajectories=len(runs), events=sum(256*s["L"] for s in runs),
              phase_fit=False))
    print(f"Analyzed {len(runs)} trajectories; tables and plots saved.")

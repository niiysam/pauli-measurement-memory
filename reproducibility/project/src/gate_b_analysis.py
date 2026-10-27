"""Gate B paired trajectory statistics. No time-point pseudoreplication."""
from collections import defaultdict
import json
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.reproduce import ROOT, dump, write_csv
from src.gate_b import OUT

FIG=ROOT/"figures"/"gate_b"
WINDOWS=["p129_256_event_mean","p257_384_event_mean","p385_512_event_mean",
         "p385_512_sweep_end_mean","a65_128_mean","a129_256_mean",
         "p385_512_active_fraction","p385_512_segment_mean","p385_512_waiting_bias",
         "p385_512_plus_given_active","p385_512_minus_given_active",
         "p385_512_zero_given_active","adjacent_anticommute",
         "active_adjacent_anticommute","p385_512_ZZZ_given_active",
         "p385_512_active_given_ZZZ","p385_512_active_given_other",
         "visit_anti_lag1","visit_repeat_lag1","global_repeat",
         "Zcorr_lag1","Zcorr_lag2","Zcorr_lag4","Zcorr_lag8"]


def estimate(values, family_size=1):
    values=np.array(values,float)
    n=len(values)
    mean=float(values.mean())
    se=float(values.std(ddof=1)/np.sqrt(n))
    half=float(t.ppf(1-.05/(2*family_size),n-1)*se)
    return dict(n=n,mean=mean,half95=half,lower=mean-half,upper=mean+half)


def analyze():
    rows=json.loads((OUT/"trajectories.json").read_text())
    FIG.mkdir(parents=True,exist_ok=True)
    grouped=defaultdict(list)
    indexed={}
    for r in rows:
        grouped[r["L"],r["q"],r["c"],r["arm"]].append(r)
        indexed[r["L"],r["q"],r["c"],r["rep"],r["arm"]]=r
    summaries=[]
    for (L,q,c,arm), rr in sorted(grouped.items()):
        for metric in WINDOWS:
            assert all(r[metric] is not None for r in rr), (L,q,c,arm,metric)
            summaries.append(dict(L=L,q=q,c=c,arm=arm,metric=metric,
                                  **estimate([r[metric] for r in rr])))
    write_csv(OUT/"summary.csv",summaries)
    comparisons=[]
    # Cohorts are prespecified; no selecting replicates based on entropy.
    for L in (12,16):
        for q in (.1,.13,.33):
            for cohort,reps in (("all48",range(48)),("old16",range(16)),
                                ("new32",range(16,48)),
                                ("new16a",range(16,32)),("new16b",range(32,48))):
                for metric in WINDOWS:
                    contrasts=[
                        ("c06_minus_c0_original",(.6,"original"),(0,"original")),
                        ("c06_minus_c0_full",(.6,"full"),(0,"full")),
                        ("c06_minus_c0_block",(.6,"block"),(0,"block")),
                        ("full_minus_original_c0",(0,"full"),(0,"original")),
                        ("full_minus_original_c06",(.6,"full"),(.6,"original")),
                        ("block_minus_original_c0",(0,"block"),(0,"original")),
                        ("block_minus_original_c06",(.6,"block"),(.6,"original"))]
                    for contrast,(ca,aa),(cb,ab) in contrasts:
                        dd=[indexed[L,q,ca,rep,aa][metric]
                            - indexed[L,q,cb,rep,ab][metric] for rep in reps]
                        est=estimate(dd)
                        adjusted=estimate(dd,8)
                        comparisons.append(dict(L=L,q=q,cohort=cohort,metric=metric,
                                                contrast=contrast,**est,
                                                half95_family8=adjusted["half95"]))
                    for arm in ("full","block"):
                        dd=[(indexed[L,q,.6,rep,arm][metric]-indexed[L,q,.6,rep,"original"][metric])
                            -(indexed[L,q,0,rep,arm][metric]-indexed[L,q,0,rep,"original"][metric])
                            for rep in reps]
                        comparisons.append(dict(L=L,q=q,cohort=cohort,metric=metric,
                                                contrast=f"shuffle_effect_difference_{arm}",
                                                **estimate(dd),
                                                half95_family8=estimate(dd,8)["half95"]))
    write_csv(OUT/"paired_comparisons.csv",comparisons)
    # Paired temporal drifts, with trajectory as unit.
    drifts=[]
    for (L,q,c,arm),rr in sorted(grouped.items()):
        for cohort,reps in (("all48",range(48)),("new32",range(16,48))):
            for early,late in (("p129_256_event_mean","p257_384_event_mean"),
                               ("p257_384_event_mean","p385_512_event_mean")):
                values=[r[late]-r[early] for r in rr if r["rep"] in reps]
                drifts.append(dict(L=L,q=q,c=c,arm=arm,cohort=cohort,
                                   early=early,late=late,**estimate(values)))
    write_csv(OUT/"window_drifts.csv",drifts)

    # Curves are sampled at each sweep of each clock, but their bands use
    # independent parents; within-trajectory points are never the sample size.
    curves=defaultdict(list)
    for path in sorted((OUT/"raw").glob("*.npz")):
        with np.load(path) as data:
            L=int(path.stem.split("_")[0][1:])
            q=float(path.stem.split("_")[1][1:])
            c=float(path.stem.split("_")[2][1:])
            for arm in ("original","full","block"):
                S=data[arm+"_entropy"]
                changed=data[arm+"_changed"]
                curves[L,q,c,arm,"physical"].append(S[::L])
                active=np.flatnonzero(changed)+1
                jump=np.r_[S[0],S[active[L-1::L]]]
                curves[L,q,c,arm,"active"].append(jump)
    curve_stats={}
    curve_rows=[]
    for k,aa in sorted(curves.items()):
        # Never average a shrinking subset of trajectories at late jump times.
        length=min(len(a) for a in aa)
        array=np.array([a[:length] for a in aa],float)
        mean=array.mean(axis=0)
        half=t.ppf(.975,len(aa)-1)*array.std(axis=0,ddof=1)/np.sqrt(len(aa))
        curve_stats[k]=(mean,half)
        for n,(m,h) in enumerate(zip(mean,half)):
            curve_rows.append(dict(L=k[0],q=k[1],c=k[2],arm=k[3],clock=k[4],
                                   sweeps=n,n=len(aa),mean=m,half95=h))
    write_csv(OUT/"curves.csv",curve_rows)
    plt.rcParams.update({"font.size":9,"savefig.dpi":170})
    colors={"original":"tab:blue","full":"tab:orange","block":"tab:green"}
    fig,axes=plt.subplots(2,2,figsize=(10,6.5))
    for row,L in enumerate((12,16)):
        for col,clock in enumerate(("physical","active")):
            ax=axes[row,col]
            for c in (0,.6):
                for arm in ("original","full","block"):
                    m,h=curve_stats[L,.13,c,arm,clock]
                    ax.plot(np.arange(len(m)),m,color=colors[arm],
                            ls="--" if c==0 else "-",label=f"c={c:g}, {arm}")
            ax.set(xlim=(0,256),xlabel=f"{clock.capitalize()} sweeps",
                   ylabel="Half-chain entropy (bits)",title=f"q=0.13, L={L}")
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.tight_layout()
    fig.savefig(FIG/"history_curves.png")
    fig.savefig(FIG/"history_curves.pdf")
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    for ax,metric,title in zip(axes,("p385_512_event_mean","a129_256_mean"),
                               ("Physical sweeps 385-512","Active sweeps 129-256")):
        for shift,arm in enumerate(("full","block")):
            selected=[r for r in comparisons if r["cohort"]=="new32" and r["q"]==.13
                      and r["metric"]==metric and r["contrast"]==f"{arm}_minus_original_c06"]
            ax.errorbar([r["L"]+(.12 if shift else -.12) for r in selected],
                         [r["mean"] for r in selected],
                         yerr=[r["half95_family8"] for r in selected],
                         marker="o",capsize=4,label=arm)
        ax.axhline(0,color="black",lw=.7)
        ax.set(xticks=[12,16],xlabel="L",ylabel="Shuffled minus original (bits)",
               title=title)
        ax.legend()
    fig.suptitle("Independent new32 cohort; simultaneous t intervals (8 contrasts)")
    fig.tight_layout()
    fig.savefig(FIG/"confirmation_effects.png")
    fig.savefig(FIG/"confirmation_effects.pdf")
    plt.close(fig)
    dump(OUT/"analysis_notes.json",dict(
        unit="48 parent seeds per L,q; c and arms paired, NOT 1728 independent samples",
        confirmation="new32 rep16..47 did not inform Gate A+ selection",
        primary_family="q=.13,c=.6: two L x full/block x physical/active = 8 contrasts, "
                       "Bonferroni t95. New32 family used for confirmation.",
        pointwise="Other comparisons exploratory; t intervals approximate",
        active_missing=sum(r["a129_256_mean"] is None for r in rows),
        max_waiting_identity_error=max(r["p385_512_identity_error"] for r in rows)))
    print("Gate B paired analysis and plots complete.",flush=True)

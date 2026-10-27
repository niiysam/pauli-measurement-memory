"""Convergence screen precedes descriptive size comparisons; never critical fits."""
import json
from collections import defaultdict
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.gate_c import OUT,CONFIG
from src.reproduce import ROOT,dump,write_csv

FIG=ROOT/"figures"/"gate_c"


def estimate(values):
    a=np.array(values,float);n=len(a)
    m=float(a.mean());half=float(t.ppf(.975,n-1)*a.std(ddof=1)/np.sqrt(n))
    return dict(n=n,mean=m,half95=half,lower=m-half,upper=m+half)


def analyze():
    rows=json.loads((OUT/"original_trajectories.json").read_text())
    original=list(rows)
    idx={(r["L"],r["q"],r["c"],r["rep"],r["initial"]):r for r in original}
    grouped=defaultdict(list)
    for r in original:grouped[r["L"],r["q"],r["c"],r["initial"]].append(r)
    screen=[]
    tolerance=CONFIG["convergence_tolerance_bits"]
    for L in CONFIG["sizes"]:
        for q in CONFIG["q"]:
            for c in CONFIG["c"]:
                rr=grouped[L,q,c,"zero"]
                checks={}
                estimates={}
                for obs in ("S","I3"):
                    for init in ("zero","plus"):
                        values=[idx[L,q,c,r["rep"],init][f"p1537_2048_{obs}"]
                                -idx[L,q,c,r["rep"],init][f"p1025_1536_{obs}"] for r in rr]
                        label=f"late_drift_{init}_{obs}"
                        est=estimate(values);estimates[label]=est
                        checks[label]=est["lower"]>=-tolerance and est["upper"]<=tolerance
                    values=[idx[L,q,c,r["rep"],"plus"][f"p1537_2048_{obs}"]
                            -r[f"p1537_2048_{obs}"] for r in rr]
                    label=f"initial_difference_{obs}"
                    est=estimate(values);estimates[label]=est
                    checks[label]=est["lower"]>=-tolerance and est["upper"]<=tolerance
                coupled=[r["coalesced_sweeps"] for r in rr]
                screen.append(dict(L=L,q=q,c=c,n=len(rr),screen_pass=all(checks.values()),
                    checks=checks,estimates=estimates,
                    coupled_before_1024=sum(v is not None and v<=1024 for v in coupled),
                    not_coupled=sum(v is None for v in coupled),
                    max_observed_coupling_sweeps=max((v for v in coupled if v is not None),default=None)))
    dump(OUT/"CONVERGENCE_SCREEN.json",dict(
        tolerance_bits=tolerance,status="practical pointwise screen, not stationarity theorem",
        passed=sum(r["screen_pass"] for r in screen),total=len(screen),cells=screen,
        no_fit=True))
    if (OUT/"control_trajectories.json").exists():
        rows+=json.loads((OUT/"control_trajectories.json").read_text())
    allgroups=defaultdict(list)
    for r in rows:allgroups[r["L"],r["q"],r["c"],r["arm"],r["initial"]].append(r)
    summaries=[]
    metrics=["late_active_fraction"]
    for obs in ("S","I3"):
        metrics += [f"p{lo}_{hi}_{obs}" for lo,hi in CONFIG["physical_windows"]]
        metrics += [f"a513_1024_{obs}",f"p1537_2048_{obs}_stride8",f"p1537_2048_{obs}_stride16"]
    for (L,q,c,arm,init),rr in sorted(allgroups.items()):
        for metric in metrics:
            vv=[r[metric] for r in rr]
            if any(v is None for v in vv):
                summaries.append(dict(L=L,q=q,c=c,arm=arm,initial=init,metric=metric,
                                      n=len(vv),mean="",half95="",lower="",upper=""))
                continue
            summaries.append(dict(L=L,q=q,c=c,arm=arm,initial=init,metric=metric,**estimate(vv)))
    write_csv(OUT/"summary.csv",summaries)
    comparisons=[]
    allidx={(r["L"],r["q"],r["c"],r["rep"],r["arm"],r["initial"]):r for r in rows}
    for L in CONFIG["sizes"]:
        for q in CONFIG["q"]:
            reps=sorted(r["rep"] for r in grouped[L,q,0,"zero"])
            for metric in metrics:
                vv=[idx[L,q,.6,rep,"zero"][metric]-idx[L,q,0,rep,"zero"][metric]
                    for rep in reps if idx[L,q,.6,rep,"zero"][metric] is not None
                    and idx[L,q,0,rep,"zero"][metric] is not None]
                comparisons.append(dict(L=L,q=q,contrast="c06_minus_c0_original",
                                        metric=metric,**estimate(vv)))
            if q==.13 and (L,q,.6,"full","zero") in allgroups:
                for c in (0,.6):
                    for arm in ("full","block"):
                        for metric in metrics:
                            vv=[allidx[L,q,c,rep,arm,"zero"][metric]
                                -idx[L,q,c,rep,"zero"][metric] for rep in reps]
                            comparisons.append(dict(L=L,q=q,contrast=f"{arm}_minus_original_c{c:g}",
                                                    metric=metric,**estimate(vv)))
    write_csv(OUT/"paired_comparisons.csv",comparisons)
    # Across L seeds are independent. Same rep identifiers do not imply pairing.
    increments=[]
    for q in CONFIG["q"]:
        for c in CONFIG["c"]:
            for obs in ("S","I3"):
                for a,b in zip(CONFIG["sizes"][:-1],CONFIG["sizes"][1:]):
                    va=np.array([r[f"p1537_2048_{obs}"] for r in grouped[a,q,c,"zero"]])
                    vb=np.array([r[f"p1537_2048_{obs}"] for r in grouped[b,q,c,"zero"]])
                    ea=va.var(ddof=1)/len(va);eb=vb.var(ddof=1)/len(vb)
                    se=np.sqrt(ea+eb)
                    df=(ea+eb)**2/(ea**2/(len(va)-1)+eb**2/(len(vb)-1)) if se else 100
                    half=float(t.ppf(.975,df)*se)
                    eligible=all(next(s["screen_pass"] for s in screen if s["L"]==L
                                      and s["q"]==q and s["c"]==c) for L in (a,b))
                    increments.append(dict(q=q,c=c,observable=obs,L1=a,L2=b,
                                           difference=float(vb.mean()-va.mean()),half95=half,
                                           convergence_screen_pass=eligible))
    write_csv(OUT/"size_increments.csv",increments)
    FIG.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for col,q in enumerate(CONFIG["q"]):
        for row,obs in enumerate(("S","I3")):
            ax=axes[row,col]
            for c in (0,.6):
                selected=[r for r in summaries if r["q"]==q and r["c"]==c
                          and r["arm"]=="original" and r["initial"]=="zero"
                          and r["metric"]==f"p1537_2048_{obs}"]
                ax.errorbar([r["L"] for r in selected],[r["mean"] for r in selected],
                            yerr=[r["half95"] for r in selected],marker="o",capsize=3,
                            label=f"c={c:g}")
            ax.set(xlabel="L",ylabel=f"{obs} (bits)",title=f"q={q:g}; finite-time window",
                   xticks=CONFIG["sizes"])
            ax.legend()
    fig.tight_layout();fig.savefig(FIG/"size_dependence.png");fig.savefig(FIG/"size_dependence.pdf")
    plt.close(fig)
    # Curves show original paths and initialization convergence; bands are
    # across independent parent means, never independent time points.
    curves=defaultdict(list)
    for path in sorted((OUT/"raw"/"original").glob("*.npz")):
        parts=path.stem.split("_");L=int(parts[0][1:]);q=float(parts[1][1:]);c=float(parts[2][1:])
        if q!=.13:continue
        with np.load(path) as data:
            for init in ("zero","plus"):
                curves[L,c,init].append(data[init+"_physical"][:,2:])
    fig,axes=plt.subplots(2,2,figsize=(11,6))
    for row,L in enumerate((20,24)):
        for col,obs in enumerate(("S","I3")):
            for c in (0,.6):
                for init in ("zero","plus"):
                    values=np.array(curves[L,c,init])[:,:,col]
                    mean=values.mean(axis=0)
                    axes[row,col].plot(np.arange(len(mean))*4,mean,
                                      ls="-" if init=="zero" else "--",
                                      label=f"c={c:g}, {init}")
            axes[row,col].set(xlabel="Physical sweeps",ylabel=f"{obs} (bits)",
                              title=f"L={L}, q=.13")
    axes[0,0].legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(FIG/"long_time.png");fig.savefig(FIG/"long_time.pdf")
    plt.close(fig)
    print(f"Practical convergence screen {sum(s['screen_pass'] for s in screen)}/24. "
          "No phase-boundary or exponent fits performed.",flush=True)

"""Finite-size Gate D summaries and exploratory I3 roots; no exponent fitting."""
import argparse
from itertools import combinations
import json
import numpy as np
from scipy.stats import t
from src.gate_d import OUT,BASE,ci,k_global,k_visit
from src.gate_c2 import classify
from src.ensemble import analytic_index
from src.reproduce import ROOT,dump,write_csv


def difference(a,b,alpha=.05):
    a,b=np.asarray(a),np.asarray(b); va=a.var(ddof=1)/len(a); vb=b.var(ddof=1)/len(b)
    se=np.sqrt(va+vb); m=float(a.mean()-b.mean())
    if se==0: return dict(mean=m,half95=0.,lower=m,upper=m)
    df=(va+vb)**2/(va*va/(len(a)-1)+vb*vb/(len(b)-1))
    half=float(t.isf(alpha/2,df)*se)
    return dict(mean=m,half95=half,lower=m-half,upper=m+half)


def load(stage):
    rows=[]; kr=[]
    for p in sorted((OUT/stage).glob('rows_*.json')):rows+=json.loads(p.read_text())
    for p in sorted((OUT/stage).glob('K_*.json')):kr+=json.loads(p.read_text())
    return rows,kr


def analyze_stage(stage):
    folder=OUT/stage; config=json.loads((folder/'config.json').read_text())
    rows,kr=load(stage); idx={(r['L'],r['q'],r['c'],r['rep'],r['initial']):r for r in rows}
    def vals(L,q,c,metric,init='zero'):
        return np.array([idx[L,q,c,r,init][metric] for r in range(config['n'])])
    summary=[]; checks=[]; screens=[]; roots=[]; ksummary=[]; drifts=[]
    for q in config['q']:
        for c in BASE['c']:
            for L in BASE['sizes']:
                cell=dict(L=L,q=q,c=c,n=config['n'],primary_pass=True,all_fixed_pass=True,
                          max_coalesced_sweeps=max(vals(L,q,c,'coalesced_sweeps')),
                          not_coalesced=int(np.sum(vals(L,q,c,'coalesced_sweeps')<0)))
                for obs in ('S','I3'):
                    for init in ('zero','plus'):
                        for w in range(6):
                            summary.append(dict(stage=stage,L=L,q=q,c=c,initial=init,metric=f'w{w}_{obs}',**ci(vals(L,q,c,f'w{w}_{obs}',init))))
                        for name,wlate,wearly in (('primary',3,2),('previous',2,1),('last_half',5,4)):
                            est=ci(vals(L,q,c,f'w{wlate}_{obs}',init)-vals(L,q,c,f'w{wearly}_{obs}',init))
                            decision=classify(est)
                            checks.append(dict(L=L,q=q,c=c,initial=init,observable=obs,test=name,decision=decision,**est))
                            cell['all_fixed_pass'] &= decision=='PASS'
                            if name=='primary':cell['primary_pass'] &= decision=='PASS'
                    est=ci(vals(L,q,c,'w3_'+obs,'plus')-vals(L,q,c,'w3_'+obs,'zero'))
                    decision=classify(est); cell['primary_pass'] &= decision=='PASS'; cell['all_fixed_pass'] &= decision=='PASS'
                    checks.append(dict(L=L,q=q,c=c,initial='plus-minus-zero',observable=obs,test='initial',decision=decision,**est))
                for metric in ('active_fraction','active_513_1024_S'):
                    v=vals(L,q,c,metric)
                    if all(x is not None for x in v):summary.append(dict(stage=stage,L=L,q=q,c=c,initial='zero',metric=metric,**ci(v)))
                screens.append(cell)
            # Analyze each protocol's roots only once, after all its q cells.
            if q!=config['q'][-1]: continue
            for small,large in combinations(BASE['sizes'],2):
                for qlo,qhi in zip(config['q'][:-1],config['q'][1:]):
                    a0,b0=vals(large,qlo,c,'w3_I3'),vals(small,qlo,c,'w3_I3')
                    a1,b1=vals(large,qhi,c,'w3_I3'),vals(small,qhi,c,'w3_I3')
                    d0=float(a0.mean()-b0.mean()); d1=float(a1.mean()-b1.mean())
                    if d0*d1>=0: continue
                    point=qlo-d0*(qhi-qlo)/(d1-d0)
                    rng=np.random.default_rng([2026102423,small,large,int(qlo*100000),int(c*10),len(stage)])
                    samples=[]; draws=2000
                    for _ in range(draws):
                        x0=rng.choice(a0,len(a0)).mean()-rng.choice(b0,len(b0)).mean()
                        x1=rng.choice(a1,len(a1)).mean()-rng.choice(b1,len(b1)).mean()
                        if x0*x1<0:samples.append(qlo-x0*(qhi-qlo)/(x1-x0))
                    e0=difference(a0,b0); e1=difference(a1,b1)
                    # Family-24 diagnostic is deliberately conservative. A point
                    # root or conditional bootstrap is never a certified crossing.
                    s0=difference(a0,b0,alpha=.05/24); s1=difference(a1,b1,alpha=.05/24)
                    sign_support=(s0['lower']>0 and s1['upper']<0) or (s0['upper']<0 and s1['lower']>0)
                    converge=all(next(s['all_fixed_pass'] for s in screens if s['L']==L and s['q']==q and s['c']==c)
                                 for L in (small,large) for q in (qlo,qhi))
                    roots.append(dict(stage=stage,c=c,L_small=small,L_large=large,qlo=qlo,qhi=qhi,
                        interpolated_root=point,Dlo=d0,Dhi=d1,Dlo_low=e0['lower'],Dlo_high=e0['upper'],
                        Dhi_low=e1['lower'],Dhi_high=e1['upper'],family24_opposite_sign=bool(sign_support),
                        all_endpoints_converged=converge,bootstrap_bracket_fraction=len(samples)/draws,
                        conditional_bootstrap_low=float(np.quantile(samples,.025)) if len(samples)>=.95*draws else None,
                        conditional_bootstrap_high=float(np.quantile(samples,.975)) if len(samples)>=.95*draws else None))
    for L in BASE['sizes']:
        for q in config['q']:
            for c in BASE['c']:
                subset=[r for r in kr if r['L']==L and r['q']==q and r['c']==c]
                keys=sorted(set((r['clock'],r['lag']) for r in subset))
                for clock,lag in keys:
                    rr=[r for r in subset if r['clock']==clock and r['lag']==lag]
                    ksummary.append(dict(stage=stage,L=L,q=q,c=c,clock=clock,lag=lag,
                        theory=rr[0]['theory'],**ci([r['value'] for r in rr])))
    write_csv(folder/'summary.csv',summary);write_csv(folder/'convergence_checks.csv',checks)
    write_csv(folder/'crossings.csv',roots);write_csv(folder/'K_summary.csv',ksummary)
    dump(folder/'convergence.json',dict(cells=screens,primary_pass=sum(s['primary_pass'] for s in screens),
                                      all_fixed_pass=sum(s['all_fixed_pass'] for s in screens)))
    dump(folder/'crossings.json',roots)
    dump(folder/'analysis_notes.json',dict(stage=stage,parents_per_cell=config['n'],
        window_length=config['T']//4,point_roots=len(roots),
        robust_roots=sum(r['all_endpoints_converged'] and r['family24_opposite_sign'] for r in roots),
        restriction='Conditional bootstrap interval excludes missing roots; it is not unconditional 95% coverage. Exploratory scan roots are not confirmatory.'))
    plot_stage(stage,summary)
    print(json.dumps(dict(stage=stage,point_roots=len(roots),primary_pass=sum(s['primary_pass'] for s in screens),
                    all_fixed_pass=sum(s['all_fixed_pass'] for s in screens),cells=len(screens)),indent=2))


def plot_stage(stage,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'figures'/'gate_d';folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    colors=['#2864a0','#e57c22','#3b9958','#a4469e']
    for col,c in enumerate(BASE['c']):
        for L,color in zip(BASE['sizes'],colors):
            for row,obs in enumerate(('S','I3')):
                rr=sorted([r for r in summary if r['c']==c and r['L']==L and r['initial']=='zero' and r['metric']=='w3_'+obs],key=lambda r:r['q'])
                axes[row,col].errorbar([r['q'] for r in rr],[r['mean'] for r in rr],yerr=[r['half95'] for r in rr],color=color,marker='o',label=f'L={L}',capsize=2,lw=1)
        axes[0,col].set_title(f'c={c:g}');axes[1,col].set_xlabel('q');axes[1,col].axhline(0,color='.7',lw=.7)
    axes[0,0].set_ylabel('S (bits)');axes[1,0].set_ylabel('I3 (bits)');axes[0,0].legend()
    fig.suptitle(f'{stage}: finite-time observations, trajectory t95 intervals; convergence checked separately')
    fig.tight_layout();fig.savefig(folder/f'{stage}_curves.png',dpi=150);fig.savefig(folder/f'{stage}_curves.pdf');plt.close(fig)


def analyze():
    for stage in ('coarse','refine','confirm'):
        if (OUT/stage/'manifest.json').exists():analyze_stage(stage)
    qs=sorted({q for stage in ('coarse','refine','confirm') if (OUT/stage/'config.json').exists() for q in json.loads((OUT/stage/'config.json').read_text())['q']})
    write_csv(OUT/'static_index.csv',[dict(L=L,q=q,I_iid=analytic_index(L,3,[q,q,1-2*q]),
        K_overlap_iid=L/5*analytic_index(L,3,[q,q,1-2*q]),c_independent=True) for L in BASE['sizes'] for q in qs])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',nargs='?',choices=['coarse','refine','confirm'])
    a=p.parse_args().stage
    analyze_stage(a) if a else analyze()

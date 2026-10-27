"""Trajectory-level, fixed-window Gate C2 analysis; no critical scaling fit."""
import json
import numpy as np
from scipy.stats import t
from src.gate_c2 import OUT,CONFIG,ci,classify
from src.reproduce import ROOT,dump,write_csv


def welch(a,b):
    a,b=np.asarray(a),np.asarray(b)
    va,vb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b)
    se=np.sqrt(va+vb)
    df=(va+vb)**2/(va*va/(len(a)-1)+vb*vb/(len(b)-1))
    mean=float(a.mean()-b.mean()); half=float(t.ppf(.975,df)*se)
    return dict(mean=mean,half95=half,lower=mean-half,upper=mean+half,df=float(df))


def analyze():
    rows=[]
    for L in CONFIG['sizes']:
        rows+=json.loads((OUT/f'trajectories_L{L}.json').read_text())
    dump(OUT/'trajectories.json',rows)
    idx={(r['L'],r['c'],r['rep'],r['initial']):r for r in rows}
    def vals(L,c,initial,key):
        return np.array([idx[L,c,r,initial][key] for r in CONFIG['parent_reps']])
    summary=[]; screens=[]; checks=[]; contrasts=[]; sizes=[]
    for L in CONFIG['sizes']:
        for c in CONFIG['c']:
            screen=dict(L=L,c=c,n=48,primary_pass=True,secondary_pass=True)
            for obs in ('S','I3'):
                for initial in CONFIG['initials']:
                    for lo,hi in CONFIG['windows']:
                        est=ci(vals(L,c,initial,f'w{lo}_{hi}_{obs}'))
                        summary.append(dict(L=L,c=c,initial=initial,window=f'{lo}_{hi}',observable=obs,**est))
                    for label,left,right,primary in (
                        ('primary_drift','w3073_4096','w2049_3072',True),
                        ('previous_drift','w2049_3072','w1025_2048',False),
                        ('last_half_drift','w3585_4096','w3073_3584',False)):
                        dd=vals(L,c,initial,left+'_'+obs)-vals(L,c,initial,right+'_'+obs)
                        est=ci(dd); decision=classify(est)
                        checks.append(dict(L=L,c=c,initial=initial,observable=obs,test=label,
                                           decision=decision,**est))
                        screen['primary_pass' if primary else 'secondary_pass'] &= decision=='PASS'
                dd=vals(L,c,'plus','w3073_4096_'+obs)-vals(L,c,'zero','w3073_4096_'+obs)
                est=ci(dd); decision=classify(est)
                checks.append(dict(L=L,c=c,initial='plus-minus-zero',observable=obs,
                                   test='initial_difference',decision=decision,**est))
                screen['primary_pass'] &= decision=='PASS'
            screen['all_fixed_checks_pass']=screen['primary_pass'] and screen['secondary_pass']
            screen['max_coalesced_sweeps']=max(idx[L,c,r,'zero']['coalesced_sweeps'] for r in CONFIG['parent_reps'])
            screen['not_coalesced']=sum(idx[L,c,r,'zero']['coalesced_sweeps']<0 for r in CONFIG['parent_reps'])
            screens.append(screen)
        for obs in ('S','I3'):
            for lo,hi in CONFIG['windows'][1:]:
                dd=vals(L,.6,'zero',f'w{lo}_{hi}_{obs}')-vals(L,0,'zero',f'w{lo}_{hi}_{obs}')
                est=ci(dd); adjusted=ci(dd,alpha=.05/8)
                rng=np.random.default_rng([2026102420,L,0 if obs=='S' else 1,lo])
                boots=np.mean(rng.choice(dd,size=(5000,len(dd)),replace=True),axis=1)
                contrasts.append(dict(L=L,observable=obs,window=f'{lo}_{hi}',contrast='c06-minus-c0',
                    **est,family8_lower=adjusted['lower'],family8_upper=adjusted['upper'],
                    bootstrap_lower=float(np.quantile(boots,.025)),bootstrap_upper=float(np.quantile(boots,.975)),
                    cohort72_95_mean=float(dd[:24].mean()),cohort96_119_mean=float(dd[24:].mean())))
    for c in CONFIG['c']:
        for small,large in zip(CONFIG['sizes'][:-1],CONFIG['sizes'][1:]):
            both=all(s['all_fixed_checks_pass'] for s in screens if s['c']==c and s['L'] in (small,large))
            for obs in ('S','I3'):
                est=welch(vals(large,c,'zero','w3073_4096_'+obs),vals(small,c,'zero','w3073_4096_'+obs))
                sizes.append(dict(c=c,L_small=small,L_large=large,observable=obs,
                                  both_converged=both,**est))
    write_csv(OUT/'summary.csv',summary); write_csv(OUT/'convergence_checks.csv',checks)
    write_csv(OUT/'paired_contrasts.csv',contrasts); write_csv(OUT/'size_increments.csv',sizes)
    dump(OUT/'CONVERGENCE_SCREEN.json',dict(tolerance=.15,confidence=.95,cells=screens,
         primary_pass_count=sum(s['primary_pass'] for s in screens),
         all_fixed_pass_count=sum(s['all_fixed_checks_pass'] for s in screens),
         note='Finite-window equivalence, not a proof of stationarity. Intersection criterion; no time points used as independent samples.'))
    sensitivity=[]
    for L in CONFIG['sizes']:
        for c in CONFIG['c']:
            for obs in ('S','I3'):
                for stride in ((4,) if obs=='S' else (8,16)):
                    key='w3073_4096_'+obs
                    dd=vals(L,c,'zero',key+f'_grid{stride}')-vals(L,c,'zero',key)
                    sensitivity.append(dict(L=L,c=c,observable=obs,stride=stride,
                        mean_difference=float(dd.mean()),max_individual_difference=float(np.max(np.abs(dd)))))
    write_csv(OUT/'sampling_sensitivity.csv',sensitivity)
    plot(idx,summary)
    print(json.dumps(dict(cells=screens,late_contrasts=[r for r in contrasts if r['window']=='3073_4096']),indent=2),flush=True)


def plot(idx,summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=ROOT/'figures'/'gate_c2'; folder.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(2,4,figsize=(14,6),sharex=True)
    for col,L in enumerate(CONFIG['sizes']):
        for c,color in ((0,'#2465ad'),(.6,'#d65d22')):
            ss=[]; ii=[]
            for rep in CONFIG['parent_reps']:
                with np.load(OUT/'raw'/f'L{L}_c{c:g}_R{rep}.npz') as data:
                    S=data['S'][0]; grid=data['grid']
                    ss.append([S[(a-1)*L+1:b*L+1].mean() for a,b in
                               [(1+256*j,256*(j+1)) for j in range(16)]])
                    ii.append([grid[(grid[:,0]>256*j*L)&(grid[:,0]<=256*(j+1)*L),5].mean() for j in range(16)])
            for ax,data in zip(axes[:,col],(ss,ii)):
                x=np.arange(128,4096,256); data=np.asarray(data)
                mean=data.mean(axis=0); half=t.ppf(.975,47)*data.std(axis=0,ddof=1)/np.sqrt(48)
                ax.errorbar(x,mean,yerr=half,color=color,label=f'c={c:g}',lw=1,capsize=2)
                ax.axvline(2048,color='.7',ls=':'); ax.axvline(3072,color='.7',ls=':')
            axes[0,col].set_title(f'L={L}'); axes[1,col].set_xlabel('Sweeps')
    axes[0,0].set_ylabel('S (bits)'); axes[1,0].set_ylabel('I3 (bits)'); axes[0,0].legend()
    fig.suptitle('q=0.13; 48 independent parents/cell; pointwise trajectory t95 intervals')
    fig.tight_layout(); fig.savefig(folder/'long_time.png',dpi=150); fig.savefig(folder/'long_time.pdf'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.8))
    for obs,ax in zip(('S','I3'),axes):
        for c,color in ((0,'#2465ad'),(.6,'#d65d22')):
            data=[next(r for r in summary if r['L']==L and r['c']==c and r['initial']=='zero' and r['window']=='3073_4096' and r['observable']==obs) for L in CONFIG['sizes']]
            ax.errorbar(CONFIG['sizes'],[r['mean'] for r in data],yerr=[r['half95'] for r in data],marker='o',color=color,label=f'c={c:g}')
        ax.set_xlabel('L'); ax.set_ylabel(obs+' (bits)'); ax.set_xticks(CONFIG['sizes']); ax.legend()
    fig.suptitle('Final-window finite-size observations; no asymptotic law fitted')
    fig.tight_layout(); fig.savefig(folder/'size_dependence.png',dpi=150); fig.savefig(folder/'size_dependence.pdf'); plt.close(fig)


if __name__=='__main__': analyze()

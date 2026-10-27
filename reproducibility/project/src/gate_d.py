"""Gate D bounded batches; old sampler law and signed stabilizer engine reused."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import io
import json
import time
import unittest
import numpy as np
from scipy.stats import t
from src.reproduce import ROOT,dump,write_csv,environment
from src.ensemble import LocalMemory,analytic_index,enumerated_index
from src.history_controls import pauli_table,WORDS
from src.critical_controls import GateCState,observables
from src.gate_c2 import simulate as simulate_both

OUT=ROOT/'results'/'gate_d'
BASE=dict(sizes=[12,16,20,24],c=[0,.6],r=3,boundary='periodic',
    master=2026102421,initials=['zero','plus'],tolerance_bits=.15,
    coarse_q=[.05,.10,.15,.20,.25,.30,.35,.40,.45],
    coarse_n=8,coarse_sweeps=1024,refine_n=8,refine_sweeps=2048,
    confirm_n=32,confirm_sweeps=4096,max_refine_q=12,max_confirm_q=4,
    controls_n=16,max_control_q=2,control_sweeps=4096,block_sweeps=128,
    global_lags=[1,2,4,8,16,32,64,128],visit_lags=[1,2,4,8,16],
    windows='four equal quarters; primary last minus previous; secondary previous minus second and last half split',
    no_exponent_fits=True,workers=4)
STAGE_ID={'coarse':0,'refine':1,'confirm':2}


def ci(x,alpha=.05):
    x=np.asarray(x,float); m=float(x.mean())
    h=float(t.isf(alpha/2,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(n=len(x),mean=m,half95=h,lower=m-h,upper=m+h)


def source_hashes():
    paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    paths+=list((ROOT/'scripts').glob('*gate_d*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def archives():
    counts={}
    for name in ('gate_a_plus','gate_b','gate_c','gate_c2'):
        lines=(ROOT/'results'/name/'FILE_HASHES.sha256').read_text().splitlines()
        for line in lines:
            expected,rel=line.split('  ',1)
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
        counts[name]=len(lines)
    return counts


def history(L,q,c,rep,sweeps,stage):
    if L not in BASE['sizes'] or c not in BASE['c'] or not .05<=q<=.45:
        raise ValueError('Frozen model/grid bounds')
    key=[BASE['master'],STAGE_ID[stage],L,int(round(q*100000)),rep]
    ar,fr,hr,br,ir=[np.random.default_rng(s) for s in np.random.SeedSequence(key).spawn(5)]
    N=L*sweeps
    anchors=ar.integers(L,size=N).astype(np.uint8)
    fresh,hold,born=fr.random((N,3)),hr.random((N,3)),br.random(N)
    memory=LocalMemory(L,[q,q,1-2*q],c,ir.random((L,3)))
    codes=np.empty(N,np.uint8)
    for n,j in enumerate(anchors):
        a=memory.update(int(j),fresh[n],hold[n]); codes[n]=9*int(a[0])+3*int(a[1])+int(a[2])
    return anchors,codes,born,key


def k_visit(q,c,k):
    D=1-float(np.dot([q,q,1-2*q],[q,q,1-2*q]))
    return (1-(1-2*D*(1-c**k))**3)/2


def k_global(L,q,c,tau):
    if tau<1: raise ValueError('Positive global-event lag required')
    D=1-float(np.dot([q,q,1-2*q],[q,q,1-2*q])); a=1-2*D; b=2*D
    moment=a**3
    for j,coef in ((1,3*a*a*b),(2,3*a*b*b),(3,b**3)):
        moment+=coef*c**j*(1-(1-c**j)/L)**(tau-1)
    A1=(1-a)/2; A2=(1-a*a)/2
    return ((1-moment)/2+2*A1+2*A2)/5


def anti_table(L):
    p=[x for row in pauli_table(L) for x in row]
    return np.array([[a.anti(b) for b in p] for a in p],bool)


def temporal(anchors,codes,L,q,c,table):
    ids=anchors.astype(int)*27+codes; result=[]
    for tau in sorted(set(BASE['global_lags']+[L,2*L,4*L])):
        shift=(anchors[tau:].astype(int)-anchors[:-tau].astype(int))%L
        overlap=(shift<=2)|(shift>=L-2)
        hits=table[ids[:-tau],ids[tau:]]
        den=int(overlap.sum()); num=int(hits[overlap].sum())
        result.append(dict(clock='global_event',lag=tau,numerator=num,denominator=den,
                           value=num/den,theory=k_global(L,q,c,tau)))
    local=table[:27,:27]
    seq=[codes[anchors==j] for j in range(L)]
    for k in BASE['visit_lags']:
        den=sum(max(0,len(x)-k) for x in seq)
        num=sum(int(local[x[:-k],x[k:]].sum()) for x in seq if len(x)>k)
        result.append(dict(clock='same_anchor_visit',lag=k,numerator=num,denominator=den,
                           value=num/den,theory=k_visit(q,c,k)))
    return result


def window_records(data,L,q,c,rep,T,stage):
    grid=data['grid']; result=[]; quarter=T//4
    windows=[(i*quarter+1,(i+1)*quarter) for i in range(4)]
    windows += [(3*quarter+1,7*T//8),(7*T//8+1,T)]
    for k,init in enumerate(('zero','plus')):
        r=dict(stage=stage,L=L,q=q,c=c,rep=rep,initial=init,T=T,
            coalesced_sweeps=float(data['coalesced_event'])/L,
            active_fraction=float(data['changed'][k,3*T*L//4:].mean()))
        for w,(lo,hi) in enumerate(windows):
            sample=grid[(grid[:,0]>(lo-1)*L)&(grid[:,0]<=hi*L)]
            r[f'w{w}_S']=float(data['S'][k,(lo-1)*L+1:hi*L+1].mean())
            r[f'w{w}_I3']=float(sample[:,5+k].mean())
        active=np.flatnonzero(data['changed'][k])
        r['active_513_1024_S']=float(data['S'][k,active[512*L:1024*L]+1].mean()) if len(active)>=1024*L else None
        result.append(r)
    return result


def simulate_zero(anchors,codes,born,L):
    state=GateCState(L); table=pauli_table(L); N=len(anchors)
    outcome=np.empty(N,np.int8); changed=np.empty(N,bool); S=np.zeros(N+1,np.int8)
    grid=[[0,0,0,0]]; count=0
    for n,(j,code) in enumerate(zip(anchors,codes),1):
        out,p,ch=state.measure(table[int(j)][int(code)],float(born[n-1]))
        outcome[n-1]=out; changed[n-1]=ch; count+=int(ch)
        S[n]=state.half_entropy() if ch else S[n-1]
        if n%(4*L)==0: grid.append([n,count,*observables(state)])
    state.check()
    return dict(outcome=outcome,changed=changed,S=S,grid=np.array(grid,np.int32),
                final=np.array([[g.x,g.z,g.phase] for g in state.g],np.int64))


def run_job(stage,L,q,config):
    start=time.monotonic(); folder=OUT/stage/'raw'; folder.mkdir(parents=True,exist_ok=True)
    rows=[]; kr=[]; table=anti_table(L)
    for rep in range(config['n']):
        for c in BASE['c']:
            path=folder/f'L{L}_q{q:.3f}_c{c:g}_R{rep}.npz'
            assert not path.exists(),'No trajectory replacement or silent resume'
            anchors,codes,born,key=history(L,q,c,rep,config['T'],stage)
            data=simulate_both(anchors,codes,born,L)
            np.savez_compressed(path,anchors=anchors,codes=codes,born=born,seed_key=np.array(key),**data)
            rows+=window_records(data,L,q,c,rep,config['T'],stage)
            kr += [dict(L=L,q=q,c=c,rep=rep,stage=stage,**r) for r in temporal(anchors,codes,L,q,c,table)]
    dump(OUT/stage/f'rows_L{L}_q{q:.3f}.json',rows)
    dump(OUT/stage/f'K_L{L}_q{q:.3f}.json',kr)
    result=dict(L=L,q=q,parents=config['n'],packages=2*config['n'],
                signed_events=4*config['n']*L*config['T'],elapsed=time.monotonic()-start)
    print(f'{stage} {result}',flush=True)
    return result


def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    checked=archives(); log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    (OUT/'tests.txt').write_text(log.getvalue()); assert result.wasSuccessful(),log.getvalue()
    dump(OUT/'base_config.json',BASE)
    dump(OUT/'audit.json',dict(status='PASS',tests=result.testsRun,archives=checked,source_sha256=source_hashes(),environment=environment()))
    print(f'{result.testsRun} tests PASS; archives {checked}',flush=True)


def run(stage):
    assert json.loads((OUT/'audit.json').read_text())['source_sha256']==source_hashes()
    folder=OUT/stage
    assert not (folder/'manifest.json').exists()
    config=json.loads((folder/'config.json').read_text())
    cap=dict(coarse=9,refine=12,confirm=4)[stage]
    assert len(config['q'])<=cap and config['n']==BASE[stage+'_n'] and config['T']==BASE[stage+'_sweeps']
    started=time.monotonic(); runs=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        jobs=[pool.submit(run_job,stage,L,q,config) for q in config['q'] for L in BASE['sizes']]
        for job in as_completed(jobs):
            runs.append(job.result()); dump(folder/'batch_progress.json',runs)
    dump(folder/'manifest.json',dict(status='COMPLETE',runs=runs,config=config,
        signed_events=sum(r['signed_events'] for r in runs),elapsed=time.monotonic()-started,
        environment=environment(),source_sha256=source_hashes()))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__); p.add_argument('stage',choices=['audit','coarse','refine','confirm','analyze'])
    a=p.parse_args().stage
    if a=='audit': audit()
    elif a=='analyze':
        from src.gate_d_analysis import analyze
        analyze()
    else: run(a)

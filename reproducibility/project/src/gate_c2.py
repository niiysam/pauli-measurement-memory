"""Gate C2 only: fixed q=.13, L<=24, new parents 72..119, 4096 sweeps."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import io
import json
import time
import unittest
import numpy as np
from scipy.stats import t
from src.reproduce import ROOT, dump, write_csv, environment
from src.critical_controls import GateCState, history, observables, same_unsigned
from src.history_controls import pauli_table

OUT=ROOT/'results'/'gate_c2'
CONFIG=dict(sizes=[12,16,20,24], q=.13, c=[0,.6],
    parent_reps=list(range(72,120)), model_seed=2026102407,
    sweeps=4096, initials=['zero','plus'], boundary='periodic', r=3,
    S_sampling='every post-measurement event, including deterministic events',
    I3_stride_sweeps=4,
    windows=[[1,1024],[1025,2048],[2049,3072],[3073,4096]],
    primary_drift='W3-W2 (3073..4096 minus 2049..3072)',
    secondary_drift='W2-W1 and last-half W3 minus first-half W3',
    tolerance_bits=.15, confidence=.95,
    screen='Both observables and both initial states: primary drift and W3 initial difference CI wholly within +/-0.15; secondary drift separately reported',
    pairing='c and product initial states share parent stream; L and rep are independent parents',
    no_optional_stopping=True, no_fits=True, workers=4)


def ci(values, alpha=.05):
    x=np.asarray(values,dtype=float)
    if len(x)<2: raise ValueError('At least two independent parents required')
    mean=float(x.mean())
    half=float(t.ppf(1-alpha/2,len(x)-1)*x.std(ddof=1)/np.sqrt(len(x)))
    return dict(n=len(x),mean=mean,half95=half,lower=mean-half,upper=mean+half)


def classify(interval, tolerance=.15):
    lo,hi=interval['lower'],interval['upper']
    if lo>=-tolerance and hi<=tolerance:
        return 'PASS'
    if lo>tolerance or hi<-tolerance:
        return 'DRIFT_BEYOND_TOLERANCE'
    if lo<=0<=hi:
        return 'INSUFFICIENT_PRECISION'
    return 'DETECTED_DRIFT_PRACTICAL_SIZE_UNRESOLVED'


def source_hashes():
    files=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    files+=list((ROOT/'scripts').glob('gate_c2*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def archive_check():
    counts={}
    for stage in ('gate_a_plus','gate_b','gate_c'):
        lines=(ROOT/'results'/stage/'FILE_HASHES.sha256').read_text().splitlines()
        for line in lines:
            expected,rel=line.split('  ',1)
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
        counts[stage]=len(lines)
    return counts


def simulate(anchors,codes,born,L):
    """Reuse signed Born updates; coalescence only shares entropy calculations."""
    states=[GateCState(L,s) for s in CONFIG['initials']]
    table=pauli_table(L); N=len(anchors)
    outcome=np.empty((2,N),np.int8); changed=np.empty((2,N),bool)
    S=np.zeros((2,N+1),np.int8)
    grid=[[0,0,0,0,0,0,0]]  # event, two active counts, two S, two I3
    counts=[0,0]; coalesced=-1
    for n,(j,code) in enumerate(zip(anchors,codes),1):
        M=table[int(j)][int(code)]
        for k,state in enumerate(states):
            out,p,ch=state.measure(M,float(born[n-1]))
            outcome[k,n-1],changed[k,n-1]=out,ch
            counts[k]+=int(ch)
        if coalesced<0:
            for k,state in enumerate(states):
                S[k,n]=state.half_entropy() if changed[k,n-1] else S[k,n-1]
        else:
            assert changed[0,n-1]==changed[1,n-1]
            value=states[0].half_entropy() if changed[0,n-1] else S[0,n-1]
            S[:,n]=value
        if n%(CONFIG['I3_stride_sweeps']*L)==0:
            if coalesced<0 and same_unsigned(*states): coalesced=n
            v=observables(states[0]); w=v if coalesced>=0 else observables(states[1])
            assert v[0]==S[0,n] and w[0]==S[1,n]
            grid.append([n,*counts,v[0],w[0],v[1],w[1]])
    for state in states: state.check()
    if coalesced>=0: assert same_unsigned(*states)
    assert np.all(np.abs(np.diff(S.astype(int),axis=1))<=1)
    assert np.all(np.diff(S.astype(int),axis=1)[~changed]==0)
    return dict(outcome=outcome,changed=changed,S=S,
        grid=np.asarray(grid,np.int32),coalesced_event=np.array(coalesced,np.int32),
        final=np.asarray([[[g.x,g.z,g.phase] for g in s.g] for s in states],np.int64))


def records(data,L,c,rep):
    result=[]; grid=data['grid']
    windows=CONFIG['windows']+[[3073,3584],[3585,4096]]
    for k,initial in enumerate(CONFIG['initials']):
        row=dict(L=L,q=.13,c=c,rep=rep,initial=initial,
            coalesced_sweeps=float(data['coalesced_event'])/L,
            late_active_fraction=float(data['changed'][k,3072*L:].mean()))
        for lo,hi in windows:
            row[f'w{lo}_{hi}_S']=float(data['S'][k,(lo-1)*L+1:hi*L+1].mean())
            sample=grid[(grid[:,0]>(lo-1)*L)&(grid[:,0]<=hi*L)]
            assert len(sample)==(hi-lo+1)//CONFIG['I3_stride_sweeps']
            row[f'w{lo}_{hi}_I3']=float(sample[:,5+k].mean())
            # Fixed, diagnostic thinning. Never increases statistical n.
            row[f'w{lo}_{hi}_S_grid4']=float(sample[:,3+k].mean())
            for stride in (8,16):
                thin=sample[sample[:,0]%(stride*L)==0]
                row[f'w{lo}_{hi}_I3_grid{stride}']=float(thin[:,5+k].mean())
        result.append(row)
    return result


def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'manifest.json').exists(),'Completed run is immutable'
    checked=archive_check()
    log=io.StringIO()
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    (OUT/'tests.txt').write_text(log.getvalue())
    assert result.wasSuccessful(),log.getvalue()
    dump(OUT/'config.json',CONFIG)
    dump(OUT/'audit.json',dict(status='PASS',tests=result.testsRun,
        archived_hash_entries=checked,source_sha256=source_hashes(),environment=environment()))
    print(f'Audit PASS: {result.testsRun} tests; archives {checked}',flush=True)


def run_size(L):
    start=time.monotonic(); rows=[]
    folder=OUT/'raw'; folder.mkdir(parents=True,exist_ok=True)
    for rep in CONFIG['parent_reps']:
        for c in CONFIG['c']:
            path=folder/f'L{L}_c{c:g}_R{rep}.npz'
            assert not path.exists(),'Do not silently reuse or replace a trajectory'
            anchors,codes,born,key=history(L,.13,c,rep,CONFIG['sweeps'])
            data=simulate(anchors,codes,born,L)
            np.savez_compressed(path,anchors=anchors,codes=codes,born=born,
                                seed_key=np.array(key),**data)
            rows+=records(data,L,c,rep)
        if (rep-71)%8==0:
            print(f'L={L}: {rep-71}/48 new parents complete',flush=True)
    dump(OUT/f'trajectories_L{L}.json',rows)
    return dict(L=L,parents=48,packages=96,histories=192,
                events=192*L*CONFIG['sweeps'],elapsed_seconds=time.monotonic()-start)


def run():
    audit_record=json.loads((OUT/'audit.json').read_text())
    assert audit_record['source_sha256']==source_hashes()
    assert json.loads((OUT/'config.json').read_text())==CONFIG
    start=time.monotonic(); runs=[]
    with ProcessPoolExecutor(max_workers=CONFIG['workers']) as pool:
        jobs={pool.submit(run_size,L):L for L in CONFIG['sizes']}
        for job in as_completed(jobs):
            runs.append(job.result()); print(f'COMPLETE {runs[-1]}',flush=True)
    dump(OUT/'manifest.json',dict(status='COMPLETE',runs=runs,
        elapsed_seconds=time.monotonic()-start,source_sha256=source_hashes(),
        environment=environment(),config=CONFIG))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['audit','run','analyze'])
    stage=parser.parse_args().stage
    if stage=='audit': audit()
    elif stage=='run': run()
    else:
        from src.gate_c2_analysis import analyze
        analyze()

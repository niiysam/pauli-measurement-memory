"""Explicitly bounded L=28/32 extension; unchanged stabilizer update methods."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse
import hashlib
import io
import json
import time
import unittest
import numpy as np
from src.reproduce import ROOT,dump,environment
from src.stabilizer import Pauli
from src.critical_controls import GateCState,observables,same_unsigned
from src.ensemble import LocalMemory
from src.history_controls import pauli_table
from src.gate_d import window_records
from src.gate_e_audit import OUT

CONFIG=dict(sizes=[28,32],q=[.12,.13,.14],c=[0,.6],n=24,T=4096,
    master=2026102424,initials=['zero','plus'],workers=4,r=3,boundary='periodic',
    tolerance_bits=.15,I3_stride_sweeps=4,S_sampling='every event',
    seed='[master,L,round(100000q),rep]; five streams; c and initial paired',
    endpoint_family_combined=90,bootstrap_replicates=4000,bootstrap_seed=2026102426,
    no_optional_stopping=True,hard_limit='L<=32; no further q or n additions')


class GateEState(GateCState):
    def __init__(self,L,initial='zero'):
        if not isinstance(L,int) or not 1<=L<=32:raise ValueError('Gate E explicit bound 1..32')
        self.L=L
        if initial=='zero':self.g=[Pauli(0,1<<j) for j in range(L)]
        elif initial=='plus':self.g=[Pauli(1<<j,0) for j in range(L)]
        else:raise ValueError('Only frozen product states')
        self.check()


def hashes():
    paths=list((ROOT/'src').glob('*.py'))+list((ROOT/'tests').glob('*.py'))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def history(L,q,c,rep):
    if L not in CONFIG['sizes'] or q not in CONFIG['q'] or c not in CONFIG['c']:raise ValueError('Frozen E grid')
    key=[CONFIG['master'],L,int(round(q*100000)),rep]
    ar,fr,hr,br,ir=[np.random.default_rng(s) for s in np.random.SeedSequence(key).spawn(5)]
    N=L*CONFIG['T'];anchors=ar.integers(L,size=N).astype(np.uint8)
    fresh,hold,born=fr.random((N,3)),hr.random((N,3)),br.random(N)
    memory=LocalMemory(L,[q,q,1-2*q],c,ir.random((L,3)))
    codes=np.empty(N,np.uint8)
    for n,j in enumerate(anchors):
        x=memory.update(int(j),fresh[n],hold[n]);codes[n]=9*int(x[0])+3*int(x[1])+int(x[2])
    return anchors,codes,born,key


def simulate(anchors,codes,born,L):
    states=[GateEState(L,s) for s in CONFIG['initials']];table=pauli_table(L);N=len(anchors)
    outcome=np.empty((2,N),np.int8);changed=np.empty((2,N),bool);S=np.zeros((2,N+1),np.int8)
    grid=[[0,0,0,0,0,0,0]];counts=[0,0];coalesced=-1
    for n,(j,code) in enumerate(zip(anchors,codes),1):
        for k,state in enumerate(states):
            o,p,ch=state.measure(table[int(j)][int(code)],float(born[n-1]))
            outcome[k,n-1]=o;changed[k,n-1]=ch;counts[k]+=int(ch)
        if coalesced<0:
            for k,state in enumerate(states):S[k,n]=state.half_entropy() if changed[k,n-1] else S[k,n-1]
        else:
            assert changed[0,n-1]==changed[1,n-1]
            s=states[0].half_entropy() if changed[0,n-1] else S[0,n-1];S[:,n]=s
        if n%(4*L)==0:
            if coalesced<0 and same_unsigned(*states):coalesced=n
            v=observables(states[0]);w=v if coalesced>=0 else observables(states[1])
            assert v[0]==S[0,n] and w[0]==S[1,n]
            grid.append([n,*counts,v[0],w[0],v[1],w[1]])
    for state in states:state.check()
    if coalesced>=0:assert same_unsigned(*states)
    assert np.all(np.abs(np.diff(S.astype(int),axis=1))<=1)
    assert np.all(np.diff(S.astype(int),axis=1)[~changed]==0)
    return dict(S=S,changed=changed,outcome=outcome,grid=np.array(grid,np.int32),
        coalesced_event=np.array(coalesced,np.int32),
        final=np.array([[[g.x,g.z,g.phase] for g in state.g] for state in states],np.int64))


def audit():
    assert (OUT/'existing_data_review.json').exists(),'Review old data first'
    log=io.StringIO();tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    (OUT/'tests_extended.txt').write_text(log.getvalue());assert tests.wasSuccessful(),log.getvalue()
    dump(OUT/'extension_config.json',CONFIG)
    dump(OUT/'extension_audit.json',dict(status='PASS',tests=tests.testsRun,source_hashes=hashes(),environment=environment()))
    print(f'Extended audit PASS: {tests.testsRun} tests',flush=True)


def job(L,q):
    folder=OUT/'raw';folder.mkdir(parents=True,exist_ok=True);rows=[];start=time.monotonic()
    for rep in range(CONFIG['n']):
        for c in CONFIG['c']:
            a,w,u,key=history(L,q,c,rep);data=simulate(a,w,u,L)
            path=folder/f'L{L}_q{q:.3f}_c{c:g}_R{rep}.npz';assert not path.exists()
            np.savez_compressed(path,anchors=a,codes=w,born=u,seed_key=np.array(key),**data)
            rows+=window_records(data,L,q,c,rep,CONFIG['T'],'gate_e')
        if (rep+1)%8==0:print(f'L={L} q={q:g}: {rep+1}/24 parents',flush=True)
    dump(OUT/f'rows_L{L}_q{q:.3f}.json',rows)
    r=dict(L=L,q=q,n=24,packages=48,signed_events=4*24*4096*L,elapsed=time.monotonic()-start)
    print(f'COMPLETE {r}',flush=True);return r


def run():
    assert json.loads((OUT/'extension_audit.json').read_text())['source_hashes']==hashes()
    assert json.loads((OUT/'extension_config.json').read_text())==CONFIG
    assert not (OUT/'extension_manifest.json').exists()
    start=time.monotonic();runs=[]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for r in as_completed([pool.submit(job,L,q) for q in CONFIG['q'] for L in CONFIG['sizes']]):
            runs.append(r.result());dump(OUT/'extension_progress.json',runs)
    dump(OUT/'extension_manifest.json',dict(status='COMPLETE',runs=runs,config=CONFIG,
        signed_events=sum(r['signed_events'] for r in runs),elapsed=time.monotonic()-start,
        source_hashes=hashes(),environment=environment()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','run'])
    audit() if p.parse_args().stage=='audit' else run()

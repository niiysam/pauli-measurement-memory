"""Frozen minimal falsification search for connected bandwidth-two capacity."""
from collections import deque
from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.sci_route_20261026.phase1_exact import reachable,form_update,components
from research.sci_route_20261026.exact_check import closure
from src.stabilizer import Pauli,rank

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase2_results.json'
CAP=120
START=time.monotonic()


def graph(n,edges):
    omega=[0]*n
    for i,j in edges:omega[i]|=1<<j;omega[j]|=1<<i
    return tuple(omega)


def bicoloring(omega):
    colors={}
    for initial in range(len(omega)):
        if initial in colors:continue
        colors[initial]=0;queue=[initial]
        while queue:
            i=queue.pop()
            for j in range(len(omega)):
                if not(omega[i]>>j&1):continue
                if j not in colors:colors[j]=1-colors[i];queue.append(j)
                elif colors[i]==colors[j]:return None
    return colors


def analyze(omega):
    if time.monotonic()-START>CAP:raise TimeoutError('Frozen runtime cap')
    n=len(omega);path,top=reachable(omega)
    a=rank(omega)//2;rstar=len(top).bit_length()-1
    rec=dict(n=n,omega=list(omega),a=a,rstar=rstar,delta=n-a-rstar,
        states=len(path),word=list(path[top]),connected=len(components(omega))==1)
    coloring=bicoloring(omega)
    if coloring is not None:
        word=[j for bit in (0,1) for j in range(n) if coloring[j]==bit]
        state=frozenset([0])
        for j in word:state=form_update(state,j,omega)
        assert len(state)==1<<(n-a)
        assert rstar==n-a
        rec['bipartite_exact_witness']=word
    return rec


def independent(omega):
    # Algebraic realization on n qubits, NOT an original local-chain protocol.
    ps=[Pauli(1<<j,sum((1<<i) for i in range(j) if omega[i]>>j&1))
        for j in range(len(omega))]
    n=len(omega)
    assert all(ps[i].anti(ps[j])==((omega[i]>>j)&1)
               for i in range(n) for j in range(n))
    ms=[p.x|(p.z<<n) for p in ps]
    assert rank(ms)==n
    path,top=closure(ms,n)
    return dict(rstar=len(top).bit_length()-1,states=len(path),
                canonical_paulis=[[p.x,p.z,p.phase] for p in ps],
                note='Independent physical-symplectic set enumeration; algebraic embedding only')


def main():
    if OUT.exists():raise FileExistsError(OUT)
    old=json.loads((HERE/'phase1_results.json').read_text())
    protected=list(old['protected_before'])+[
        str((HERE/f).relative_to(ROOT)) for f in (
            'phase1_frozen.json','phase1_exact.py','phase1_results.json',
            'phase1_cut_dependence_results.json')]
    before={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in protected}
    records=[];counter=None;counts={}
    for n in range(1,7):
        edges=[(i,j) for i in range(n) for j in range(i+1,min(n,i+3))]
        counts[n]=0
        for code in range(1<<len(edges)):
            omega=graph(n,[e for k,e in enumerate(edges) if code>>k&1])
            if len(components(omega))!=1:continue
            rec=analyze(omega);rec['edge_bit_code']=code
            records.append(rec);counts[n]+=1
            if rec['delta']>1:
                counter=rec;break
        if counter:break
    extra=[]
    if counter is None:
        for triangles in (3,4):
            edges=[]
            for t in range(triangles):edges.extend(combinations((2*t,2*t+1,2*t+2),2))
            rec=analyze(graph(2*triangles+1,edges))
            rec['family']='articulation_triangle_chain';extra.append(rec)
            if rec['delta']>1:counter=rec;break
    k5=analyze(graph(5,list(combinations(range(5),2))))
    assert (k5['a'],k5['rstar'],k5['delta'])==(2,1,2)
    validation=None
    if counter:
        validation=independent(counter['omega'])
        assert validation['rstar']==counter['rstar']
    after={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in before}
    assert before==after
    result=dict(status='COUNTEREXAMPLE' if counter else 'NO_COUNTEREXAMPLE_IN_FROZEN_SET',
        plan=json.loads((HERE/'phase2_frozen.json').read_text()),
        counts_by_n=counts,primary_records=records,targeted_records=extra,
        counterexample=counter,independent_validation=validation,
        K5_outside_family=k5,elapsed_seconds=time.monotonic()-START,
        protected_before=before,protected_after=after,production_trajectories=0,QPU_jobs=0)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','counts_by_n','targeted_records',
        'counterexample','independent_validation','elapsed_seconds')},indent=2))


if __name__=='__main__':main()

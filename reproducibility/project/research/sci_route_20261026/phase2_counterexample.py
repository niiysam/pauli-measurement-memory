"""Single analytically constructed counterexample, with independent matching."""
from collections import deque
from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.sci_route_20261026.phase2_exact import graph,analyze
from research.sci_route_20261026.exact_check import closure,parameters,selected_branches
from src.stabilizer import Pauli,rank
from src.ensemble import local_word

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase2_counterexample_results.json'


def matching_update(matching,edge):
    a,b=edge
    partner={x:y for p in matching for x,y in (p,p[::-1])}
    if partner.get(a)==b:return matching
    keep={p for p in matching if a not in p and b not in p}
    keep.add(tuple(sorted((a,b))))
    if a in partner and b in partner:
        keep.add(tuple(sorted((partner[a],partner[b]))))
    return frozenset(keep)


def matching_closure(edges,A):
    initial=frozenset();paths={initial:()};queue=deque([initial]);best=initial
    while queue:
        state=queue.popleft()
        for j,e in enumerate(edges):
            nxt=matching_update(state,e)
            assert all((a in A)!=(b in A) for a,b in nxt)
            if nxt not in paths:
                paths[nxt]=paths[state]+(j,);queue.append(nxt)
                if len(nxt)>len(best):best=nxt
            if len(paths)>20000:raise RuntimeError('Frozen state cap')
    return dict(max_pairs=len(best),states=len(paths),word=list(paths[best]))


def main():
    if OUT.exists():raise FileExistsError(OUT)
    prior=json.loads((HERE/'phase2_results.json').read_text())
    paths=list(prior['protected_before'])+[
        str((HERE/f).relative_to(ROOT)) for f in ('phase2_exact.py','phase2_results.json','phase2_frozen.json')]
    before={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in paths}
    start=time.monotonic()
    edges=[]
    for t in range(3):edges.extend(combinations((3*t,3*t+1,3*t+2),2))
    edges.extend(((2,3),(5,6)))
    omega=graph(9,edges)
    formal=analyze(omega)
    assert (formal['a'],formal['rstar'],formal['delta'])==(4,3,2)
    # Tree centres 0,1,2 in A; separator/leaves 3..9 in B.
    root_edges=[(0,5),(0,6),(0,3),(1,3),(1,7),(1,4),(2,4),(2,8),(2,9)]
    assert graph(9,[(i,j) for i in range(9) for j in range(i+1,9)
                    if set(root_edges[i])&set(root_edges[j])])==omega
    match=matching_closure(root_edges,{0,1,2})
    assert match['max_pairs']==3
    # All nine tree-edge bilinears represented independently on five qubits.
    gammas=[]
    for j in range(10):
        q=j//2
        gammas.append(Pauli.word('Z'*q+('X' if j%2==0 else 'Y')+'I'*(4-q)))
    bilinears=[]
    for i,j in root_edges:
        p=gammas[i]*gammas[j]
        bilinears.append(Pauli(p.x,p.z,(p.phase+1)%4))
    assert all(bilinears[i].anti(bilinears[j])==((omega[i]>>j)&1)
               for i in range(9) for j in range(9))
    bms=[p.x|(p.z<<5) for p in bilinears]
    assert rank(bms)==9
    bp,bt=closure(bms,5)
    assert len(bt)==8
    # One fixed original periodic ensemble, no new production dynamics.
    labels=['XXX']*13
    for j,w in {0:'ZXX',3:'XZX',4:'YXX',5:'XZX',7:'YXX',8:'XZX',10:'YXX'}.items():labels[j]=w
    ps=[Pauli.word(local_word(13,j,w)) for j,w in enumerate(labels)]
    ms,pars=parameters(ps,13)
    assert tuple(sum(ps[i+2].anti(ps[j+2])<<j for j in range(9)) for i in range(9))==omega
    assert all(not ps[i].anti(ps[j]) for i in range(2,11) for j in (0,1,11,12))
    phys_paths,top=closure(ms,13)
    rstar=len(top).bit_length()-1
    physical=dict(L=13,labels=labels,**pars,rstar=rstar,Smin=13-rstar,
                  gap=13-rstar-pars['k_static'],states=len(phys_paths),word=list(phys_paths[top]))
    assert (pars['s'],pars['a'],rstar,physical['Smin'],pars['k_static'],physical['gap'])==(13,5,6,7,5,2),physical
    # All 64 branches of this six-event witness; new physical counterexample only.
    signed_check=selected_branches(ps,phys_paths[top],13)
    assert signed_check['final_entropies']==[7]
    after={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in before}
    assert before==after
    elapsed=time.monotonic()-start
    assert elapsed<120
    result=dict(status='COUNTEREXAMPLE_CONFIRMED',plan=json.loads((HERE/'phase2_counterexample_frozen.json').read_text()),
        formal=formal,majorana_matching=match,root_edges=root_edges,
        majorana_pauli=dict(qubits=5,s=9,rstar=3,reachable_states=len(bp)),
        original_geometry=physical,signed_stim=signed_check,elapsed_seconds=elapsed,
        protected_before=before,protected_after=after,production_trajectories=0,QPU_jobs=0)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','formal','majorana_matching','original_geometry','signed_stim','elapsed_seconds')},indent=2))


if __name__=='__main__':main()

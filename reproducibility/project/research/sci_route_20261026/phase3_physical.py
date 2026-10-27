"""One frozen non-line-graph example in the original periodic ensemble."""
from pathlib import Path
import hashlib
import json
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from research.sci_route_20261026.exact_check import closure,parameters,selected_branches
from src.stabilizer import Pauli,rank
from src.ensemble import local_word

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase3_physical_results.json'


def cyclic_middle(target):
    n=len(target)
    third=[int(bool(target[j]>>((j+2)%n)&1)) for j in range(n)]
    adjacent=[int(bool(target[j]>>((j+1)%n)&1)) for j in range(n)]
    def allowed(j,a,b):return ((a!=0)^(third[j]!=b))==adjacent[j]
    for initial in range(3):
        feasible=[set() for _ in range(n)]
        feasible[-1]={a for a in range(3) if allowed(n-1,a,initial)}
        for j in range(n-2,-1,-1):
            feasible[j]={a for a in range(3) if any(allowed(j,a,b) for b in feasible[j+1])}
        if initial not in feasible[0]:continue
        middle=[initial]
        for j in range(n-1):
            middle.append(min(b for b in feasible[j+1] if allowed(j,middle[-1],b)))
        return ['X'+'XYZ'[s]+'XY'[t] for s,t in zip(middle,third)]
    return None


def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.monotonic()
    plan=json.loads((HERE/'phase3_physical_frozen.json').read_text())
    classification=json.loads((HERE/'phase3_graph_results.json').read_text())
    rec=classification['selected'];n=plan['L']
    assert rec['edge_bit_code']==plan['selected_edge_code']
    assert not rec['is_line_graph']
    target=[0]*n
    for i in range(rec['n']):
        for j in range(rec['n']):
            if rec['omega'][i]>>j&1:target[i+2]|=1<<(j+2)
    for i,j in plan['guard_edges']:
        target[i]|=1<<j;target[j]|=1<<i
    labels=cyclic_middle(target)
    assert labels==plan['labels_hand_derived_before_verification']
    old=classification['protected_before']
    protected=list(old)+[str((HERE/f).relative_to(ROOT)) for f in
        ('phase3_frozen.json','phase3_graphs.py','phase3_graph_results.json')]
    before={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in protected}
    ps=[Pauli.word(local_word(n,j,w)) for j,w in enumerate(labels)]
    observed=[sum(ps[i].anti(ps[j])<<j for j in range(n)) for i in range(n)]
    assert observed==target
    ms,pars=parameters(ps,n)
    block=[ms[i] for i in plan['block_anchors']]
    assert rank(block)==rec['n']
    path,top=closure(ms,n)
    rstar=len(top).bit_length()-1
    found=dict(**pars,rstar=rstar,Smin=n-rstar,gap=n-rstar-pars['k_static'])
    assert all(found[k]==value for k,value in plan['predicted'].items()),found
    witness=plan['maximum_rank_witness']
    assert rank([ms[j] for j in witness])==rstar
    assert all(not ps[i].anti(ps[j]) for i in witness for j in witness)
    check=selected_branches(ps,witness,n)
    assert check['final_entropies']==[found['Smin']]
    after={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in before}
    assert before==after
    elapsed=time.monotonic()-start
    assert elapsed<plan['cap']['seconds']
    result=dict(status='ORIGINAL_GEOMETRIC_COUNTEREXAMPLE_CONFIRMED',plan=plan,
        L=n,labels=labels,omega=observed,block_rank=rank(block),**found,
        reachable_states=len(path),witness=witness,signed_stim=check,
        elapsed_seconds=elapsed,protected_before=before,protected_after=after,
        production_trajectories=0,QPU_jobs=0)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in
        ('plan','protected_before','protected_after')},indent=2))


if __name__=='__main__':main()

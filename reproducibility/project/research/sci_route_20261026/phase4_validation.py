"""Finite-vector check of the specific projector regrouping proof step."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.stabilizer import Pauli
from src.ensemble import local_word
from src import reference
from research.sci_route_20261026.phase1_exact import components

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase4_validation_results.json'


class Action:
    def __init__(self,p,L):
        assert L<=10
        self.indices=np.arange(1<<L,dtype=np.int64)
        self.permuted=self.indices^p.x
        self.phases=np.array([(1j)**p.phase*(-1)**((i&p.z).bit_count())
                              for i in range(1<<L)])
    def __matmul__(self,psi):
        result=np.empty_like(psi)
        result[self.permuted]=self.phases*psi
        return result


def initial(kind,L):
    if kind=='zero':
        psi=np.zeros(1<<L,complex);psi[0]=1;return psi
    psi=np.ones(1,complex)
    for j in range(L):
        theta=np.pi*(j+1)/(2*L+3);phi=np.pi*j/7
        one=np.array([np.cos(theta/2),np.exp(1j*phi)*np.sin(theta/2)])
        psi=np.kron(one,psi)
    return psi


def unnormalized(psi,items,actions):
    for j,outcome in items:psi=(psi+outcome*(actions[j]@psi))/2
    return psi


def srank(psi,L):
    return int(np.sum(np.linalg.svd(psi.reshape(1<<(L//2),-1),compute_uv=False)>1e-10))


def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.monotonic();plan=json.loads((HERE/'phase4_validation_frozen.json').read_text())
    prior=json.loads((HERE/'phase3_physical_results.json').read_text())
    protected=list(prior['protected_before'])+[str((HERE/f).relative_to(ROOT)) for f in
        ('phase3_physical.py','phase3_physical_frozen.json','phase3_physical_results.json',
         'phase3_quotient_audit.py','phase3_quotient_audit_results.json','phase4_frozen.json',
         'phase4_validation_frozen.json')]+['src/reference.py']
    before={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in protected}
    records=[];L=10;A=set(range(5));B=set(range(5,10))
    for name,labels in [('commuting',['XXX']*L),('nonline',prior['labels'])]:
        ps=[Pauli.word(local_word(L,j,w)) for j,w in enumerate(labels)]
        omega=[sum(p.anti(q)<<j for j,q in enumerate(ps)) for p in ps]
        groups=components(omega)
        cross=[]
        for g in groups:
            support={j for idx in g for j in range(L) if (ps[idx].x|ps[idx].z)>>j&1}
            if support&A and support&B:cross.extend(g)
        cross=set(cross);actions=[Action(p,L) for p in ps]
        for kind in ('zero','generic'):
            for branch in ('all+','alternating'):
                outcomes=[1 if branch=='all+' or k%2==0 else -1 for k in range(len(plan['histories']))]
                items=list(zip(plan['histories'],outcomes))
                first=[i for i in items if i[0] in cross];local=[i for i in items if i[0] not in cross]
                psi=initial(kind,L)
                original=unnormalized(psi,items,actions)
                mid=unnormalized(psi,first,actions)
                regrouped=unnormalized(mid,local,actions)
                error=float(np.max(np.abs(original-regrouped)))
                prob=float(np.vdot(original,original).real)
                assert error<=plan['numerical_tolerances']['vector_absolute']
                assert abs(prob-float(np.vdot(regrouped,regrouped).real))<=1e-12
                rec=dict(ensemble=name,initial=kind,branch=branch,probability=prob,vector_error=error,
                    cross_anchors=sorted(cross),grouped_event_order=[j for j,_ in first+local])
                if prob<1e-13:
                    rec['status']='ZERO_BRANCH_RETAINED';records.append(rec);continue
                norm=float(np.vdot(mid,mid).real)
                final=original/np.sqrt(prob);mid=mid/np.sqrt(norm)
                rm,rf=srank(mid,L),srank(final,L)
                ent=reference.entropy(final,A);em=reference.entropy(mid,A)
                assert rf<=rm and ent<=np.log2(rm)+1e-9
                # Independent sequential projector routine, with vector actions.
                seq=psi.copy();pseq=1.
                for j,outcome in items:
                    seq,p1=reference.project(seq,actions[j],outcome);pseq*=p1
                assert abs(pseq-prob)<1e-12
                assert np.max(np.abs(seq-final))<1e-12
                if name=='commuting':assert ent<=4+1e-9
                rec.update(status='PASS',cross_Schmidt_rank=rm,final_Schmidt_rank=rf,
                    entropy=ent,cross_only_entropy=em,
                    local_outcome_entropy_increased=ent>em+1e-9)
                records.append(rec)
    bell=np.zeros(1<<L,complex)
    for v in range(32):bell[v|(v<<5)]=1/np.sqrt(32)
    eb=reference.entropy(bell,A)
    assert abs(eb-5)<1e-12 and eb>4
    after={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in before}
    assert before==after
    elapsed=time.monotonic()-start;assert elapsed<plan['cap_seconds']
    output=dict(status='TARGETED_REGROUPING_AND_RANK_CHECKS_PASS',plan=plan,records=records,
        negative_control=dict(initial='five Bell pairs across half chain',time=0,entropy=eb,
            conclusion='Product-state assumption is essential; not a counterexample to theorem'),
        elapsed_seconds=elapsed,protected_before=before,protected_after=after,
        new_production=0,QPU_jobs=0,statistical_samples=0)
    OUT.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ('plan','protected_before','protected_after')},indent=2))


if __name__=='__main__':main()

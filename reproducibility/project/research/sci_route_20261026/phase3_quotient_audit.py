"""Fixed algebraic audit, no quantum-state evolution or new graph search.

Before computation: central vector c=e4+e5; quotient identifies e5=e4.
Predicted quotient Omega=[6,13,3,18,8], with an A2/B4 tree root.
Read both Rstar values from already frozen round2 records, never refit them.
"""
from itertools import combinations
from pathlib import Path
import hashlib
import json

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase3_quotient_audit_results.json'


def pairing(x,y,omega):
    z=0
    for j,row in enumerate(omega):
        if x>>j&1:z^=(row&y).bit_count()&1
    return z


def project(x):return (x&31)^(((x>>5)&1)<<4)


def main():
    if OUT.exists():raise FileExistsError(OUT)
    source=HERE/'phase2_results.json';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    data=json.loads(source.read_text())['primary_records']
    full=next(r for r in data if r['n']==6 and r['edge_bit_code']==207)
    reduced=next(r for r in data if r['n']==5 and r['omega']==[6,13,3,18,8])
    c=(1<<4)|(1<<5)
    assert all(pairing(c,1<<j,full['omega'])==0 for j in range(6))
    assert all(pairing(1<<i,1<<j,full['omega'])==
               pairing(project(1<<i),project(1<<j),reduced['omega'])
               for i in range(6) for j in range(6))
    assert [x for x in range(64) if project(x)==0]==[0,c]
    root=[(0,2),(0,3),(0,4),(1,3),(1,5)]
    observed=[0]*5
    for i,j in combinations(range(5),2):
        if set(root[i])&set(root[j]):observed[i]|=1<<j;observed[j]|=1<<i
    assert observed==reduced['omega']
    assert all((a<2)!=(b<2) for a,b in root)
    assert full['rstar']==3 and reduced['rstar']==2
    assert all(not (full['omega'][i]>>j&1) for i in (0,4,5) for j in (0,4,5))
    assert digest==hashlib.sha256(source.read_bytes()).hexdigest()
    result=dict(status='CENTRAL_QUOTIENT_REDUCES_TO_KNOWN_PAIRING_MECHANISM',
        full_omega=full['omega'],quotient_omega=reduced['omega'],central_vector=c,
        quotient_map='(x&31) XOR (((x>>5)&1)<<4)',kernel=[0,c],root_edges=root,
        sublattice_A=[0,1],sublattice_B=[2,3,4,5],
        upper_bound='dimH<=dim project(H)+1<=2+1=3 for every history',
        lower_witness=[0,4,5],full_Rstar_from_old_data=full['rstar'],
        quotient_Rstar_from_old_data=reduced['rstar'],
        independent_theorem='Pairing and central quotient proofs in MASTER section15.3; this arithmetic audit checks their fixed matrices, not general originality.',
        source_sha256=digest,new_state_trajectories=0,QPU_jobs=0)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

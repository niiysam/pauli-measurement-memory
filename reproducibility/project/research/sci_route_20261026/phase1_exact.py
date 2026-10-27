"""Exact checks of local probability polynomials and block-rank absorption.

No quantum production, no Monte Carlo, no third-party source implementation.
Polynomial coefficients count full labelled Pauli configurations exactly.
"""
from collections import deque, defaultdict
from fractions import Fraction
from itertools import product, combinations
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.stabilizer import Pauli, rank
from src.ensemble import local_word

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase1_results.json'
WORDS=list(product('XYZ',repeat=3))


def add(a,b):
    c=[0]*max(len(a),len(b))
    for i,v in enumerate(a):c[i]+=v
    for i,v in enumerate(b):c[i]+=v
    return c


def mul(a,b):
    c=[0]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):c[i+j]+=x*y
    return c


def power(a,n):
    c=[1]
    for _ in range(n):c=mul(c,a)
    return c


def value(poly,q):
    z=1-2*q
    return sum(c*q**k*z**(len(poly)-1-k) for k,c in enumerate(poly))


def count_poly(indices):
    counts=[0]*4
    for i in indices:
        counts[sum(s!='Z' for s in WORDS[i])]+=1
    return counts


def check_local():
    # Cut between the second and third of four anchors. Three cross edges.
    cut=[0]*7
    for c,e,f,g,h,j in product('XYZ',repeat=6):
        labels=[('X','X',c),('X',e,f),(g,h,'X'),(j,'X','X')]
        ps=[Pauli.word(local_word(9,k+1,w)) for k,w in enumerate(labels)]
        if all(not ps[a].anti(ps[b]) for a,b in ((0,2),(1,2),(1,3))):
            cut[sum(s!='Z' for s in (c,e,f,g,h,j))]+=1
    # Homogeneous degree-three forms, using pX=pY=q, pZ=z.
    U3=[1,0,0,2]
    B1=[0,2,2,2]  # u2*(2q+z)-u3
    cut_pred=add(mul(U3,U3),mul(B1,B1))
    assert cut==cut_pred,(cut,cut_pred)
    # Seven anchors -2..4 on a nine-site open embedding; the middle three
    # are the candidate component. All 27 choices at every guard are summed.
    ops=[[Pauli.word(local_word(9,j,w)) for w in WORDS] for j in range(7)]
    t=[[[x.anti(y) for y in ops[j]] for x in ops[i]]
       for i,j in ((0,2),(1,2),(1,3),(2,3),(2,4),(3,4),(3,5),(4,5),(4,6))]
    am,ab,bb,ac,ad,bd,be,de,df=t
    # left(a,b) and right(b,d) each count six labels (two guards).
    left={};right={}
    for a,b in product(range(27),repeat=2):
        l0=count_poly([x for x in range(27) if not am[x][a]])
        l1=count_poly([x for x in range(27) if not ab[x][a] and not bb[x][b]])
        left[a,b]=mul(l0,l1)
    for b,d in product(range(27),repeat=2):
        r0=count_poly([x for x in range(27) if not be[b][x] and not de[d][x]])
        r1=count_poly([x for x in range(27) if not df[d][x]])
        right[b,d]=mul(r0,r1)
    iso=[0]*22
    triangle_choices=0
    for a,b,d in product(range(27),repeat=3):
        if not(ac[a][b] and ad[a][d] and bd[b][d]):continue
        triangle_choices+=1
        poly=mul(left[a,b],right[b,d])
        k=sum(s!='Z' for i in (a,b,d) for s in WORDS[i])
        for i,c in enumerate(poly):iso[k+i]+=c
    # p_iso= D*u3^4+2*u3^2*B1^3; D=6 q^2 z, then six free labels.
    D=[0,0,6,0]
    iso15=add(mul(D,power(U3,4)),[2*c for c in mul(power(U3,2),power(B1,3))])
    iso_pred=mul(iso15,power([1,2],6))
    assert iso==iso_pred,(iso,iso_pred)
    records=[]
    for q in (Fraction(0),Fraction(13,100),Fraction(1,3),Fraction(1,2)):
        p=(q,q,1-2*q)
        u2=sum(x*x for x in p);u3=sum(x**3 for x in p)
        rho=u3*u3+(u2-u3)**2
        pit=6*p[0]*p[1]*p[2]*u3**4+2*u3*u3*(u2-u3)**3
        assert value(cut,q)==rho and value(iso,q)==pit
        if q==Fraction(1,3):assert rho==Fraction(5,81) and pit==Fraction(2,6561)
        records.append(dict(q=str(q),cut=str(rho),cut_decimal=float(rho),
            isolated_triangle=str(pit),triangle_decimal=float(pit)))
    # Asymmetric distribution: direct weighted guard sums, not the polynomial.
    pi=dict(zip('XYZ',(Fraction(1,5),Fraction(3,10),Fraction(1,2))))
    weights=[pi[w[0]]*pi[w[1]]*pi[w[2]] for w in WORDS]
    lweight={};rweight={}
    for a,b in product(range(27),repeat=2):
        lweight[a,b]=sum(weights[x] for x in range(27) if not am[x][a])*sum(
            weights[x] for x in range(27) if not ab[x][a] and not bb[x][b])
        rweight[a,b]=sum(weights[x] for x in range(27) if not be[a][x] and not de[b][x])*sum(
            weights[x] for x in range(27) if not df[b][x])
    weighted=sum(weights[a]*weights[b]*weights[d]*lweight[a,b]*rweight[b,d]
        for a,b,d in product(range(27),repeat=3) if ac[a][b] and ad[a][d] and bd[b][d])
    u2=sum(x*x for x in pi.values());u3=sum(x**3 for x in pi.values())
    expected=6*pi['X']*pi['Y']*pi['Z']*u3**4+2*u3*u3*(u2-u3)**3
    assert weighted==expected,(weighted,expected)
    return dict(cut_assignments=3**6,triangle_choices=triangle_choices,
        total_triangle_assignments=27**3,cut_coefficients=cut,
        isolated_triangle_coefficients=iso,formula_coefficients=iso_pred,
        rational_cases=records,asymmetric_pi={k:str(v) for k,v in pi.items()},
        asymmetric_direct=str(weighted),asymmetric_formula=str(expected))


def form_update(g,j,omega):
    m=1<<j
    c={x for x in g if not((x & omega[j]).bit_count() & 1)}
    return frozenset(c|{x^m for x in c})


def reachable(omega):
    init=frozenset([0]);path={init:()};queue=deque([init]);largest=init
    while queue:
        g=queue.popleft()
        for j in range(len(omega)):
            h=form_update(g,j,omega)
            if h not in path:
                path[h]=path[g]+(j,);queue.append(h)
                if len(h)>len(largest):largest=h
            if len(path)>20000:raise RuntimeError('Exact state cap')
    return path,largest


def components(omega):
    unseen=set(range(len(omega)));out=[]
    while unseen:
        q=[unseen.pop()];comp=set(q)
        while q:
            i=q.pop()
            for j in list(unseen):
                if (omega[i]>>j)&1:unseen.remove(j);comp.add(j);q.append(j)
        out.append(sorted(comp))
    return out


def check_forms():
    forms=set()
    for n in range(1,7):
        edges=[(i,j) for i in range(n) for j in range(i+1,min(n,i+3))]
        if n==4:edges=list(combinations(range(n),2))
        for bits in range(1<<len(edges)):
            omega=[0]*n
            for k,(i,j) in enumerate(edges):
                if bits>>k&1:omega[i]|=1<<j;omega[j]|=1<<i
            forms.add(tuple(omega))
    cache={};groups=0;word_checks=0;collision=defaultdict(dict)
    c4=k4=None
    for omega in sorted(forms,key=lambda x:(len(x),x)):
        n=len(omega);paths,gmax=reachable(omega);rstar=len(gmax).bit_length()-1
        a=rank(omega)//2
        assert rstar<=n-a
        witness=paths[gmax]
        for g in paths:
            h=g
            for j in witness:h=form_update(h,j,omega)
            assert len(h)==len(gmax)
            word_checks+=1
        rsum=0
        for comp in components(omega):
            sub=tuple(sum(((omega[i]>>j)&1)<<b for b,j in enumerate(comp)) for i in comp)
            if sub not in cache:
                _,top=reachable(sub);cache[sub]=len(top).bit_length()-1
            rsum+=cache[sub]
        assert rsum==rstar
        edges=sum(v.bit_count() for v in omega)//2
        collision[n,a,edges][rstar]=dict(omega=list(omega),rstar=rstar,
            reachable_states=len(paths),maximum_word=list(witness),
            bandwidth_two=all(abs(i-j)<=2 for i in range(n) for j in range(n) if omega[i]>>j&1))
        groups+=len(paths)
        if n==4 and list(omega)==[10,5,10,5]:c4=dict(rstar=rstar,a=a,edges=edges)
        if n==4 and edges==6:k4=dict(rstar=rstar,a=a,edges=edges)
    # A C4 ordering of bandwidth two is 0-1-3-2-0.
    cyc=(6,9,9,6)
    cp,top=reachable(cyc)
    assert len(top)==8  # rank three, while alpha(C4)=two
    assert k4['rstar']==1 and k4['a']==2
    matched=[]
    for key,byr in sorted(collision.items()):
        if len(byr)>1:
            vals=list(byr.values())
            if all(v['bandwidth_two'] for v in vals):
                matched.append(dict(n_a_edges=list(key),examples=vals))
    return dict(alternating_forms=len(forms),reachable_states_total=groups,
        maximum_word_from_every_state_checks=word_checks,
        component_factorization_checks=len(forms),
        C4=dict(omega=list(cyc),rstar=3,independence_number=2),
        K4_outside_spatial_model=k4,
        same_n_rank_edge_count_different_rstar=matched[:1],
        all_matched_classes_count=len(matched))


def main():
    if OUT.exists():raise FileExistsError(OUT)
    before={}
    prior=json.loads((HERE/'exact_results.json').read_text())
    paths=list(prior['protected_before'])+[
        str((HERE/'exact_check.py').relative_to(ROOT)),
        str((HERE/'exact_results.json').relative_to(ROOT)),
        str((HERE/'frozen_plan.json').relative_to(ROOT))]
    for s in paths:before[s]=hashlib.sha256((ROOT/s).read_bytes()).hexdigest()
    started=time.monotonic()
    result=dict(frozen_plan=json.loads((HERE/'phase1_frozen.json').read_text()),
        protected_before=before,python=platform.python_version(),
        local_counting=check_local(),finite_forms=check_forms())
    elapsed=time.monotonic()-started
    assert elapsed<120,elapsed
    after={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in before}
    assert before==after
    result.update(status='PASS',elapsed_seconds=elapsed,protected_after=after,
        random_samples=0,QPU_jobs=0)
    OUT.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],seconds=elapsed,
        local_rational=result['local_counting']['rational_cases'],
        forms=result['finite_forms']),indent=2))


if __name__=='__main__':main()

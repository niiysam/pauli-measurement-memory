"""Cut-rank identities and a classical protected-repeat calculation.

No replacement of the original signed stabilizer dynamics. These are analysis
functions on an existing pure-state subspace and its external controller.
"""
from itertools import product
import numpy as np

WORDS=np.array(list(product(range(3),repeat=3)),int)


def symp(v,w,L):
    mask=(1<<L)-1
    return (((v&mask)&(w>>L)).bit_count()+((v>>L)&(w&mask)).bit_count())&1


def canonical(rows):
    basis={}
    for v in rows:
        while v:
            k=v.bit_length()-1
            if k in basis:v^=basis[k]
            else:
                basis[k]=v;break
    for k in sorted(basis):
        for h in sorted(basis):
            if h>k and (basis[h]>>k)&1:basis[h]^=basis[k]
    return tuple(basis[k] for k in sorted(basis,reverse=True))


def projected_basis(rows,mask):
    """Eliminate a projection, retaining lifts and its kernel in the full space."""
    basis={};kernel=[]
    for v in rows:
        p=v&mask;lift=v
        while p:
            k=p.bit_length()-1
            if k in basis:
                b,w=basis[k];p^=b;lift^=w
            else:
                basis[k]=(p,lift);break
        if not p:kernel.append(lift)
    return basis,kernel


def lift_projection(v,basis):
    lift=0
    while v:
        k=v.bit_length()-1
        if k not in basis:return None
        p,w=basis[k];v^=p;lift^=w
    return lift


class Cut:
    def __init__(self,rows,L,nA=None):
        self.L=L;self.rows=list(rows);self.nA=L//2 if nA is None else nA
        a=(1<<self.nA)-1;b=((1<<L)-1)^a
        self.A=a|(a<<L);self.B=b|(b<<L)
        self.pB,self.localA=projected_basis(rows,self.B)
        _,self.localB=projected_basis(rows,self.A)
        self.full,_=projected_basis(rows,(1<<(2*L))-1)
        self.S=self.nA-len(self.localA)
        assert self.S==L-self.nA-len(self.localB)

    def classify(self,m):
        """D=0, G=1, loss=2, active-neutral=3; chi=-1 if not needed."""
        member=lift_projection(m,self.full) is not None
        a=int(any(symp(m,v,self.L) for v in self.localA))
        b=int(any(symp(m,v,self.L) for v in self.localB))
        if member:
            assert a==b==0
            return 0,a,b,-1
        if a and b:return 1,a,b,-1
        if a or b:return 3,a,b,-1
        v=lift_projection(m&self.B,self.pB)
        assert v is not None  # P_B V = (V_B)^perp for a pure state.
        chi=symp(m,v,self.L)
        return (2 if chi==0 else 3),a,b,chi


def unsigned_update(rows,m,L):
    anti=[i for i,v in enumerate(rows) if symp(v,m,L)]
    if not anti:return canonical(rows)
    pivot=anti[0];old=rows[pivot];r=list(rows)
    for i in anti[1:]:r[i]^=old
    r[pivot]=m
    return canonical(r)


def proposal(q,c,code):
    pi=np.array([q,q,1-2*q])
    if code<0:raise ValueError('Previous controller word required')
    old=WORDS[code]
    return np.prod(c*(WORDS==old)+(1-c)*pi[WORDS],axis=1)


def protected_repeat(q,c):
    """Exact controller-only probability; no quantum independence assumption.

    Four overlapping neighbors have 1,2,2,1 relevant match bits (64 states).
    Ignore disjoint-anchor visits. Return-to-anchor competes with a killed
    neighbor visit; the remaining classical memory is retained exactly.
    """
    pi=np.array([q,q,1-2*q]);bits=np.array(list(product((0,1),repeat=6)))
    groups=[(0,),(1,2),(3,4),(5,)]
    # j-2 overlaps site0; j-1 sites0,1; j+1 sites1,2; j+2 site2.
    word_sites=[0,0,1,1,2,2]
    result=0.;simple=0.;cache={};detail=[]
    for w in WORDS:
        p=float(np.prod(pi[w]));stay=float(np.prod(c+(1-c)*pi[w]))
        probs=tuple(float(pi[w[i]]) for i in word_sites)
        if probs not in cache:
            mu=np.prod(np.where(bits==1,np.array(probs),1-np.array(probs)),axis=1)
            K=np.zeros((64,64))
            B=0.
            for group in groups:
                B+=(1-float(np.prod([2*probs[k]-1 for k in group])))/2
                unchanged=[i for i in range(6) if i not in group]
                same=np.all(bits[:,None,unchanged]==bits[None,:,unchanged],axis=2)
                P=same.astype(float)
                for k in group:
                    P*=c*(bits[:,None,k]==bits[None,:,k])+(1-c)*np.where(bits[None,:,k]==1,probs[k],1-probs[k])
                even=(np.sum(1-bits[:,group],axis=1)%2==0)
                K+=P*even[None,:]/5
            assert np.max(K.sum(axis=1))<=.8+1e-12
            survival=float(mu@np.linalg.solve(np.eye(64)-K,np.ones(64))/5)
            cache[probs]=(survival,B)
        survival,B=cache[probs]
        result+=p*stay*survival
        simple+=p*p/(1+B)
        detail.append(dict(word=''.join('XYZ'[i] for i in w),p=p,repeat=stay,
            no_anti_until_revisit=survival,contribution=p*stay*survival))
    return dict(q=q,c=c,probability=result,c0_closed_form=simple,details=detail)

"""Minimal mixed-state extension for the c=0 physical baseline.

Reuses the existing signed Pauli algebra and projective replacement rule.
Variable stabilizer rank is the new operation; no changes to old engines.
"""
import numpy as np
from src.stabilizer import Stabilizer, Pauli, rank
from src.audit_validation import make_stim, stim_pauli, stim_entropy


class MixedState(Stabilizer):
    def __init__(self, L):
        if not isinstance(L,int) or not 1<=L<=48:
            raise ValueError('Baseline hard cap L<=48')
        self.L=L
        self.g=[]

    def measure(self, measured, uniform=.5, forced=None):
        if (measured.phase-(measured.x & measured.z).bit_count())%2:
            raise ValueError('Hermitian measurement required')
        if (measured.x | measured.z)>>self.L:
            raise ValueError('Outside system')
        if not 0<=uniform<1 or forced not in (None,-1,1):
            raise ValueError('Invalid outcome control')
        anti=any(g.anti(measured) for g in self.g)
        if anti:
            out,p,_=super().measure(measured,uniform,forced)
            return out,p,1  # replacement, entropy unchanged
        bits=[g.x|(g.z<<self.L) for g in self.g]
        v=measured.x|(measured.z<<self.L)
        if rank(bits+[v])==len(bits):
            out,p,_=super().measure(measured,uniform,forced)
            return out,p,0  # redundant, possibly negative deterministic outcome
        out=forced if forced is not None else (1 if uniform<.5 else -1)
        self.g.append(measured.signed(out))
        return out,.5,2  # independent commuting logical measurement

    def entropy(self,subset):
        subset=set(subset)
        if not subset<=set(range(self.L)):
            raise ValueError('Invalid subsystem')
        mask=((1<<self.L)-1)^sum(1<<j for j in subset)
        rows=[(g.x&mask)|((g.z&mask)<<self.L) for g in self.g]
        return len(subset)-len(self.g)+rank(rows)

    def check(self):
        assert 0<=len(self.g)<=self.L
        assert all((g.x|g.z)>>self.L==0 for g in self.g)
        assert all((g.phase-(g.x&g.z).bit_count())%2==0 for g in self.g)
        assert all(not a.anti(b) for i,a in enumerate(self.g) for b in self.g[:i])
        assert rank([g.x|(g.z<<self.L) for g in self.g])==len(self.g)

    @property
    def mixed_entropy(self):
        return self.L-len(self.g)


class BaselinePureState(Stabilizer):
    """Constructor-only extension, only the newly authorized L=48."""
    def __init__(self,L,initial='zero'):
        if not isinstance(L,int) or not 1<=L<=48:
            raise ValueError('Baseline hard cap L<=48')
        self.L=L
        if initial=='zero':self.g=[Pauli(0,1<<j) for j in range(L)]
        elif initial=='plus':self.g=[Pauli(1<<j,0) for j in range(L)]
        else:raise ValueError(initial)


def mixed_replay(anchors,codes,born,L,table,verify_stim=True):
    """Stop at the absorbing zero of S_mix; retain every processed outcome.

    The complete original controller history remains in the immutable source.
    No Born outcomes after purification are needed to compute S_mix=0.
    """
    state=MixedState(L)
    sim=None
    if verify_stim:
        sim=make_stim(2*L,'zero')
        for j in range(L):
            sim.h(j);sim.cnot(j,j+L)
        sm=[[stim_pauli(p,2*L) for p in row] for row in table]
    drops=[];outcomes=[];kinds=[];entropy=[];checks=0
    for event,(j,code,u) in enumerate(zip(anchors,codes,born),1):
        M=table[int(j)][int(code)]
        if sim is not None:
            observable=sm[int(j)][int(code)]
            expected=sim.peek_observable_expectation(observable)
        out,p,kind=state.measure(M,float(u))
        s=state.mixed_entropy
        if sim is not None:
            assert out==(1 if u<(1+expected)/2 else -1)
            assert p==(.5 if expected==0 else 1.)
            sim.postselect_observable(observable,desired_value=(out==-1))
            if kind==2:
                assert stim_entropy(sim,range(L,2*L))==s
                checks+=1
        outcomes.append(out);kinds.append(kind);entropy.append(s)
        if kind==2:drops.append(event)
        if s==0:break
    state.check()
    if sim is not None:
        assert all(sim.peek_observable_expectation(stim_pauli(g,2*L))==1 for g in state.g)
    purify_event=len(outcomes) if state.mixed_entropy==0 else -1
    return dict(outcome=np.array(outcomes,np.int8),kind=np.array(kinds,np.uint8),
        mixed_entropy=np.array(entropy,np.uint8),drops=np.array(drops,np.int64),
        purification_event=np.array(purify_event,np.int64),
        horizon_events=np.array(len(anchors),np.int64),
        stim_entropy_checks=np.array(checks,np.int64),
        final=np.array([[g.x,g.z,g.phase] for g in state.g],np.uint64))


def entropy_at(drops,L,event):
    return L-int(np.searchsorted(drops,event,side='right'))

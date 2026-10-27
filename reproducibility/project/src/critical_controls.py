"""Gate C: explicitly authorized L<=24 extension; archived engine unchanged."""
import numpy as np
from src.stabilizer import Stabilizer, Pauli, rank
from src.ensemble import LocalMemory
from src.history_controls import (generate_history, Q_CASE, MODEL_SEED,
                                  pauli_table)


class GateCState(Stabilizer):
    """Reuse all phase-aware updates/rank methods with an explicit new bound."""
    def __init__(self, L, initial="zero"):
        if not 1 <= L <= 24:
            raise ValueError("Gate C hard bound: 1<=L<=24")
        self.L=L
        if initial=="zero":
            self.g=[Pauli(0,1<<j) for j in range(L)]
        elif initial=="plus":
            self.g=[Pauli(1<<j,0) for j in range(L)]
        elif initial=="ghz":  # validation only, not a production initial state
            self.g=[Pauli((1<<L)-1,0)]+[Pauli(0,1|(1<<j)) for j in range(1,L)]
        else:
            raise ValueError(initial)
        self.check()


def history(L,q,c,rep,sweeps):
    if L not in (12,16,20,24):
        raise ValueError("Frozen size grid")
    if L<=16:
        return generate_history(L,q,c,rep,sweeps)
    if q not in Q_CASE or c not in (0,.6):
        raise ValueError("Frozen model")
    key=[MODEL_SEED,L,Q_CASE[q],rep]
    ar,fr,hr,br,ir=[np.random.default_rng(s)
                    for s in np.random.SeedSequence(key).spawn(5)]
    N=L*sweeps
    anchors=ar.integers(L,size=N).astype(np.uint8)
    fresh,hold,born=fr.random((N,3)),hr.random((N,3)),br.random(N)
    memory=LocalMemory(L,[q,q,1-2*q],c,ir.random((L,3)))
    codes=np.empty(N,np.uint8)
    for n,j in enumerate(anchors):
        labels=memory.update(int(j),fresh[n],hold[n])
        codes[n]=9*int(labels[0])+3*int(labels[1])+int(labels[2])
    return anchors,codes,born,key


def same_unsigned(a,b):
    L=a.L
    return rank([g.x|(g.z<<L) for g in a.g+b.g])==L


def observables(state):
    """Half entropy and I3 of four consecutive equal quarters, in bits."""
    L=state.L
    if L%4:
        raise ValueError("Four equal contiguous quarters required")
    whole=set(range(L))
    A,B,C,D=[set(range(j*L//4,(j+1)*L//4)) for j in range(4)]
    def entropy(subset):
        # Pure state: S(X)=S(complement X). Minimize retained bit dimensions.
        if len(subset)<L//2:
            subset=whole-subset
        return state.entropy(subset)
    half=entropy(A|B)
    i3=(entropy(A)+entropy(B)+entropy(C)+entropy(D)
        -half-entropy(A|C)-entropy(B|C))
    return half,i3


def simulate_pair(anchors,codes,born,L,initials=("zero","plus"),stride=4):
    """Every Born outcome retained; observables on physical and active grids.

    Unsigned-state coalescence certifies subsequent entropies match for this
    operator path, not equality of signed states or a universal mixing bound.
    Both signed states continue to evolve and retain their own outcomes.
    """
    states=[GateCState(L,s) for s in initials]
    table=pauli_table(L)
    N=len(anchors)
    outcome=np.empty((len(states),N),np.int8)
    changed=np.empty((len(states),N),bool)
    counts=[0]*len(states)
    physical=[[[0,0,0,0]] for _ in states]
    active=[[[0,0,0,0]] for _ in states]
    coalesced=-1
    for n,(j,code) in enumerate(zip(anchors,codes),1):
        due=[]
        for k,state in enumerate(states):
            out,p,ch=state.measure(table[int(j)][int(code)],float(born[n-1]))
            outcome[k,n-1],changed[k,n-1]=out,ch
            counts[k]+=int(ch)
            due.append(ch and counts[k]%(stride*L)==0)
        if n%(stride*L)==0:
            if len(states)==2 and coalesced<0 and same_unsigned(*states):
                coalesced=n
            values=[observables(s) for s in states]
            if coalesced>=0:
                assert values[0]==values[1]
            for k,value in enumerate(values):
                physical[k].append([n,counts[k],*value])
                if due[k]:
                    active[k].append([n,counts[k],*value])
        else:
            for k,needed in enumerate(due):
                if needed:
                    active[k].append([n,counts[k],*observables(states[k])])
    for state in states:
        state.check()
    if coalesced>=0:
        assert same_unsigned(*states)
    result=dict(outcome=outcome,changed=changed,
                coalesced_event=np.array(coalesced,np.int32))
    for k,initial in enumerate(initials):
        result[initial+"_physical"]=np.array(physical[k],np.int32)
        result[initial+"_active"]=np.array(active[k],np.int32)
        result[initial+"_final"]=np.array(
            [[g.x,g.z,g.phase] for g in states[k].g],np.int64)
    return result

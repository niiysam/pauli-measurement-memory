"""Bounded L<=96 constructor/storage extension, archived algorithms unchanged."""
import numpy as np
import time
from src.stabilizer import Stabilizer, Pauli
from src.baseline_validation import MixedState
from src.audit_validation import make_stim, stim_pauli, stim_entropy


class Pure96(Stabilizer):
    def __init__(self, L, initial='zero'):
        if not isinstance(L,int) or not 1<=L<=96:
            raise ValueError('This extension is bounded by L96')
        self.L=L
        if initial=='zero':self.g=[Pauli(0,1<<j) for j in range(L)]
        elif initial=='plus':self.g=[Pauli(1<<j,0) for j in range(L)]
        else:raise ValueError(initial)


class Mixed96(MixedState):
    def __init__(self,L):
        if not isinstance(L,int) or not 1<=L<=96:
            raise ValueError('This extension is bounded by L96')
        self.L=L;self.g=[]


def pack(generators,L):
    """Lossless fixed-width hexadecimal arrays; never cast x/z to uint64."""
    width=(L+3)//4
    return dict(x_hex=np.array([format(g.x,f'0{width}x') for g in generators],dtype=f'U{width}'),
                z_hex=np.array([format(g.z,f'0{width}x') for g in generators],dtype=f'U{width}'),
                phase=np.array([g.phase for g in generators],np.uint8))


def unpack(data):
    return [Pauli(int(x,16),int(z,16),int(p)) for x,z,p in zip(data['x_hex'],data['z_hex'],data['phase'])]


def mixed96_replay(anchors,codes,born,L,table,verify=True,deadline=None):
    """Same mixed-state updates; Bell reference is checked at each entropy drop.

    Stops only at the absorbing pure state, or the fixed full horizon.
    """
    state=Mixed96(L);sim=None
    if verify:
        sim=make_stim(2*L,'zero')
        for j in range(L):sim.h(j);sim.cnot(j,j+L)
        stable=[[stim_pauli(p,2*L) for p in row] for row in table]
    drops=[];outcomes=[];kinds=[];entropy=[];checks=0
    for event,(j,code,u) in enumerate(zip(anchors,codes,born),1):
        if deadline is not None and event%(64*L)==0 and time.monotonic()>deadline:
            raise TimeoutError('Frozen batch budget reached during mixed replay')
        M=table[int(j)][int(code)]
        if sim is not None:
            operator=stable[int(j)][int(code)]
            expected=sim.peek_observable_expectation(operator)
        out,p,kind=state.measure(M,float(u));s=state.mixed_entropy
        if sim is not None:
            assert out==(1 if u<(1+expected)/2 else -1)
            assert p==(.5 if expected==0 else 1.)
            sim.postselect_observable(operator,desired_value=(out==-1))
            if kind==2:
                assert stim_entropy(sim,range(L,2*L))==s
                checks+=1
        outcomes.append(out);kinds.append(kind);entropy.append(s)
        if kind==2:drops.append(event)
        if s==0:break
    state.check()
    if sim is not None:
        assert all(sim.peek_observable_expectation(stim_pauli(g,2*L))==1 for g in state.g)
    packed=pack(state.g,L)
    assert unpack(packed)==state.g
    return dict(outcome=np.array(outcomes,np.int8),kind=np.array(kinds,np.uint8),
        mixed_entropy=np.array(entropy,np.uint8),drops=np.array(drops,np.int64),
        purification_event=np.array(len(outcomes) if state.mixed_entropy==0 else -1,np.int64),
        horizon_events=np.array(len(anchors),np.int64),stim_entropy_checks=np.array(checks,np.int64),
        final_x_hex=packed['x_hex'],final_z_hex=packed['z_hex'],final_phase=packed['phase'])

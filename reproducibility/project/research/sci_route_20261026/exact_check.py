"""Bounded scientific counterexample checks, not a production simulator.

Independent exhaustive *subspace-set* update; existing signed engine and Stim
are used only for the small explicitly chosen outcome branches. No random
samples, hardware access, old-code edits, or asymptotic inference from fits.
"""
from collections import deque
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import scipy
import stim
from src.stabilizer import Pauli, rank
from src.baseline_validation import MixedState
from src.audit_validation import stim_entropy, stim_pauli
from src.ensemble import local_word

HERE = Path(__file__).resolve().parent
PROTECTED = [ROOT / p for p in [
    'CRITICALITY_IBM_MASTER.md', 'RESEARCH_DIRECTION_REVIEW.md',
    'GATE_H_REPORT.md', 'src/stabilizer.py', 'src/baseline_validation.py',
    'src/audit_validation.py', 'src/ensemble.py',
    'results/criticality_followup_20261026/config.json',
    'results/criticality_followup_20261026/final_decision.json']]


def hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in PROTECTED}


def symp(a, b, L):
    mask = (1 << L) - 1
    return (((a & mask) & (b >> L)).bit_count()
            + ((a >> L) & (b & mask)).bit_count()) & 1


def update(group, m, L):
    # Enumerates ALL unsigned group elements. No tableau pivot/update reuse.
    commute = {v for v in group if not symp(v, m, L)}
    return frozenset(commute | {v ^ m for v in commute})


def closure(ms, L):
    initial = frozenset([0])
    paths = {initial: ()}
    queue = deque([initial])
    max_group = initial
    while queue:
        g = queue.popleft()
        for j, m in enumerate(ms):
            h = update(g, m, L)
            if h not in paths:
                paths[h] = paths[g] + (j,)
                queue.append(h)
                if len(h) > len(max_group):
                    max_group = h
            if len(paths) > 20000:
                raise RuntimeError('Frozen exact-enumeration budget exceeded')
    assert all(len(g) & (len(g)-1) == 0 for g in paths)
    return paths, max_group


def words(L, A='X', B='Z', C='Y', modify=False, commuting=False):
    assert L in (7, 14)
    labels = [A*3 for _ in range(L)]
    if not commuting:
        for offset in range(0, L, 7):
            labels[offset+3] = A+B+A
            labels[offset+4] = C+A+A
        if modify:
            labels[0] = B+A+A  # unconstrained guard label, not triangle edge
    return [Pauli.word(local_word(L, j, word))
            for j, word in enumerate(labels)], labels


def parameters(paulis, L):
    ms = [p.x | (p.z << L) for p in paulis]
    omega = [sum(symp(a, b, L) << j for j, b in enumerate(ms)) for a in ms]
    s = rank(ms)
    a = rank(omega)//2
    return ms, dict(s=s, a=a, kernel=L-s, k_static=L-s+a)


def selected_branches(paulis, sequence, L):
    """All 2^n input uniforms, including deterministic cases; no postselection.

    Stim Bell-pair purification is initialized independently. Forced matching
    is only an exact check of each Born branch; no branches are discarded.
    """
    entropies = set()
    checked_events = 0
    for choices in itertools.product((.25, .75), repeat=len(sequence)):
        state = MixedState(L)
        sim = stim.TableauSimulator(seed=2026102600)
        sim.set_num_qubits(2*L)
        for j in range(L):
            sim.h(j)
            sim.cnot(j, j+L)
        for idx, u in zip(sequence, choices):
            p = paulis[idx]
            op = stim_pauli(p, 2*L)
            ex = sim.peek_observable_expectation(op)
            out, prob, _ = state.measure(p, u)
            assert out == (1 if u < (1+ex)/2 else -1)
            assert prob == (.5 if ex == 0 else 1.)
            sim.postselect_observable(op, desired_value=(out == -1))
            assert state.mixed_entropy == stim_entropy(sim, range(L, 2*L))
            checked_events += 1
        state.check()
        entropies.add(state.mixed_entropy)
    return dict(branch_inputs=2**len(sequence), checked_events=checked_events,
                final_entropies=sorted(entropies))


def rational_moments(paths, ms, L, T=16):
    """Uniform-anchor finite-state expectation, exact rational arithmetic."""
    prob = {frozenset([0]): Fraction(1)}
    data = []
    for t in range(T+1):
        expectation = sum(p*(L-(len(g).bit_length()-1)) for g, p in prob.items())
        data.append([t, str(expectation)])
        nxt = {}
        for g, p in prob.items():
            for m in ms:
                h = update(g, m, L)
                assert h in paths
                nxt[h] = nxt.get(h, Fraction(0)) + p / len(ms)
        prob = nxt
        assert sum(prob.values()) == 1
    return data


def main():
    started = time.monotonic()
    before = hashes()
    config = json.loads((HERE/'frozen_plan.json').read_text())
    results = dict(config=config, protected_before=before, cases=[],
                   versions=dict(python=platform.python_version(),
                       numpy=np.__version__, scipy=scipy.__version__,
                       stim=stim.__version__), random_samples=0, qpu_jobs=0)
    for name, L, modify, commuting, target in [
        ('single_triangle_global_relation', 7, False, False, (6,1,2,2)),
        ('single_triangle_broken_relation', 7, True, False, (7,2,2,3)),
        ('two_isolated_triangles', 14, False, False, (13,2,3,4)),
        ('commuting_negative_control', 7, False, True, (7,0,0,0))]:
        ps, labels = words(L, modify=modify, commuting=commuting)
        ms, pars = parameters(ps, L)
        paths, largest = closure(ms, L)
        smin = L-(len(largest).bit_length()-1)
        observed = (pars['s'],pars['a'],pars['k_static'],smin)
        assert observed == target, (name, observed, target)
        rec = dict(name=name,L=L,labels=labels,**pars,
            reachable_subspaces=len(paths),minimum_entropy=smin,
            gap=smin-pars['k_static'],witness=list(paths[largest]))
        # Enumeration checks ALL allowed successor subspaces, not random runs.
        assert all(len(update(g,m,L)) >= len(g) for g in paths for m in ms)
        if name == 'single_triangle_broken_relation':
            rec['stim'] = selected_branches(ps, paths[largest], L)
            assert rec['stim']['final_entropies'] == [smin]
            central = Pauli.word('IIXIIIX')
            assert all(not central.anti(p) for p in ps)
            assert rank(ms+[central.x|(central.z<<L)]) == pars['s']
            extended = ps+[central]
            exms, _ = parameters(extended, L)
            ex_paths, ex_largest = closure(exms, L)
            ex_min = L-(len(ex_largest).bit_length()-1)
            assert ex_min == pars['k_static']
            rec['center_access_control'] = dict(minimum_entropy=ex_min,
                reachable_subspaces=len(ex_paths),
                note='Diagnostic extra weight-2 measurement, outside original model')
            rec['exact_uniform_anchor_mean_entropy'] = rational_moments(paths,ms,L)
        results['cases'].append(rec)
    # Six label relabelings of the broken-relation scientific counterexample.
    relabel=[]
    for A,B,C in itertools.permutations('XYZ'):
        ps,_ = words(7,A,B,C,modify=True)
        ms,pars=parameters(ps,7)
        paths,largest=closure(ms,7)
        ent=7-(len(largest).bit_length()-1)
        assert (pars['s'],pars['a'],pars['k_static'],ent)==(7,2,2,3)
        relabel.append(dict(ABC=A+B+C, minimum_entropy=ent))
    results['relabelings']=relabel
    # Enumerate all 3^6 free guard labels for ONE fixed triangle permutation.
    # This is a finite theorem-constraint check, not a parameter/data search.
    local_count=0
    for free in itertools.product('XYZ',repeat=6):
        labels=[free[0]+free[1]+'X',free[2]+'XX','XXX','XZX','YXX',
                'XX'+free[3],'X'+free[4]+free[5]]
        ps=[Pauli.word(local_word(7,j,v)) for j,v in enumerate(labels)]
        ms,pars=parameters(ps,7)
        assert pars['kernel'] <= 2
        assert all(ps[i].anti(ps[j]) == int(i!=j) for i in (2,3,4) for j in (2,3,4))
        assert all(not ps[i].anti(ps[j]) for i in (2,3,4) for j in (0,1,5,6))
        local_count+=1
    results['guard_constraint_cases']=local_count
    results['motif_probability']={}
    for q in (Fraction(13,100),Fraction(1,3)):
        pi=(q,q,1-2*q)
        p=sum(2*pi[i]**13*pi[(i+1)%3]*pi[(i+2)%3] for i in range(3))
        results['motif_probability'][str(q)]=dict(exact=str(p),
            probability=float(p),asymptotic_gap_density_lower_bound=float(p/7))
    after=hashes()
    assert before == after, 'Protected existing files changed'
    results['protected_after']=after
    results['elapsed_seconds']=time.monotonic()-started
    results['status']='PASS'
    out=HERE/'exact_results.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(dict(status=results['status'],
        cases=[{k:r[k] for k in ('name','s','a','k_static','minimum_entropy','gap','reachable_subspaces')}
               for r in results['cases']],
        guard_cases=local_count, elapsed_seconds=results['elapsed_seconds']),indent=2))


if __name__ == '__main__':
    main()

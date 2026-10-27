"""Bounded independent checks for the revised protocol; no phase scan.

Stim evolves its own inverse tableau. The existing engine is never used to
initialize or update Stim. postselect_observable only couples a Born-sampled
branch between implementations: no histories or outcomes are discarded.
"""
import itertools
import numpy as np
import stim
from src.stabilizer import Pauli, Stabilizer
from src.ensemble import LocalMemory, local_word
from src import reference as dense


def stim_pauli(p, n):
    word = ''.join('IXZY'[((p.x >> j) & 1) + 2*((p.z >> j) & 1)]
                   for j in range(n))
    result = stim.PauliString(word)
    result.sign = (1j)**((p.phase - (p.x & p.z).bit_count()) % 4)
    return result


def array_rank(a):
    """Independent byte-array elimination, not the project's integer rank."""
    a = np.asarray(a, dtype=np.uint8).copy()
    if a.ndim != 2:
        raise ValueError('Matrix required')
    pivot = 0
    for col in range(a.shape[1]):
        nz = np.flatnonzero(a[pivot:, col])
        if not len(nz):
            continue
        k = pivot + int(nz[0])
        a[[pivot, k]] = a[[k, pivot]]
        for row in range(a.shape[0]):
            if row != pivot and a[row, col]:
                a[row] ^= a[pivot]
        pivot += 1
        if pivot == len(a):
            break
    return pivot


def stim_entropy(sim, subset):
    n = sim.num_qubits
    other = sorted(set(range(n)) - set(subset))
    a = []
    for g in sim.canonical_stabilizers():
        x, z = g.to_numpy()
        a.append(np.concatenate((x[other], z[other])))
    return array_rank(np.array(a).reshape(n, 2*len(other))) - len(other)


def i3(entropy, n):
    if n % 4:
        raise ValueError('Equal quarters required')
    A, B, C, D = [set(range(j*n//4, (j+1)*n//4)) for j in range(4)]
    return (entropy(A)+entropy(B)+entropy(C)+entropy(D)
            -entropy(A|B)-entropy(A|C)-entropy(B|C))


def make_stim(n, initial):
    sim = stim.TableauSimulator(seed=2026102500)
    sim.set_num_qubits(n)
    if initial == 'plus':
        sim.h(*range(n))
    elif initial == 'ghz':
        sim.h(0)
        for j in range(1, n):
            sim.cnot(0, j)
    elif initial != 'zero':
        raise ValueError(initial)
    return sim


def history(n, q, c, events, seed):
    rngs = [np.random.default_rng(s) for s in
            np.random.SeedSequence(seed).spawn(5)]
    ar, fr, hr, br, ir = rngs
    anchors = ar.integers(n, size=events, dtype=np.uint8)
    fresh, hold, born = fr.random((events, 3)), hr.random((events, 3)), br.random(events)
    initial_u = ir.random((n, 3))
    mem = LocalMemory(n, [q, q, 1-2*q], c, initial_u)
    labels = np.array([mem.update(int(j), f, h)
                       for j, f, h in zip(anchors, fresh, hold)], np.uint8)
    return dict(anchors=anchors, labels=labels, born=born,
                fresh=fresh, hold=hold, initial_uniforms=initial_u,
                seed_key=np.array(seed, dtype=np.int64))


def check_step(state, sim, measured, uniform):
    operator = stim_pauli(measured, state.L)
    expected = sim.peek_observable_expectation(operator)
    out, probability, changed = state.measure(measured, float(uniform))
    assert out == (1 if uniform < (1+expected)/2 else -1)
    assert probability == (0.5 if expected == 0 else 1.0)
    assert changed == (expected == 0)
    sim.postselect_observable(operator, desired_value=(out == -1))
    # L independent +1 checks determine the full signed pure state uniquely.
    state.check()
    assert all(sim.peek_observable_expectation(stim_pauli(g, state.L)) == 1
               for g in state.g)
    return out, probability, changed


def validate_pure(n, initial, h, state_factory=Stabilizer, use_dense=True):
    state, sim = state_factory(n, initial), make_stim(n, initial)
    psi = dense.initial_state(n, initial) if use_dense else None
    subsets = [[], list(range(n//2)), list(range(0, n, 2)), list(range(n))]
    records, max_error = [], 0.0
    for anchor, labels, u in zip(h['anchors'], h['labels'], h['born']):
        word = local_word(n, int(anchor), ''.join('XYZ'[a] for a in labels))
        measured = Pauli.word(word)
        if use_dense:
            operator = dense.matrix(word)
            pplus = float((1+np.vdot(psi, operator@psi).real)/2)
        out, prob, changed = check_step(state, sim, measured, float(u))
        if use_dense:
            assert out == (1 if u < pplus else -1)
            psi, pdense = dense.project(psi, operator, out)
            max_error = max(max_error, abs(prob-pdense))
            assert abs(prob-pdense) < 2e-10
        for subset in subsets:
            value = state.entropy(subset)
            assert value == stim_entropy(sim, subset)
            if use_dense:
                error = abs(value-dense.entropy(psi, subset))
                max_error = max(max_error, error)
                assert error < 2e-10
        half = state.half_entropy()
        tripartite = i3(state.entropy, n) if n % 4 == 0 else np.nan
        if n % 4 == 0:
            assert tripartite == i3(lambda a: stim_entropy(sim, a), n)
            if use_dense:
                error = abs(tripartite-i3(lambda a: dense.entropy(psi, a), n))
                max_error = max(max_error, error)
                assert error < 2e-10
        records.append([out, prob, int(changed), half, tripartite])
    return dict(records=np.array(records), final=np.array(
        [[p.x, p.z, p.phase] for p in state.g], np.int64),
        max_dense_error=np.array(max_error))


def validate_purification(n, h):
    """I/2^n via Bell references, plus direct density-matrix projectors.

    References are never measured. Only n=3,4 (total 6,8 qubits).
    This tests purification, not a pure-state substitute for mixed entropy.
    """
    if n not in (3, 4):
        raise ValueError('Bounded density-matrix oracle')
    state = Stabilizer(2*n)
    state.g = ([Pauli((1 << j) | (1 << (j+n)), 0) for j in range(n)]
               + [Pauli(0, (1 << j) | (1 << (j+n))) for j in range(n)])
    sim = make_stim(2*n, 'zero')
    for j in range(n):
        sim.h(j)
        sim.cnot(j, j+n)
    psi = np.zeros(2**(2*n), complex)
    for j in range(2**n):
        psi[j | (j << n)] = 2**(-n/2)
    rho = np.eye(2**n, dtype=complex) / 2**n
    records, max_error = [], 0.0
    before = n
    for anchor, labels, u in zip(h['anchors'], h['labels'], h['born']):
        word = local_word(n, int(anchor), ''.join('XYZ'[a] for a in labels))
        M = dense.matrix(word)
        expected = float(np.trace(rho@M).real)
        out, prob, changed = check_step(state, sim, Pauli.word(word+'I'*n), float(u))
        assert out == (1 if u < (1+expected)/2 else -1)
        P = (np.eye(2**n)+out*M)/2
        r = P@rho@P
        pdense = float(np.trace(r).real)
        rho = r/pdense
        psi, pbell = dense.project(psi, dense.matrix(word+'I'*n), out)
        eigenvalues = np.linalg.eigvalsh(rho)
        v = eigenvalues[eigenvalues > 1e-12]
        smix = float(-np.sum(v*np.log2(v)))
        sref = state.entropy(range(n, 2*n))
        checks = [abs(smix-sref), abs(dense.entropy(psi, range(n))-sref),
                  abs(prob-pdense), abs(prob-pbell)]
        max_error = max(max_error, *checks)
        assert max(checks) < 2e-10
        assert sref == stim_entropy(sim, range(n))
        assert sref in (before, before-1)
        before = sref
        records.append([out, prob, int(changed), sref, smix])
    return dict(records=np.array(records), final_mixed_entropy=np.array(before),
                max_dense_error=np.array(max_error))


def symplectic_audit(n, labels):
    words = [local_word(n, j, ''.join(w)) for j in range(n)
             for w in itertools.product(labels, repeat=3)]
    rows = []
    for word in words:
        rows.append([int(a in 'XY') for a in word]+[int(a in 'YZ') for a in word])
    rank = array_rank(rows)
    return dict(L=n, labels=labels, rank=rank, common_pauli_dimension=2*n-rank)

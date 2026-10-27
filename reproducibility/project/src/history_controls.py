"""Gate B controls. Original model and quantum engine are reused unchanged."""
import numpy as np
from src.ensemble import LocalMemory, local_word, LABELS
from src.stabilizer import Pauli, Stabilizer

Q_CASE = {.1: 1, .13: 2, .33: 3}
MODEL_SEED = 2026102407
SHUFFLE_SEED = 2026102408
WORDS = np.array([[a, b, c] for a in range(3) for b in range(3)
                  for c in range(3)], dtype=np.uint8)


def generate_history(L, q, c, rep, sweeps):
    if L not in (12, 16) or q not in Q_CASE or c not in (0, .6):
        raise ValueError("Gate B model is frozen")
    key = [MODEL_SEED, L, Q_CASE[q], rep]
    children = np.random.SeedSequence(key).spawn(5)
    ar, fr, hr, br, ir = [np.random.default_rng(s) for s in children]
    N = L*sweeps
    anchors = ar.integers(L, size=N).astype(np.uint8)
    fresh, hold, born = fr.random((N, 3)), hr.random((N, 3)), br.random(N)
    memory = LocalMemory(L, [q, q, 1-2*q], c, ir.random((L, 3)))
    codes = np.empty(N, np.uint8)
    for i, anchor in enumerate(anchors):
        a = memory.update(int(anchor), fresh[i], hold[i])
        codes[i] = 9*int(a[0])+3*int(a[1])+int(a[2])
    return anchors, codes, born, key


def shuffle_mapping(anchors, L, q, rep, arm, block_sweeps=128):
    """Map each target event to a source event AT THE SAME ANCHOR.

    Whole three-letter words are permuted, never the labels separately.
    A separate independently seeded permutation is used for the block control.
    c is omitted from permutation seeds to pair c=0 and c=.6.
    """
    N = len(anchors)
    mapping = np.arange(N, dtype=np.int32)
    if arm == "original":
        return mapping
    if arm not in ("full", "block"):
        raise ValueError(arm)
    arm_id = 0 if arm == "full" else 1
    rng = np.random.default_rng(
        np.random.SeedSequence([SHUFFLE_SEED, L, Q_CASE[q], rep, arm_id]))
    block = N if arm == "full" else block_sweeps*L
    for start in range(0, N, block):
        end = min(start+block, N)
        for anchor in range(L):
            positions = np.flatnonzero(anchors[start:end] == anchor)+start
            mapping[positions] = rng.permutation(positions)
    assert np.array_equal(anchors[mapping], anchors)
    assert np.array_equal(np.sort(mapping), np.arange(N))
    return mapping


def count_table(anchors, codes, L):
    return np.bincount(anchors.astype(int)*27+codes, minlength=L*27).reshape(L, 27)


def pauli_table(L):
    return [[Pauli.word(local_word(L, j, LABELS[labels]))
             for labels in WORDS] for j in range(L)]


def simulate(anchors, codes, born, L, table=None):
    """Record EVERY event, including state updates that leave entropy fixed."""
    table = pauli_table(L) if table is None else table
    state = Stabilizer(L)
    N = len(anchors)
    outcome = np.empty(N, np.int8)
    changed = np.empty(N, bool)
    entropy = np.zeros(N+1, np.int8)
    current = 0
    for i, (anchor, code) in enumerate(zip(anchors, codes)):
        out, p, active = state.measure(table[int(anchor)][int(code)], float(born[i]))
        outcome[i], changed[i] = out, active
        if active:
            current = state.half_entropy()
        entropy[i+1] = current
    state.check()
    delta = np.diff(entropy.astype(int))
    assert np.all(np.abs(delta) <= 1)
    assert np.all(delta[~changed] == 0)
    return dict(outcome=outcome, changed=changed, entropy=entropy,
                final_generators=np.array([[g.x, g.z, g.phase] for g in state.g],
                                          dtype=np.int32))


def waiting_identity(entropy, changed, lo, hi):
    """Exact event-average / state-segment decomposition on [lo,hi).

    entropy[0] is the initial state; event i has post-state entropy[i+1].
    The first/last segments may be censored by the window boundary.
    Segments are split on state changes, including entropy-neutral changes.
    This is an identity, NOT causal mediation or a stationary-law theorem.
    """
    active = np.flatnonzero(changed[lo:hi])+lo
    starts = np.unique(np.r_[lo, active])
    weights = np.diff(np.r_[starts, hi]).astype(float)
    values = entropy[starts+1].astype(float)
    physical = float(np.mean(entropy[lo+1:hi+1]))
    segment = float(np.mean(values))
    bias = float(np.mean((weights-weights.mean())*(values-values.mean()))
                 / weights.mean())
    assert abs(physical-segment-bias) < 1e-12
    return dict(event_mean=physical, segment_mean=segment, waiting_bias=bias,
                mean_segment_duration=float(weights.mean()),
                identity_error=abs(physical-segment-bias))

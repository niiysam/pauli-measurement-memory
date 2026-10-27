"""Literal model sampling and a separate geometric index enumeration."""
from itertools import product
import numpy as np

LABELS = np.array(list("XYZ"))


def probabilities(pi):
    pi = np.asarray(pi, float)
    if pi.shape != (3,) or min(pi) < 0 or not np.isclose(sum(pi), 1):
        raise ValueError("Expected normalized nonnegative (pX,pY,pZ)")
    return pi


def local_word(L, anchor, labels, boundary="periodic"):
    r = len(labels)
    if r > L or anchor < 0 or anchor >= (L if boundary == "periodic" else L-r+1):
        raise ValueError("Invalid support")
    word = ["I"] * L
    for k, a in enumerate(labels):
        word[(anchor + k) % L] = str(a)
    return "".join(word)


def analytic_index(L, r, pi):
    """PBC, iid anchors, L >= 2r; includes spatially disjoint operators."""
    pi = probabilities(pi)
    if L < 2 * r:
        raise ValueError("Geometric expression requires L >= 2r")
    parity = 2 * float(pi @ pi) - 1
    A = [(1 - parity**ell) / 2 for ell in range(1, r + 1)]
    return (A[-1] + 2 * sum(A[:-1])) / L


def enumerated_index(L, r, pi):
    """Literal independent two-draw sum; translation fixes first anchor at 0.

    This uses onsite unequal-label parity, independently of the bit tableau
    and independently of the analytic overlap expression.
    """
    pi = probabilities(pi)
    words = list(product(range(3), repeat=r))
    weights = [float(np.prod(pi[list(a)])) for a in words]
    total = 0.0
    for a, wa in zip(words, weights):
        for b, wb in zip(words, weights):
            if not wa * wb:
                continue
            for shift in range(L):
                unequal = sum(a[site] != b[(site-shift) % L]
                              for site in range(r) if (site-shift) % L < r)
                total += wa * wb * (unequal % 2) / L
    return total


class LocalMemory:
    def __init__(self, L, pi, c, initial_uniforms, boundary="periodic", r=3):
        self.pi = probabilities(pi)
        if not 0 <= c < 1:
            raise ValueError("Finite-memory gate requires 0 <= c < 1")
        self.c, self.L, self.r, self.boundary = c, L, r, boundary
        self.anchors = L if boundary == "periodic" else L-r+1
        self.labels = np.searchsorted(np.cumsum(self.pi), initial_uniforms, side="right")
        if self.labels.shape != (self.anchors, r):
            raise ValueError("Wrong memory shape")

    def update(self, anchor, fresh_uniforms, hold_uniforms):
        fresh = np.searchsorted(np.cumsum(self.pi), fresh_uniforms, side="right")
        self.labels[anchor] = np.where(hold_uniforms < self.c,
                                      self.labels[anchor], fresh)
        return self.labels[anchor].copy()

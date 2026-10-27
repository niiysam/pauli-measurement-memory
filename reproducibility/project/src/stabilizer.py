"""Pure stabilizers with explicit phases; no third-party quantum simulator.

Convention: P = i**phase X**x Z**z, qubit 0 is the least significant bit.
The algorithm uses L generators, not a copied destabilizer-tableau package.
"""
from dataclasses import dataclass


def rank(rows):
    basis = {}
    for v in rows:
        while v:
            k = v.bit_length() - 1
            if k not in basis:
                basis[k] = v
                break
            v ^= basis[k]
    return len(basis)


@dataclass(frozen=True)
class Pauli:
    x: int
    z: int
    phase: int = 0

    def __mul__(self, other):
        return Pauli(self.x ^ other.x, self.z ^ other.z,
                     (self.phase + other.phase
                      + 2 * (self.z & other.x).bit_count()) % 4)

    def anti(self, other):
        return ((self.x & other.z).bit_count()
                + (self.z & other.x).bit_count()) % 2

    def signed(self, outcome):
        if outcome not in (-1, 1):
            raise ValueError("Outcome must be +/-1")
        return Pauli(self.x, self.z, (self.phase + (outcome < 0) * 2) % 4)

    @classmethod
    def word(cls, word):
        """word[k] acts on qubit k (opposite to printed Kronecker order)."""
        if any(a not in "IXYZ" for a in word):
            raise ValueError(word)
        x = sum((a in "XY") << k for k, a in enumerate(word))
        z = sum((a in "YZ") << k for k, a in enumerate(word))
        return cls(x, z, (x & z).bit_count() % 4)


class Stabilizer:
    def __init__(self, L, initial="zero"):
        if not 1 <= L <= 16:
            raise ValueError("This gate is restricted to 1 <= L <= 16")
        self.L = L
        if initial == "zero":
            self.g = [Pauli(0, 1 << k) for k in range(L)]
        elif initial == "plus":
            self.g = [Pauli(1 << k, 0) for k in range(L)]
        elif initial == "ghz":
            self.g = [Pauli((1 << L) - 1, 0)]
            self.g += [Pauli(0, 1 | (1 << k)) for k in range(1, L)]
        else:
            raise ValueError(initial)

    def check(self):
        mask = (1 << self.L) - 1
        assert len(self.g) == self.L
        assert all((p.x | p.z) & ~mask == 0 for p in self.g)
        assert all((p.phase - (p.x & p.z).bit_count()) % 2 == 0
                   for p in self.g)
        assert all(not a.anti(b) for i, a in enumerate(self.g)
                   for b in self.g[:i])
        assert rank([p.x | (p.z << self.L) for p in self.g]) == self.L

    def eigenvalue(self, measured):
        """Return eigenvalue when measured commutes with the full pure group."""
        basis = {}
        for p in self.g:
            v = p.x | (p.z << self.L)
            while v:
                k = v.bit_length() - 1
                if k not in basis:
                    basis[k] = (v, p)
                    break
                b, h = basis[k]
                v ^= b
                p = p * h
        v = measured.x | (measured.z << self.L)
        p = measured
        while v:
            k = v.bit_length() - 1
            if k not in basis:
                raise ValueError("Not in the maximal commuting group")
            b, h = basis[k]
            v ^= b
            p = p * h
        assert p.x == p.z == 0 and p.phase in (0, 2)
        return 1 if p.phase == 0 else -1

    def measure(self, measured, uniform=0.5, forced=None):
        """Return (outcome, Born probability, changed).

        Random measurements have probability 1/2. Deterministic measurements
        leave the state unchanged. The caller supplies one uniform per event,
        including deterministic events, to make paired streams unambiguous.
        """
        if (measured.phase - (measured.x & measured.z).bit_count()) % 2:
            raise ValueError("Measurement must be Hermitian")
        if (measured.x | measured.z) >> self.L:
            raise ValueError("Measurement outside chain")
        if not 0 <= uniform < 1 or forced not in (None, -1, 1):
            raise ValueError("Invalid outcome control")
        anti = [i for i, p in enumerate(self.g) if p.anti(measured)]
        if not anti:
            outcome = self.eigenvalue(measured)
            if forced is not None and forced != outcome:
                raise ValueError("Impossible forced outcome")
            return outcome, 1.0, False
        outcome = forced if forced is not None else (1 if uniform < 0.5 else -1)
        pivot = anti[0]
        old = self.g[pivot]
        for i in anti[1:]:
            self.g[i] = self.g[i] * old
        self.g[pivot] = measured.signed(outcome)
        return outcome, 0.5, True

    def entropy(self, subset):
        """Pure-trajectory von Neumann entropy, in bits."""
        subset = set(subset)
        if not subset <= set(range(self.L)):
            raise ValueError("Invalid subsystem")
        complement = ((1 << self.L) - 1) ^ sum(1 << k for k in subset)
        rows = [(p.x & complement) | ((p.z & complement) << self.L)
                for p in self.g]
        return rank(rows) - (self.L - len(subset))

    def half_entropy(self):
        return self.entropy(range(self.L // 2))

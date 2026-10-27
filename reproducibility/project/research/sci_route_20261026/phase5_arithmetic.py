"""Bounded exact arithmetic audit; no quantum evolution or graph search."""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "phase5_arithmetic_results.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise FileExistsError("Immutable result already exists; use a new audit version")
    start = time.perf_counter()
    old = json.loads((HERE / "phase4_validation_results.json").read_text())
    protected = dict(old["protected_after"])
    h4 = json.loads((HERE / "phase4_output_hashes.json").read_text())["hashes"]
    protected.update(h4)
    before = {name: digest(ROOT / name) for name in protected}
    assert before == protected, "Prior records differ before audit"
    words = list(product(range(3), repeat=3))
    cases = []
    for q in (F(13, 100), F(1, 3), F(0)):
        pi = (q, q, 1 - 2*q)
        D = 1 - sum(p*p for p in pi)
        A = {ell: (1 - (1 - 2*D)**ell)/2 for ell in (1, 2, 3)}
        weights = {w: pi[w[0]]*pi[w[1]]*pi[w[2]] for w in words}
        same_anchor = sum(weights[u]*weights[v]
                          for u in words for v in words
                          if sum(a != b for a, b in zip(u, v)) % 2)
        assert same_anchor == A[3] == 3*D*(1-D)**2 + D**3
        Jann = A[3] + 2*A[2] + 2*A[1]
        Jque = 2*A[2] + 2*A[1]
        assert Jann - Jque == same_anchor
        assert (same_anchor == 0) == (q == 0)
        cases.append(dict(q=str(q), pi=list(map(str, pi)),
                          D=str(D), J_annealed=str(Jann),
                          E_J_quenched=str(Jque),
                          difference=str(same_anchor),
                          J_annealed_float=float(Jann),
                          E_J_quenched_float=float(Jque),
                          difference_float=float(same_anchor),
                          same_anchor_word_pairs=27**2))
    # For L=16k+s, m=k-1, the enclosure is 8m-4 sites long.
    # Between opposite boxes: L/2-(8m-4)=12+s/2 >=12.
    residues = [{"s": s, "gap_sites": 12+s//2} for s in range(0, 16, 2)]
    assert min(row["gap_sites"] for row in residues) == 12
    after = {name: digest(ROOT / name) for name in protected}
    assert after == before, "Protected inputs changed during audit"
    result = dict(round=5, python=sys.version, exact_rational_cases=cases,
                  periodic_enclosure_residues=residues,
                  elapsed_seconds=time.perf_counter()-start,
                  protected_before=before, protected_after=after,
                  production=0, quantum_state_runs=0, QPU=0,
                  interpretation="Exact ensemble-conditioning arithmetic only; not a phase test or novelty proof")
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items()
                      if not k.startswith("protected_")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

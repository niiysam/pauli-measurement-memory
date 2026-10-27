"""Exact claim-specific fixtures for transient barriers and a purification word."""
from fractions import Fraction as F
from itertools import product
from math import comb
from pathlib import Path
import hashlib
import json
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from src.stabilizer import Pauli, rank
from src.ensemble import local_word
from src.baseline_validation import MixedState
from research.sci_route_20261026.exact_check import selected_branches

OUT = HERE / "phase6_check_results.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def occupancy_case(spec):
    L, m, t = (spec[x] for x in ("L", "m", "t"))
    c, rho = F(spec["c"]), F(spec["rho"])
    refresh = 1-c**3
    cell = 4*refresh/L
    probs = [1-m*cell]+[cell]*m
    assert min(probs) >= 0 and sum(probs) == 1
    total = fail = joint = F(0)
    for initial in product((0, 1), repeat=m):
        w0 = rho**sum(initial)*(1-rho)**(m-sum(initial))
        for cats in product(range(m+1), repeat=t):
            weight = w0
            for cat in cats:
                weight *= probs[cat]
            alive = [initial[j] and (j+1 not in cats) for j in range(m)]
            total += weight
            if not any(alive):
                fail += weight
            if m >= 2 and alive[0] and alive[1]:
                joint += weight
    single = rho*(1-cell)**t
    inclusion = sum((-1)**k*comb(m, k)*rho**k*(1-k*cell)**t
                    for k in range(m+1))
    upper = (1-single)**m
    assert total == 1 and fail == inclusion and fail <= upper
    covariance = joint-single**2 if m >= 2 else None
    if m >= 2:
        assert joint == rho**2*(1-2*cell)**t
        assert covariance <= 0
        if c < 1 and t > 0:
            assert covariance < 0 and fail < upper
    return dict(**spec, per_group_refresh_probability=str(cell),
                protected_cut_probability=str(single),
                all_fail_enumerated=str(fail), inclusion_exclusion=str(inclusion),
                product_upper=str(upper), inequality_slack=str(upper-fail),
                covariance=None if covariance is None else str(covariance),
                elementary_outcomes=2**m*(m+1)**t)


def main():
    if OUT.exists():
        raise FileExistsError("Existing result is immutable; use a new audit version")
    start = time.monotonic()
    plan = json.loads((HERE/"phase6_checks_frozen.json").read_text())
    prior = json.loads((HERE/"phase5_output_hashes.json").read_text())
    protected = dict(prior["protected_immutable_after"])
    protected.update(prior["hashes"])
    before = {n: digest(ROOT/n) for n in protected}
    assert before == protected
    cases = [occupancy_case(spec) for spec in plan["occupancy_fixtures"]]
    clusters = []
    for L in plan["cluster_generators"]["lengths"]:
        for word in plan["cluster_generators"]["words"]:
            ps = [Pauli.word(local_word(L, j, word)) for j in range(L)]
            assert all(not a.anti(b) for a in ps for b in ps)
            r = rank([p.x | p.z << L for p in ps])
            assert r == L
            state = MixedState(L)
            entropy = [state.mixed_entropy]
            for p in ps:
                _, prob, _ = state.measure(p, .25)
                assert prob == .5
                entropy.append(state.mixed_entropy)
            assert entropy == list(range(L, -1, -1))
            clusters.append(dict(L=L, word=word, commuting=True, rank=r,
                                 conditional_entropy_by_event=entropy))
    L = 6
    ps = [Pauli.word(local_word(L, j, "ZXZ")) for j in range(L)]
    stim = selected_branches(ps, list(range(L)), L)
    assert stim["final_entropies"] == [0]
    # Bound comparison is algebra, not a time evolution at these lengths.
    q = F(13, 100)
    pi = (q, q, 1-2*q)
    u2, u3 = sum(p*p for p in pi), sum(p**3 for p in pi)
    rho = u3**2+(u2-u3)**2
    import math
    K = F(1, 10)
    comparison = dict(q=str(q), rho=str(rho), K=str(K),
        asymptotic_surviving_cut_bound=8*math.exp(4*float(K))/float(rho)-4,
        b_sequence="L^-2", sweeps="K*L^2", events="K*L^3",
        expected_refresh_branch_events="K*L",
        universal_crossing_bound="4*K*L^2, capped at L/2",
        global_no_refresh_probability="(1-L^-2)^(K*L^3) ~ exp(-K*L) ->0",
        old_parameter_bound=dict(c="3/5", sweeps=8192,
                                asymptotic_log_surviving_cut=float(math.log(float(rho))-4*(1-(3/5)**3)*8192),
                                verdict="VACUOUS for the old long-time c=.6 observations; no explanatory claim"))
    after = {n: digest(ROOT/n) for n in protected}
    assert after == before
    elapsed = time.monotonic()-start
    assert elapsed < 120
    result = dict(round=6, status="FIXED_EXACT_CHECKS_PASS", plan=plan,
                  occupancy=cases, cluster_generators=clusters,
                  independent_stim_specific_word=stim, comparison=comparison,
                  elapsed_seconds=elapsed, protected_before=before,
                  protected_after=after, new_production=0, QPU=0,
                  quantum_fixture_system_L=6, reference_qubits=6,
                  statistical_samples=0)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items()
                      if k not in ("plan", "protected_before", "protected_after")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

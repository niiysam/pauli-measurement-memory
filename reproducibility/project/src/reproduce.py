"""Bounded, auditable Gate A+ runs. No L>16 trajectory path is provided.

Run from project root: python -m src.reproduce validate|baseline|pilot|analyze
baseline and pilot are intentionally separate commands.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import platform
from pathlib import Path
import sys
import time
import unittest
import numpy as np
import scipy
import matplotlib
from src.stabilizer import Pauli, Stabilizer
from src.ensemble import (LocalMemory, LABELS, local_word, analytic_index,
                          enumerated_index)
from src import reference as dense

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "gate_a_plus"
MASTER = 2026102407
REPS = 16
SWEEPS = 256
CONFIG = dict(master_seed=MASTER, replicates=REPS, sweeps=SWEEPS,
              sizes=[8, 12, 16], q=[0, .10, .13, .33], c=[0, .6], r=3,
              initial="zero", main_boundary="periodic",
              physical_windows=[[129, 192], [193, 256]],
              entropy_units="bits", rng="PCG64 via SeedSequence",
              seed_key="master,L,case,rep; streams anchor,fresh,hold,Born,initial",
              paired="c and initial states share seed key within each case",
              edge_cases="open BC, L=16, pi=(q,1-q,0), q=.1,.274,.5",
              probes="L=16 q=.13 plus/GHZ, c=0,.6; zero in main",
              active_clock="number of nondeterministic state-changing projections",
              hard_limits="L<=16, r=3 in quantum runs, no parameter search")


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def environment():
    return dict(python=sys.version, executable=sys.executable,
                numpy=np.__version__, scipy=scipy.__version__,
                matplotlib=matplotlib.__version__, platform=platform.platform())


def hashes():
    files = list((ROOT / "src").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    files += [ROOT/p for p in ("README.md", "AGENTS.md", "CODEX_TASK.md",
                               "PROJECT_FRAMEWORK.md", "research/GATE_A_REPORT.md",
                               "research/gate_a_sources/Ippoliti2021.pdf",
                               "research/gate_a_sources/Huang2026.pdf")]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def index_checks():
    rows = []
    for L in (6, 8, 12, 16):
        for r in (1, 2, 3):
            for q in (0, .1, .13, .33):
                pi = (q, q, 1-2*q)
                analytic = analytic_index(L, r, pi)
                literal = enumerated_index(L, r, pi)
                rows.append(dict(L=L, r=r, q=q, analytic=analytic,
                                 enumerated=literal, error=abs(analytic-literal)))
    write_csv(OUT/"index_exact.csv", rows)
    # Author-v3 Fig.4 rounded annotations. No L=256 quantum simulation.
    published = []
    for family, points in (("symmetric", [(3, .117, .0087), (4, .083, .0110),
                                         (5, .064, .0134)]),
                           ("edge", [(3, .262, .0087), (4, .176, .0110),
                                     (5, .134, .0134)])):
        for r, qprint, iprint in points:
            D = 1.16/r
            q = ((4-np.sqrt(16-24*D))/12 if family == "symmetric"
                 else (1-np.sqrt(1-2*D))/2)
            pi = (q, q, 1-2*q) if family == "symmetric" else (q, 1-q, 0)
            value = analytic_index(256, r, pi)
            published.append(dict(family=family, r=r, q_printed=qprint,
                                  q_eq6=q, I_printed=iprint, I_PBC256=value,
                                  q_rounding_match=abs(q-qprint)<=.0005,
                                  I_rounding_match=abs(value-iprint)<=.00005,
                                  provenance="Huang author v3 Fig4, Eq6 k=1.16",
                                  scope="Lnorm256 inferred; analytic only"))
    # r=2 annotation is NOT an Eq6 solution. Its threshold is close to an
    # Eq8 extrapolation; that provenance is an inference, not established text.
    published.append(dict(family="symmetric_r2_annotation", r=2, q_printed=.198,
                          q_eq6="", I_printed=.0063,
                          I_PBC256=analytic_index(256, 2, (.198, .198, .604)),
                          q_rounding_match="", I_rounding_match="",
                          provenance="Huang author v3 Fig4 annotation",
                          scope="threshold origin unresolved; close to Eq8 extrapolation"))
    write_csv(OUT/"huang_fig4.csv", published)
    rr = np.arange(3, 9)
    values = []
    for r in rr:
        q = (4-np.sqrt(16-24*1.16/r))/12
        values.append(analytic_index(256, int(r), (q, q, 1-2*q)))
    coef = np.polyfit(rr, values, 1)
    predicted = np.polyval(coef, rr)
    fit = dict(slope=float(coef[0]), intercept=float(coef[1]),
               R2=float(1-np.sum((values-predicted)**2)
                        / np.sum((values-np.mean(values))**2)),
               reported_slope=.0024, reported_intercept=.0015,
               reported_R2=.999989,
               note="Recalculation from inherited empirical Eq6, not new critical data",
               maximum_enumeration_error=max(r["error"] for r in rows),
               six_fig4_rounding_checks_pass=all(
                   r["q_rounding_match"] and r["I_rounding_match"]
                   for r in published[:6]))
    write_csv(OUT/"huang_fig5_analytic.csv",
              [dict(r=int(r), I_PBC256=float(v)) for r, v in zip(rr, values)])
    dump(OUT/"index_validation.json", fit)
    return fit


def validate():
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT/"config.json", CONFIG)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    (OUT/"unit_tests.txt").write_text(stream.getvalue())
    if not result.wasSuccessful():
        raise RuntimeError(stream.getvalue())
    # Metrics independent of assertions in the test suite.
    rng = np.random.default_rng(2026102403)
    max_p_error = max_s_error = 0.
    measurements = 0
    for L in (2, 4, 6, 8):
        for init in ("zero", "plus", "ghz"):
            state, psi = Stabilizer(L, init), dense.initial_state(L, init)
            for _ in range(48):
                word = local_word(L, int(rng.integers(L)),
                                  rng.choice(list("XYZ"), min(3, L)))
                out, p, _ = state.measure(Pauli.word(word), rng.random())
                psi, pp = dense.project(psi, dense.matrix(word), out)
                max_p_error = max(max_p_error, abs(p-pp))
                for cut in (list(range(L//2)), list(range(0, L, 2))):
                    max_s_error = max(max_s_error, abs(state.entropy(cut)
                                                      - dense.entropy(psi, cut)))
                state.check()
                measurements += 1
    born_rng = np.random.default_rng(2026102402)
    count = sum(Stabilizer(1).measure(Pauli.word("X"), born_rng.random())[0] == 1
                for _ in range(2000))
    index = index_checks()
    assert max_p_error < 1e-10 and max_s_error < 1e-10
    assert index["six_fig4_rounding_checks_pass"]
    # Exact global-hold negative control, all raw events saved.
    rng = np.random.default_rng(2026102404)
    a, b = Stabilizer(8), Stabilizer(8)
    control = []
    event = 0
    for reset in range(1, 201):
        word = local_word(8, int(rng.integers(8)), rng.choice(list("XYZ"), 3))
        u = rng.random()
        out, p, changed = a.measure(Pauli.word(word), u)
        bout, bp, bc = b.measure(Pauli.word(word), u)
        assert (out, p, changed) == (bout, bp, bc)
        event += 1
        control.append(dict(event=event, reset=reset, repeated=0, word=word,
                            outcome=out, p=p, S_base=a.half_entropy(),
                            S_held=b.half_entropy()))
        for _ in range(int(rng.geometric(.4))-1):
            bout, bp, bc = b.measure(Pauli.word(word), rng.random())
            assert not bc and bp == 1.
            event += 1
            control.append(dict(event=event, reset=reset, repeated=1, word=word,
                                outcome=bout, p=bp, S_base=a.half_entropy(),
                                S_held=b.half_entropy()))
        assert a.g == b.g
    write_csv(OUT/"global_hold_control.csv", control)
    record = dict(status="PASS", tests=result.testsRun, dense_measurements=measurements,
                  max_Born_error=max_p_error, max_entropy_error=max_s_error,
                  born_trials=2000, born_positive=int(count), born_seed=2026102402,
                  control_resets=200, control_total_events=event,
                  control_identical_states=True, environment=environment(),
                  source_and_code_sha256=hashes(), config=CONFIG)
    dump(OUT/"validation.json", record)
    print(json.dumps({k:v for k,v in record.items()
                      if k not in ("environment", "source_and_code_sha256", "config")},
                     indent=2), flush=True)


def case_definitions():
    cases = []
    for i, q in enumerate(CONFIG["q"]):
        for L in CONFIG["sizes"]:
            cases.append(dict(case=i, family="symmetric", L=L, q=q,
                              pi=[q, q, 1-2*q], initial="zero", boundary="periodic"))
    for i, q in enumerate((.1, .274, .5)):
        cases.append(dict(case=10+i, family="edge", L=16, q=q,
                          pi=[q, 1-q, 0], initial="zero", boundary="open"))
    for init in ("plus", "ghz"):
        cases.append(dict(case=2, family="symmetric", L=16, q=.13,
                          pi=[.13, .13, .74], initial=init, boundary="periodic"))
    return cases


def trajectory(spec, c, rep):
    L = spec["L"]
    assert L <= 16
    key = [MASTER, L, spec["case"], rep]
    children = np.random.SeedSequence(key).spawn(5)
    arng, frng, hrng, brng, irng = [np.random.default_rng(s) for s in children]
    memory = LocalMemory(L, spec["pi"], c, irng.random(
        (L if spec["boundary"] == "periodic" else L-2, 3)), spec["boundary"])
    state = Stabilizer(L, spec["initial"])
    ident = (f'{spec["family"]}_L{L}_q{spec["q"]:g}_c{c:g}_'
             f'{spec["initial"]}_R{rep:02d}')
    path = OUT/"events"/f"{ident}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save all measurement events and Born outcomes; RNG key regenerates proposals.
    samples = [dict(run=ident, event=0, physical_sweeps=0, active=0,
                    active_sweeps=0, entropy=state.half_entropy(), clock="physical")]
    active_samples = [dict(samples[0], clock="active")]
    active = same_global = same_anchor = 0
    label_counts = np.zeros(3, int)
    transitions = np.zeros((3, 3), int)
    previous_at_anchor = {}
    previous_global = None
    hist_anti = hist_pairs = same_with_intervening_active = 0
    last_anchor_active = {}
    with gzip.open(path, "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "anchor", "local_word", "outcome", "Born_probability",
                    "changed", "active_count", "same_global", "same_anchor",
                    "intervening_active"])
        entropy = state.half_entropy()
        for event in range(1, SWEEPS*L+1):
            anchor = int(arng.integers(memory.anchors))
            labels = memory.update(anchor, frng.random(3), hrng.random(3))
            letters = LABELS[labels]
            word = local_word(L, anchor, letters, spec["boundary"])
            current = (anchor, "".join(letters))
            old = previous_at_anchor.get(anchor)
            sg, sa = current == previous_global, old is not None and old[0] == current[1]
            intervening = active - last_anchor_active.get(anchor, active)
            if old is not None:
                old_labels = old[1]
                for aa, bb in zip(old_labels, labels):
                    transitions[aa, bb] += 1
                hist_pairs += 1
                hist_anti += int(np.count_nonzero(old_labels != labels) % 2)
            out, p, changed = state.measure(Pauli.word(word), brng.random())
            active += int(changed)
            same_global += int(sg)
            same_anchor += int(sa)
            same_with_intervening_active += int(sa and intervening > 0 and changed)
            label_counts += np.bincount(labels, minlength=3)
            w.writerow([event, anchor, current[1], out, p, int(changed), active,
                        int(sg), int(sa), intervening])
            previous_global = current
            previous_at_anchor[anchor] = (current[1], labels.copy())
            last_anchor_active[anchor] = active
            if event % L == 0 or (changed and active % L == 0):
                entropy = state.half_entropy()
                sample = dict(run=ident, event=event, physical_sweeps=event/L,
                              active=active, active_sweeps=active/L, entropy=entropy)
                if event % L == 0:
                    samples.append(dict(sample, clock="physical"))
                if changed and active % L == 0:
                    active_samples.append(dict(sample, clock="active"))
        state.check()
    late1 = np.mean([r["entropy"] for r in samples if 129 <= r["physical_sweeps"] <= 192])
    late2 = np.mean([r["entropy"] for r in samples if 193 <= r["physical_sweeps"] <= 256])
    summary = dict(run=ident, **spec, c=c, rep=rep, late1=float(late1), late2=float(late2),
                   drift=float(late2-late1), active_fraction=active/(SWEEPS*L),
                   same_global_fraction=same_global/(SWEEPS*L),
                   same_anchor_fraction=same_anchor/(SWEEPS*L),
                   repeated_anchor_changed_after_intervening=same_with_intervening_active,
                   hist_anti=hist_anti, hist_pairs=hist_pairs,
                   label_counts=label_counts.tolist(), transitions=transitions.tolist(),
                   seed_key=key, spawn_keys=[list(s.spawn_key) for s in children],
                   final_generators=[[p.x, p.z, p.phase] for p in state.g])
    return samples+active_samples, summary


def run(stage):
    if not (OUT/"validation.json").exists():
        raise RuntimeError("Run validate first")
    validation = json.loads((OUT/"validation.json").read_text())
    if validation["source_and_code_sha256"] != hashes():
        raise RuntimeError("Code/source changed since validation; rerun validate")
    if stage == "pilot" and not (OUT/"BASELINE_ACCEPTED.json").exists():
        raise RuntimeError("Inspect baseline and explicitly record bounded acceptance first")
    c = 0 if stage == "baseline" else .6
    cases = [s for s in case_definitions() if c == 0 or s["family"] == "symmetric"]
    samples, summaries = [], []
    start = time.monotonic()
    for spec in cases:
        for rep in range(REPS):
            ss, summary = trajectory(spec, c, rep)
            samples.extend(ss)
            summaries.append(summary)
        print(f'{stage}: L={spec["L"]} q={spec["q"]} {spec["initial"]} '
              f'{spec["boundary"]}, {REPS} trajectories done', flush=True)
    write_csv(OUT/f"{stage}_samples.csv", samples)
    dump(OUT/f"{stage}_trajectories.json", summaries)
    dump(OUT/f"{stage}_manifest.json",
         dict(stage=stage, trajectories=len(summaries),
              events=sum(s["L"]*SWEEPS for s in summaries),
              elapsed_seconds=time.monotonic()-start, config=CONFIG,
              environment=environment(), source_and_code_sha256=hashes()))
    print(f"{stage} complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=["validate", "baseline", "pilot", "analyze"])
    args = parser.parse_args()
    if args.stage == "validate":
        validate()
    elif args.stage == "analyze":
        from src.analyze import analyze
        analyze()
    else:
        run(args.stage)

"""Resource-bounded Gate B. Entry points: audit, run, analyze."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import time
import unittest
from pathlib import Path
import numpy as np
from src.reproduce import ROOT, OUT as OLD, dump, write_csv, environment
from src.ensemble import LABELS
from src.history_controls import (generate_history, shuffle_mapping, count_table,
                                  pauli_table, simulate, WORDS, Q_CASE,
                                  waiting_identity)

OUT = ROOT/"results"/"gate_b"
CONFIG = dict(sizes=[12, 16], q=[.10, .13, .33], c=[0, .6],
              trajectories_per_cell=48, sweeps=512, initial="zero",
              boundary="periodic", r=3, arms=["original", "full", "block"],
              block_sweeps=128, original_seed=2026102407,
              permutation_seed=2026102408, old_reps=list(range(16)),
              confirmation_reps=list(range(16,48)),
              primary_physical_window=[385,512],
              other_physical_windows=[[129,256],[257,384]],
              active_windows=[[65,128],[129,256]],
              primary_statistic="all-event time average within each trajectory window",
              intervals="trajectory-paired t95; primary q=.13 also Bonferroni t "
                        "for 2 sizes x 2 shuffle arms x 2 clocks",
              hard_limits="L<=16, no exponent/critical point fits")


def source_hashes():
    paths = list((ROOT/"src").glob("*.py"))+list((ROOT/"tests").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def old_integrity():
    records = []
    for line in (OLD/"FILE_HASHES.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()
        if actual != expected:
            raise AssertionError(f"Previous artifact changed: {relative}")
        records.append(relative)
    return len(records)


def audit():
    OUT.mkdir(parents=True, exist_ok=True)
    integrity = old_integrity()
    buffer = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT/"tests"))
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    (OUT/"tests.txt").write_text(buffer.getvalue())
    assert result.wasSuccessful(), buffer.getvalue()
    # Recalculate EVERY old per-trajectory window from saved samples,
    # then independently reconstruct paired standard errors.
    summaries = []
    physical = {}
    for stage in ("baseline", "pilot"):
        summaries += json.loads((OLD/f"{stage}_trajectories.json").read_text())
        with (OLD/f"{stage}_samples.csv").open() as f:
            for row in csv.DictReader(f):
                if row["clock"] == "physical":
                    physical.setdefault(row["run"], {})[int(float(row["physical_sweeps"]))] = int(row["entropy"])
    largest = 0.
    for s in summaries:
        values = physical[s["run"]]
        for field, lo, hi in (("late1",129,192),("late2",193,256)):
            m = np.mean([values[i] for i in range(lo,hi+1)])
            largest = max(largest, abs(m-s[field]))
    assert largest < 1e-14
    from scipy.stats import t
    by_id = {(s["family"],s["L"],s["q"],s["c"],s["initial"],s["rep"]):s
             for s in summaries}
    max_paired_error = 0.
    with (OLD/"paired_differences.csv").open() as f:
        for row in csv.DictReader(f):
            diffs = []
            for rep in range(16):
                a = by_id[row["family"],int(row["L"]),float(row["q"]),.6,row["initial"],rep]
                b = by_id[row["family"],int(row["L"]),float(row["q"]),0,row["initial"],rep]
                diffs.append(a["late2"]-b["late2"])
            half = t.ppf(.975,15)*np.std(diffs,ddof=1)/4
            max_paired_error = max(max_paired_error,
                                   abs(np.mean(diffs)-float(row["difference_c06_minus_c0"])),
                                   abs(half-float(row["CI95half"])))
    assert max_paired_error < 1e-12
    dump(OUT/"config.json", CONFIG)
    dump(OUT/"audit.json", dict(status="PASS", old_files_verified=integrity,
                               tests=result.testsRun, old_trajectories=len(summaries),
                               max_old_window_error=largest,
                               max_old_paired_CI_error=max_paired_error,
                               source_sha256=source_hashes(), environment=environment()))
    print(f"Audit PASS: {result.testsRun} tests, {integrity} old files, "
          f"{len(summaries)} old trajectories.", flush=True)


def history_metrics(anchors, codes, L, q, c, anti_table):
    N = len(codes)
    op = anchors.astype(int)*27+codes
    counts = count_table(anchors, codes, L)
    weights = counts.ravel()/N
    metrics = dict(
        empirical_static_index=float(weights@anti_table@weights),
        adjacent_anticommute=float(np.mean(anti_table[op[:-1],op[1:]])),
        global_repeat=float(np.mean(op[:-1]==op[1:])))
    for lag in (1,2,4,8):
        previous, following = [], []
        for j in range(L):
            local = codes[anchors==j]
            previous.extend(local[:-lag])
            following.extend(local[lag:])
        aa, bb = np.array(previous), np.array(following)
        metrics[f"visit_repeat_lag{lag}"] = float(np.mean(aa==bb))
        # Pauli strings at a common anchor: onsite unequal-label parity.
        metrics[f"visit_anti_lag{lag}"] = float(np.mean(
            np.sum(WORDS[aa]!=WORDS[bb],axis=1)%2))
        zz1, zz2 = (WORDS[aa]==2).astype(float), (WORDS[bb]==2).astype(float)
        pz = 1-2*q
        metrics[f"Zcorr_lag{lag}"] = float(np.mean((zz1-pz)*(zz2-pz))/(pz*(1-pz)))
    # Conditional permutation prediction for full shuffle, fixed finite counts.
    expected_pairs = expected_repeat = expected_anti = 0.
    local_anti = anti_table[:27,:27]
    for counts_j in counts:
        n = counts_j.sum()
        if n < 2:
            continue
        expected_pairs += n-1
        expected_repeat += np.sum(counts_j*(counts_j-1))/n
        expected_anti += counts_j@local_anti@counts_j/n
    metrics["full_shuffle_repeat_expectation"] = expected_repeat/expected_pairs
    metrics["full_shuffle_anti_expectation"] = expected_anti/expected_pairs
    return metrics


def trajectory_stats(data, L, codes=None):
    S, changed = data["entropy"], data["changed"]
    delta = np.diff(S.astype(int))
    active_events = np.flatnonzero(changed)
    jump = S[active_events+1]
    metrics = {}
    for lo,hi in [(129,256),(257,384),(385,512)]:
        a,b = (lo-1)*L, hi*L
        prefix = f"p{lo}_{hi}_"
        identity = waiting_identity(S,changed,a,b)
        metrics.update({prefix+k:v for k,v in identity.items()})
        metrics[prefix+"sweep_end_mean"] = float(np.mean(S[np.arange(lo,hi+1)*L]))
        active = int(changed[a:b].sum())
        metrics[prefix+"active_fraction"] = active/(b-a)
        if codes is not None:
            zzz=codes[a:b]==26
            zactive=int(np.sum(zzz & changed[a:b]))
            metrics[prefix+"ZZZ_fraction"] = float(np.mean(zzz))
            metrics[prefix+"ZZZ_given_active"] = zactive/active
            metrics[prefix+"active_given_ZZZ"] = zactive/int(zzz.sum())
            metrics[prefix+"active_given_other"] = (active-zactive)/int((~zzz).sum())
        for sign,label in ((1,"plus"),(-1,"minus"),(0,"zero")):
            number = int(np.count_nonzero((delta[a:b]==sign)&changed[a:b]))
            metrics[prefix+f"{label}_given_active"] = number/active if active else np.nan
        # Same entropy is not the same quantum state; this is only a diagnostic.
        for s in range(L//2+1):
            mask = changed[a:b] & (S[a:b]==s)
            n = mask.sum()
            for sign,label in ((1,"plus"),(-1,"minus")):
                metrics[prefix+f"preS{s}_{label}_count"] = int(np.sum((delta[a:b]==sign)&mask))
            metrics[prefix+f"preS{s}_active_count"] = int(n)
    for lo,hi in [(65,128),(129,256)]:
        a,b = (lo-1)*L,hi*L
        metrics[f"a{lo}_{hi}_n"] = max(0,min(len(jump),b)-a)
        metrics[f"a{lo}_{hi}_mean"] = float(np.mean(jump[a:b])) if len(jump)>=b else None
    metrics["total_active"] = len(jump)
    return metrics


def old_prefix_check(L,q,c,rep,anchors,codes,data):
    if rep>=16:
        return 0
    name=f"symmetric_L{L}_q{q:g}_c{c:g}_zero_R{rep:02d}"
    path=OLD/"events"/f"{name}.csv.gz"
    with gzip.open(path,"rt") as f:
        for i,row in enumerate(csv.DictReader(f)):
            assert int(row["anchor"])==anchors[i]
            assert row["local_word"]=="".join(LABELS[WORDS[codes[i]]])
            assert int(row["outcome"])==data["outcome"][i]
            assert int(row["changed"])==int(data["changed"][i])
            assert float(row["Born_probability"]) == (.5 if data["changed"][i] else 1)
    assert i+1==256*L
    return i+1


def run():
    audit_record=json.loads((OUT/"audit.json").read_text())
    assert audit_record["source_sha256"]==source_hashes(), "Rerun audit after code changes"
    start=time.monotonic()
    (OUT/"raw").mkdir(parents=True,exist_ok=True)
    records=[]
    prefix_events=0
    for L in CONFIG["sizes"]:
        table=pauli_table(L)
        flat=[p for row in table for p in row]
        anti=np.array([[p.anti(other) for other in flat] for p in flat],dtype=float)
        for q in CONFIG["q"]:
            for rep in range(CONFIG["trajectories_per_cell"]):
                for c in CONFIG["c"]:
                    anchors,codes,born,key=generate_history(L,q,c,rep,CONFIG["sweeps"])
                    counts=count_table(anchors,codes,L)
                    ident=f"L{L}_q{q:g}_c{c:g}_R{rep:02d}"
                    raw=dict(anchors=anchors,original_codes=codes,born_uniforms=born,
                             seed_key=np.array(key,dtype=np.int64))
                    static_reference=None
                    for arm in CONFIG["arms"]:
                        mapping=shuffle_mapping(anchors,L,q,rep,arm,CONFIG["block_sweeps"])
                        permuted=codes[mapping]
                        assert np.array_equal(counts,count_table(anchors,permuted,L))
                        if arm=="block":
                            for a in range(0,len(codes),128*L):
                                b=a+128*L
                                assert np.array_equal(count_table(anchors[a:b],codes[a:b],L),
                                                      count_table(anchors[a:b],permuted[a:b],L))
                        data=simulate(anchors,permuted,born,L,table)
                        if arm=="original":
                            prefix_events+=old_prefix_check(L,q,c,rep,anchors,codes,data)
                        hist=history_metrics(anchors,permuted,L,q,c,anti)
                        active_ops=(anchors.astype(int)*27+permuted)[data["changed"]]
                        hist["active_adjacent_anticommute"]=float(np.mean(
                            anti[active_ops[:-1],active_ops[1:]]))
                        if static_reference is None:
                            static_reference=hist["empirical_static_index"]
                        assert hist["empirical_static_index"]==static_reference
                        records.append(dict(run=ident, L=L,q=q,c=c,rep=rep,arm=arm,
                                            **hist,**trajectory_stats(data,L,permuted)))
                        raw[arm+"_source_index"]=mapping
                        raw.update({arm+"_"+name:value for name,value in data.items()})
                    np.savez_compressed(OUT/"raw"/f"{ident}.npz",**raw)
            # Incremental checkpoints; never exclude a parent based on result.
            dump(OUT/"trajectories.json",records)
            print(f"L={L} q={q:g}: 48 parent seeds, both c, 3 arms complete",flush=True)
    dump(OUT/"manifest.json",dict(status="COMPLETE",config=CONFIG,
                                 states_evolved=len(records),
                                 projected_events=sum(r["L"]*512 for r in records),
                                 old_prefix_events_exactly_replayed=prefix_events,
                                 count_and_anchor_constraints="All exact",
                                 elapsed_seconds=time.monotonic()-start,
                                 source_sha256=source_hashes(),environment=environment()))
    print("Gate B bounded simulation complete.",flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=["audit","run","analyze"])
    stage=parser.parse_args().stage
    if stage=="audit":
        audit()
    elif stage=="run":
        run()
    else:
        from src.gate_b_analysis import analyze
        analyze()

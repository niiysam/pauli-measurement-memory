"""Bounded Gate C: audit, originals, analyze, controls (then analyze again)."""
import argparse
import csv
import hashlib
import io
import json
import time
import unittest
import numpy as np
from src.reproduce import ROOT,dump,write_csv,environment
from src.critical_controls import history,simulate_pair
from src.history_controls import shuffle_mapping,count_table

OUT=ROOT/"results"/"gate_c"
OLD=ROOT/"results"/"gate_b"
CONFIG=dict(sizes=[12,16,20,24],q=[.10,.13,.33],c=[0,.6],
            parent_reps_q013=list(range(48,72)),
            parent_reps_controls=list(range(48,64)),
            sweeps=2048,stride_sweeps=4,initials=["zero","plus"],
            boundary="periodic",r=3,
            physical_windows=[[513,1024],[1025,1536],[1537,2048]],
            active_window=[513,1024],
            convergence_tolerance_bits=.15,
            convergence_screen="Per-cell paired t95 intervals for last two windows "
                               "and initial-state differences, for both S and I3, "
                               "must lie wholly inside +/-0.15. Practical screen only.",
            controls_q=[.13],control_arms=["full","block"],control_initial="zero",
            block_sweeps=128,model_seed=2026102407,shuffle_seed=2026102408,
            I3="four consecutive equal quarters, raw unnormalized I3 in bits",
            hard_limits="L<=24; T=2048 fixed; no critical-point/exponent fits")


def hashes():
    paths=list((ROOT/"src").glob("*.py"))+list((ROOT/"tests").glob("*.py"))
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def check_archive():
    files=0
    for archive in (ROOT/"results"/"gate_a_plus",OLD):
        for line in (archive/"FILE_HASHES.sha256").read_text().splitlines():
            expected,rel=line.split("  ",1)
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
            files+=1
    return files


def audit():
    OUT.mkdir(parents=True,exist_ok=True)
    checked=check_archive()
    buf=io.StringIO()
    result=unittest.TextTestRunner(stream=buf,verbosity=2).run(
        unittest.defaultTestLoader.discover(str(ROOT/"tests")))
    (OUT/"tests.txt").write_text(buf.getvalue())
    assert result.wasSuccessful(),buf.getvalue()
    # Independently recalculate Gate B's eight confirmatory contrasts.
    b=json.loads((OLD/"trajectories.json").read_text())
    idx={(r["L"],r["q"],r["c"],r["rep"],r["arm"]):r for r in b}
    from scipy.stats import t
    max_error=0.; checks=0
    with (OLD/"paired_comparisons.csv").open() as f:
        for row in csv.DictReader(f):
            if not (row["q"]=="0.13" and row["cohort"]=="new32"
                    and row["contrast"] in ("full_minus_original_c06","block_minus_original_c06")
                    and row["metric"] in ("p385_512_event_mean","a129_256_mean")):
                continue
            L=int(row["L"]); arm=row["contrast"].split("_")[0]; metric=row["metric"]
            dd=[idx[L,.13,.6,r,arm][metric]-idx[L,.13,.6,r,"original"][metric]
                for r in range(16,48)]
            half=t.ppf(1-.05/16,31)*np.std(dd,ddof=1)/np.sqrt(32)
            max_error=max(max_error,abs(float(np.mean(dd))-float(row["mean"])),
                          abs(half-float(row["half95_family8"])))
            checks+=1
    assert checks==8 and max_error<1e-12
    dump(OUT/"config.json",CONFIG)
    dump(OUT/"audit.json",dict(status="PASS",tests=result.testsRun,
                              archived_hash_entries=checked,GateB_primary_checks=checks,
                              max_GateB_error=max_error,source_sha256=hashes(),
                              environment=environment()))
    print(f"Gate C audit PASS: {result.testsRun} tests; Gate B 8 contrasts verified.",flush=True)


def per_trajectory(data,L,q,c,rep,arm,initials):
    rows=[]
    for k,init in enumerate(initials):
        phy=data[init+"_physical"]; act=data[init+"_active"]
        r=dict(L=L,q=q,c=c,rep=rep,arm=arm,initial=init,
               coalesced_sweeps=float(data["coalesced_event"])/L
               if int(data["coalesced_event"])>=0 else None,
               total_active_sweeps=float(data["changed"][k].sum())/L)
        for lo,hi in CONFIG["physical_windows"]:
            values=phy[(phy[:,0]>=(lo*L))&(phy[:,0]<=hi*L)]
            assert len(values)==(hi-lo+1)//CONFIG["stride_sweeps"]
            for col,name in ((2,"S"),(3,"I3")):
                r[f"p{lo}_{hi}_{name}"]=float(values[:,col].mean())
                # Deterministic grid thinning sensitivity; not extra samples.
                for stride in (8,16):
                    thin=values[values[:,0]%(stride*L)==0,col]
                    r[f"p{lo}_{hi}_{name}_stride{stride}"]=float(thin.mean())
        for lo,hi in ((1025,1280),(1281,1536),(1537,1792),(1793,2048)):
            values=phy[(phy[:,0]>=lo*L)&(phy[:,0]<=hi*L)]
            for col,name in ((2,"S"),(3,"I3")):
                r[f"p{lo}_{hi}_{name}"]=float(values[:,col].mean())
        lo,hi=CONFIG["active_window"]
        values=act[(act[:,1]>=lo*L)&(act[:,1]<=hi*L)]
        complete=data["changed"][k].sum()>=hi*L
        for col,name in ((2,"S"),(3,"I3")):
            r[f"a{lo}_{hi}_{name}"]=float(values[:,col].mean()) if complete else None
        a,b=1536*L,2048*L
        r["late_active_fraction"]=float(data["changed"][k,a:b].mean())
        rows.append(r)
    return rows


def guard():
    record=json.loads((OUT/"audit.json").read_text())
    assert record["source_sha256"]==hashes(),"Rerun audit after source changes"


def run_originals():
    guard()
    start=time.monotonic()
    folder=OUT/"raw"/"original";folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for L in CONFIG["sizes"]:
        for q in CONFIG["q"]:
            reps=CONFIG["parent_reps_q013"] if q==.13 else CONFIG["parent_reps_controls"]
            for rep in reps:
                for c in CONFIG["c"]:
                    anchors,codes,born,key=history(L,q,c,rep,CONFIG["sweeps"])
                    data=simulate_pair(anchors,codes,born,L,stride=CONFIG["stride_sweeps"])
                    np.savez_compressed(folder/f"L{L}_q{q:g}_c{c:g}_R{rep:02d}.npz",
                                        anchors=anchors,original_codes=codes,
                                        born_uniforms=born,seed_key=np.array(key),**data)
                    records+=per_trajectory(data,L,q,c,rep,"original",CONFIG["initials"])
            dump(OUT/"original_trajectories.json",records)
            print(f"Original L={L} q={q:g}, {len(reps)} new parents, both c and product states complete",flush=True)
    dump(OUT/"original_manifest.json",dict(status="COMPLETE",config=CONFIG,
         histories=len(records),events=sum(r["L"]*CONFIG["sweeps"] for r in records),
         elapsed_seconds=time.monotonic()-start,source_sha256=hashes(),environment=environment()))


def run_controls():
    guard()
    assert (OUT/"CONVERGENCE_SCREEN.json").exists(),"Inspect original convergence first"
    start=time.monotonic()
    folder=OUT/"raw"/"controls";folder.mkdir(parents=True,exist_ok=True)
    records=[]
    for L in CONFIG["sizes"]:
        q=.13
        for rep in CONFIG["parent_reps_q013"]:
            for c in CONFIG["c"]:
                name=f"L{L}_q{q:g}_c{c:g}_R{rep:02d}"
                with np.load(OUT/"raw"/"original"/f"{name}.npz") as saved:
                    anchors=saved["anchors"];codes=saved["original_codes"];born=saved["born_uniforms"]
                counts=count_table(anchors,codes,L)
                raw={}
                for arm in CONFIG["control_arms"]:
                    mapping=shuffle_mapping(anchors,L,q,rep,arm,CONFIG["block_sweeps"])
                    words=codes[mapping]
                    assert np.array_equal(anchors[mapping],anchors)
                    assert np.array_equal(count_table(anchors,words,L),counts)
                    if arm=="block":
                        for a in range(0,len(codes),128*L):
                            b=a+128*L
                            assert np.array_equal(count_table(anchors[a:b],codes[a:b],L),
                                                  count_table(anchors[a:b],words[a:b],L))
                    data=simulate_pair(anchors,words,born,L,initials=("zero",),
                                       stride=CONFIG["stride_sweeps"])
                    raw[arm+"_mapping"]=mapping
                    raw.update({arm+"_"+k:v for k,v in data.items()})
                    records+=per_trajectory(data,L,q,c,rep,arm,("zero",))
                np.savez_compressed(folder/f"{name}.npz",**raw)
        dump(OUT/"control_trajectories.json",records)
        print(f"Controls L={L} q=.13, both c, full/block complete",flush=True)
    dump(OUT/"control_manifest.json",dict(status="COMPLETE",histories=len(records),
        events=sum(r["L"]*2048 for r in records),elapsed_seconds=time.monotonic()-start,
        count_and_anchor_constraints="All exact",source_sha256=hashes()))


if __name__=="__main__":
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("stage",choices=["audit","originals","analyze","controls"])
    stage=parser.parse_args().stage
    if stage=="audit":audit()
    elif stage=="originals":run_originals()
    elif stage=="controls":run_controls()
    else:
        from src.gate_c_analysis import analyze
        analyze()

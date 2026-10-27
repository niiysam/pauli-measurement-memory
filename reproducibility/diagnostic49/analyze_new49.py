"""Offline reader for the NEW 49-setting diagnostic only.

No runtime imports or hardware submission. The simultaneous bounded-score
intervals target averages of conditional expected scores at the actual setting
events; they are not Bell-fidelity certificates for a stationary quantum state.
Old eight-block primary analysis is neither imported nor changed.
"""
from pathlib import Path
import json
import math
import sys
from statistics import mean, stdev
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ALPHA = .05
COORDINATES = 37  # 33 science axes plus four paired record-difference axes
FRAMES = {"H0": 3, "H1": 14}

def coordinate(counts, spec, omit=None):
    mask = sum(1 << (5+q) for q in spec["obs"][1])
    if spec.get("h") is not None and not spec["name"].endswith("W2"):
        frame = FRAMES[spec["group"]]
        if omit is not None: frame &= ~(1 << omit)
        mask |= frame
    sign = spec["obs"][2]
    return {text: sign*(1-2*((int(text, 2) & mask).bit_count() % 2))
            for text in counts}

def weighted_mean(counts, values):
    return sum(counts[k]*values[k] for k in counts)/sum(counts.values())

def bounded_interval(value, width, n):
    radius = width*math.sqrt(math.log(2*COORDINATES/ALPHA)/(2*n))
    return dict(mean=value, radius=radius, simultaneous_conditional_mean_interval=[value-radius, value+radius])

def linear_summary(axes, names, weights, constant=0):
    v = constant+sum(w*axes[name]["mean"] for name, w in zip(names, weights))
    radius = sum(abs(w)*axes[name]["radius"] for name, w in zip(names, weights))
    return dict(mean=v, simultaneous_conditional_score_interval=[v-radius, v+radius])

def analyze(payload):
    manifest = json.loads((HERE/"MEASUREMENT_MANIFEST_NOT_AUTHORIZED.json").read_text())
    specs = {x["name"]: x for x in manifest["settings"]}
    blocks = payload["blocks"]
    assert len(blocks) == 4 and sorted(b["block"] for b in blocks) == list(range(4))
    pooled = {name: {} for name in specs}; per_block = []
    for block in sorted(blocks, key=lambda x: x["block"]):
        rows = {row["name"]: row["counts"] for row in block["records"]}
        assert len(rows) == len(block["records"]) == 49 and set(rows) == set(specs)
        for name, counts in rows.items():
            assert sum(counts.values()) == 256
            assert all(isinstance(n, int) and n >= 0 for n in counts.values())
            assert all(len(k) == 9 and set(k) <= {"0", "1"} for k in counts)
            for key, n in counts.items(): pooled[name][key] = pooled[name].get(key, 0)+n
        vals = {}
        for group in ("BELL0", "IDLE5", "ANC5", "P5", "H0", "H1", "SEP0"):
            vals[group] = (1+sum(weighted_mean(rows[f"{group}_W{k}"],
                coordinate(rows[f"{group}_W{k}"], specs[f"{group}_W{k}"])) for k in range(3)))/4
        per_block.append(dict(block=block["block"], nominal_scores=vals))
    axes = {}
    for name, spec in specs.items():
        if "obs" in spec:
            counts = pooled[name]
            axes[name] = bounded_interval(weighted_mean(counts, coordinate(counts, spec)), 2, 1024)
    record = {}
    for group, omit in (("H0", 0), ("H1", 1)):
        for k in range(2):
            name = f"{group}_W{k}"; spec = specs[name]; counts = pooled[name]
            full = coordinate(counts, spec); masked = coordinate(counts, spec, omit)
            difference = {s: full[s]-masked[s] for s in counts}
            label = f"{group}_REC{k}"
            axes[label] = bounded_interval(weighted_mean(counts, difference), 4, 1024)
        record[group] = linear_summary(axes, [f"{group}_REC{k}" for k in range(2)], [.25, .25])
    assert len(axes) == COORDINATES
    groups = {group: linear_summary(axes, [f"{group}_W{k}" for k in range(3)], [.25]*3, .25)
              for group in ("BELL0", "IDLE5", "ANC5", "P5", "H0", "H1", "SEP0")}
    contrasts = {}
    for label, ga, gb in (("storage", "IDLE5", "BELL0"), ("ancilla_MCM", "ANC5", "IDLE5"),
                          ("joint_QND", "P5", "ANC5"), ("history", "H0", "H1")):
        names = [f"{g}_W{k}" for g in (ga, gb) for k in range(3)]
        x = linear_summary(axes, names, [.25]*3+[-.25]*3)
        values = [b["nominal_scores"][ga]-b["nominal_scores"][gb] for b in per_block]
        h = float(t.ppf(.975, 3))*stdev(values)/2
        x["four_block_descriptive_t3_95"] = [mean(values)-h, mean(values)+h]
        contrasts[label] = x
    calibration = {}
    for name, spec in specs.items():
        if spec["kind"] == "calibration":
            counts = pooled[name]
            calibration[name] = dict(empirical_terminal_mismatch=sum(n for s, n in counts.items()
                if int(s, 2) >> 5 != spec["pattern"])/1024)
    return dict(status="COMPLETE_NEW_DIAGNOSTIC_NO_ENTANGLEMENT_CERTIFICATE",
        data_origin=payload["data_origin"], blocks=per_block, axes=axes,
        nominal_groups=groups, paired_contrasts=contrasts, record_contrasts=record,
        calibration_descriptive_only=calibration,
        interval_target="Average conditional expected signed scores at actual setting events; fixed complete counts, no postselection; not stationary-state fidelity",
        assumptions="Bounded observed scores and predictable event labels; conditional expectations refer to past-outcome filtration. Quantum interpretations additionally require trusted preparations/axes/context; not established by these intervals.",
        old8_primary_changed=False, hardware_jobs_submitted=0)

if __name__ == "__main__":
    if len(sys.argv) != 3: raise SystemExit("Usage: analyze_new49.py NEW_INPUT.json NEW_OUTPUT.json")
    out = Path(sys.argv[2])
    with out.open("x") as stream:
        json.dump(analyze(json.loads(Path(sys.argv[1]).read_text())), stream, indent=2)
        stream.write("\n")

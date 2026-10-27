"""Bounded offline preparation. No submitter, account, credentials or network.

Audit the actual previous native instruments, then prepare the frozen 49-case
diagnostic. The independent dense branch engine uses explicit gate matrices.
Cached noise shots are engineering samples, never hardware data or power proof.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import json
import pickle
import socket
import time
from collections import Counter
import numpy as np

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[1]
LAYOUT = [40, 42, 36, 21, 41]
INV = {q: i for i, q in enumerate(LAYOUT)}

def forbidden(*args, **kwargs):
    raise RuntimeError("Network, account and real hardware forbidden.")

socket.create_connection = forbidden
socket.socket.connect = forbidden
import requests
requests.Session.send = forbidden
import qiskit
import qiskit_aer
from qiskit import QuantumCircuit, qpy
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits
QiskitRuntimeService.__init__ = forbidden
sys.path.insert(0, str(PROPOSAL))
import build_validate as old

class Snapshot(BackendV2):
    def __init__(self, target):
        self._cached = target
        super().__init__(name="cached_kingston_only", backend_version="1.0")
    @property
    def target(self): return self._cached
    @property
    def max_circuits(self): return 300
    @classmethod
    def _default_options(cls): return Options()
    run = forbidden

def load(path):
    with path.open("rb") as stream: return qpy.load(stream)

def compact(circuit):
    out = QuantumCircuit(5, 9, name=circuit.name)
    for item in circuit.data:
        qs = [circuit.find_bit(q).index for q in item.qubits]
        cs = [circuit.find_bit(c).index for c in item.clbits]
        assert all(q in INV for q in qs)
        out.append(item.operation, [INV[q] for q in qs], cs)
    return out

IND = np.arange(32)
def one(v, matrix, q):
    low = IND[(IND & (1 << q)) == 0]; high = low | (1 << q)
    out = v.copy()
    out[low] = matrix[0, 0]*v[low] + matrix[0, 1]*v[high]
    out[high] = matrix[1, 0]*v[low] + matrix[1, 1]*v[high]
    return out

def exact(circuit):
    state = np.zeros(32, complex); state[0] = 1
    branches = [(0, state)]
    for item in circuit.data:
        name = item.operation.name
        qs = [circuit.find_bit(q).index for q in item.qubits]
        cs = [circuit.find_bit(c).index for c in item.clbits]
        if name in ("barrier", "delay"): continue  # ideal delays are identities
        if name in ("measure", "reset"):
            new = []; q = qs[0]
            for record, v in branches:
                for bit in (0, 1):
                    vv = v.copy(); vv[((IND >> q) & 1) != bit] = 0
                    if np.vdot(vv, vv).real < 1e-25: continue
                    if name == "reset":
                        if bit: vv = vv[IND ^ (1 << q)]
                        rr = record
                    else:
                        rr = (record & ~(1 << cs[0])) | (bit << cs[0])
                    new.append((rr, vv))
            branches = new; continue
        if name == "cz":
            phase = 1 - 2*((IND >> qs[0]) & 1)*((IND >> qs[1]) & 1)
            branches = [(r, v*phase) for r, v in branches]; continue
        if name == "cx":
            perm = IND ^ (((IND >> qs[0]) & 1) << qs[1])
            branches = [(r, v[perm]) for r, v in branches]; continue
        if name == "rz":
            theta = float(item.operation.params[0])
            matrix = np.diag([np.exp(-.5j*theta), np.exp(.5j*theta)])
        elif name == "sx": matrix = .5*np.array([[1+1j, 1-1j], [1-1j, 1+1j]])
        elif name == "x": matrix = np.array([[0, 1], [1, 0]])
        elif name == "h": matrix = np.array([[1, 1], [1, -1]])/np.sqrt(2)
        elif name == "s": matrix = np.diag([1, 1j])
        elif name == "sdg": matrix = np.diag([1, -1j])
        else: raise ValueError((name, qs))
        branches = [(r, one(v, matrix, qs[0])) for r, v in branches]
    probabilities = {}
    for record, v in branches:
        key = f"{record:09b}"
        probabilities[key] = probabilities.get(key, 0.) + float(np.vdot(v, v).real)
    assert abs(sum(probabilities.values()) - 1) < 1e-11
    return probabilities

def mean(probabilities, spec):
    _, support, sign = spec["obs"]
    mask = sum(1 << (5+q) for q in support)
    if spec.get("h") is not None and not spec["name"].endswith("W2"):
        mask |= sum(x << k for k, x in enumerate(old.COEFF[spec["h"]]))
    return sign*sum(p*(1-2*((int(k, 2) & mask).bit_count() % 2))
                    for k, p in probabilities.items())

def new_specs():
    specs = []
    for group in ("BELL0", "IDLE5", "ANC5", "P5", "H0", "H1"):
        h = int(group[1]) if group in ("H0", "H1") else None
        observables = old.W if h is not None else old.IDLE
        phase = old.PHASE if h is not None else old.W[:2]
        for label, array, target in (("W", observables, 1), ("D", phase, 0)):
            for k, obs in enumerate(array):
                specs.append(dict(name=f"{group}_{label}{k}", group=group,
                                  kind="diagnostic", h=h, obs=obs,
                                  ideal_mean=target, shots=256))
    for k, obs in enumerate(old.IDLE):
        specs.append(dict(name=f"SEP0_W{k}", group="SEP0", kind="diagnostic",
                          h=None, obs=obs, ideal_mean=0, shots=256))
    for pattern in range(16):
        specs.append(dict(name=f"CAL_{pattern:04b}", group="CAL", kind="calibration",
                          pattern=pattern, shots=256))
    assert len(specs) == 49
    return specs

def build(spec, backend, timing):
    if spec["kind"] == "calibration":
        qc = QuantumCircuit(5, 9, name=spec["name"])
        for q in range(4):
            if (spec["pattern"] >> q) & 1: qc.x(q)
        for q in range(4): qc.measure(q, 5+q)
        out = old.native_section(qc, backend); out.name = spec["name"]; return out
    prep = QuantumCircuit(5, 9); old.encoder(prep, entangled=spec["group"] != "SEP0")
    out = QuantumCircuit(backend.num_qubits, 9, name=spec["name"])
    old.append_section(out, old.native_section(prep, backend), timing["prep_dt"], backend)
    if spec["group"] not in ("BELL0", "SEP0"):
        for k in range(5):
            slot = QuantumCircuit(5, 9)
            if spec["group"] == "ANC5": slot.reset(4); slot.measure(4, k)
            elif spec["group"] in ("P5", "H0", "H1"):
                letter = "P" if spec["group"] == "P5" else old.HIST[spec["h"]][k+1]
                old.joint_measurement(slot, [0, 1, 2], 4, old.WORDS[letter], k)
            # An empty idle slot is padded on all five physical qubits.
            section = old.native_section(slot, backend)
            old.append_section(out, section, timing["body_slot_dt"], backend)
    end = QuantumCircuit(5, 9)
    for q, axis in enumerate(spec["obs"][0]): old.read_basis(end, q, axis)
    for q in range(4): end.measure(q, 5+q)
    old.append_section(out, old.native_section(end, backend), timing["terminal_dt"], backend)
    return out

def main():
    start = time.perf_counter()
    assert not (HERE/"PREPARATION_RESULTS.json").exists()
    audit = json.loads((HERE/"NATIVE_AUDIT.json").read_text())
    assert audit["status"] == "PASS_TARGETED_ACTUAL_NATIVE_INSTRUMENT_AUDIT"
    assert len(audit["records"]) == 32
    # Reuse the already-passed targeted audit; do not rerun it.

    target_file = PROPOSAL/"kingston_target_readonly_20261026/export_20261026T194253Z/target_local_only.pickle"
    with target_file.open("rb") as stream: target = pickle.load(stream)
    backend = Snapshot(target); old.LAYOUT = LAYOUT.copy()
    timing = json.loads((PROPOSAL/"kingston_schedule_20261026/COMPILE_RESULTS_V1.json").read_text())["timing"]
    specs = new_specs(); circuits = [build(s, backend, timing) for s in specs]
    validate_isa_circuits(circuits, target)
    ideal = []; resources = []; probabilities = []
    for spec, circuit in zip(specs, circuits):
        p = exact(compact(circuit)); probabilities.append(p)
        if "obs" in spec:
            value = mean(p, spec)
            assert abs(value-spec["ideal_mean"]) < 1e-11, (spec["name"], value)
        else:
            value = p.get(f'{spec["pattern"] << 5:09b}', 0.)
            assert abs(value-1) < 1e-11
        ideal.append(dict(name=spec["name"], exact_ideal_mean_or_cal_probability=value))
        span, _ = old.elapsed(circuit, backend)
        resources.append(dict(name=spec["name"], duration_seconds=span*target.dt,
                              active_qubits=LAYOUT, operations=dict(circuit.count_ops())))
    with (HERE/"DIAGNOSTIC49_NOT_AUTHORIZED.qpy").open("xb") as stream: qpy.dump(circuits, stream)
    (HERE/"EXACT_IDEAL.json").write_text(json.dumps(dict(records=ideal,
        type="Deterministic exact ideal probabilities, not experimental samples"), indent=2)+"\n")

    # Retain only active qubits in the local cached noise model.
    local_target = Target(num_qubits=5, dt=target.dt,
                          qubit_properties=[target.qubit_properties[q] for q in LAYOUT])
    for name in target.operation_names:
        props = {}
        for physical, prop in target[name].items():
            if physical is not None and all(q in INV for q in physical):
                props[tuple(INV[q] for q in physical)] = prop
        if props: local_target.add_instruction(target.operation_from_name(name), props)
    noise = AerSimulator.from_backend(Snapshot(local_target), method="density_matrix",
                                     max_parallel_threads=1, max_memory_mb=256)
    compact_circuits = [compact(c) for c in circuits]
    simulated = noise.run(compact_circuits, shots=256, seed_simulator=2026102702).result()
    assert simulated.success
    noisy = []
    for i, spec in enumerate(specs):
        counts = simulated.get_counts(i)
        p = {k.replace(" ", ""): v/256 for k, v in counts.items()}
        noisy.append(dict(name=spec["name"], synthetic_counts=counts,
                          raw_signed_mean=mean(p, spec) if "obs" in spec else None))
    (HERE/"CACHED_NOISE_ENGINEERING.json").write_text(json.dumps(dict(
        data_origin="LOCAL_CACHED_SNAPSHOT_SIMULATION_NOT_IBM_DATA", seed=2026102702,
        local_shots_per_setting=256, records=noisy,
        omissions="No actual MCM crosstalk/leakage/drift/contextSPAM model; no real-hardware forecast or power guarantee",
        custom_noise_passes=[type(x).__name__ for x in noise.options.noise_model._custom_noise_passes]), indent=2)+"\n")

    rng = np.random.default_rng(2026102701)
    blocks = [dict(block=k, order=[specs[i]["name"] for i in rng.permutation(49)]) for k in range(4)]
    initialization = max(target["reset"][(q,)].duration for q in LAYOUT)
    execution = sum(256*(r["duration_seconds"]+.00025+initialization) for r in resources)
    estimates = {str(n): execution+2*n for n in (1, 4, 13, 49)}
    manifest = dict(status="OFFLINE_PREPARED_NOT_AUTHORIZED", settings=specs, blocks=blocks,
                    resources=resources, layout=LAYOUT, shots_per_setting=256,
                    shots_per_block=49*256, total_shots=4*49*256,
                    rep_delay_seconds=.00025, conservative_INIT_seconds=initialization,
                    modeled_execution_seconds_per_block=execution,
                    modeled_usage_seconds_per_job_by_subjobs=estimates,
                    proposed_jobs=4, proposed_max_execution_seconds_per_job=110,
                    proposed_aggregate_requested_cap_seconds=440,
                    user_reported_budget_seconds=560,
                    actual_billing_not_guaranteed=True, no_refresh_or_account=True,
                    circuit_file="DIAGNOSTIC49_NOT_AUTHORIZED.qpy",
                    target_date="2026-10-26 cached export; real submission requires exact current compatibility check",
                    old_frozen_statistics_changed=False)
    assert max(estimates.values()) < 110
    (HERE/"MEASUREMENT_MANIFEST_NOT_AUTHORIZED.json").write_text(json.dumps(manifest, indent=2)+"\n")
    result = dict(status="PASS_OFFLINE_PREPARATION_ONLY", original_native_cases=32,
                  new_exact_settings=49, local_noise_engineering_shots=49*256,
                  new_production_trajectories=0, QPU_jobs=0, account_connections=0,
                  live_backend_queries=0, quota_queries=0,
                  timing_seconds=time.perf_counter()-start,
                  python=sys.version, qiskit=qiskit.__version__, aer=qiskit_aer.__version__,
                  proposed_total_real_shots=manifest["total_shots"],
                  modeled_usage_per_job=estimates,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (HERE/"PREPARATION_RESULTS.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()

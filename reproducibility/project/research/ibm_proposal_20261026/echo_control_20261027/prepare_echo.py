"""One frozen, offline XX-echo control. Never creates an IBM account or job."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import pickle
import time
import numpy as np

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
spec = importlib.util.spec_from_file_location("frozen49", PROPOSAL/"next_measurements_20261027/prepare_offline_v2.py")
f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
# Import above installs hard network/account guards; its main() is not called.
from qiskit import QuantumCircuit, qpy
from qiskit.transpiler import Target
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.utils.validations import validate_isa_circuits
import qiskit, qiskit_aer, qiskit_ibm_runtime

def save(name, obj):
    with (HERE/name).open("x") as stream:
        json.dump(obj, stream, indent=2); stream.write("\n")

def settings():
    out = []
    for group in ("BELL0", "IDLE_T", "ECHO_T", "P5_T"):
        for label, observables, value in (("W", f.old.IDLE, 1), ("D", f.old.W[:2], 0)):
            for k, obs in enumerate(observables):
                out.append(dict(name=f"{group}_{label}{k}", group=group, kind="science",
                                obs=obs, ideal_mean=value, shots=256))
    for k, obs in enumerate(f.old.IDLE):
        out.append(dict(name=f"SEP0_W{k}", group="SEP0", kind="science", obs=obs,
                        ideal_mean=0, shots=256))
    for pattern in (0, 5, 10, 15):
        out.append(dict(name=f"CAL_{pattern:04b}", group="CAL", kind="calibration",
                        pattern=pattern, shots=256))
    assert len(out) == 27
    return out

def wait(out, ticks):
    for q in f.LAYOUT: out.delay(ticks, q, unit="dt")
    out.barrier(f.LAYOUT)

def build(s, backend, timing, clock):
    if s["kind"] == "calibration":
        qc = QuantumCircuit(5, 9, name=s["name"])
        for q in range(4):
            if (s["pattern"] >> q) & 1: qc.x(q)
        for q in range(4): qc.measure(q, 5+q)
        out = f.old.native_section(qc, backend); out.name = s["name"]; return out
    out = QuantumCircuit(backend.num_qubits, 9, name=s["name"])
    prep = QuantumCircuit(5, 9); f.old.encoder(prep, entangled=s["group"] != "SEP0")
    f.old.append_section(out, f.old.native_section(prep, backend), timing["prep_dt"], backend)
    if s["group"] == "IDLE_T": wait(out, clock["T_dt"])
    elif s["group"] == "ECHO_T":
        pulse = QuantumCircuit(5, 9)
        for q in range(4): pulse.x(q)
        native = f.old.native_section(pulse, backend)
        assert f.old.elapsed(native, backend)[0] == clock["g_dt"]
        wait(out, clock["D_dt"])
        f.old.append_section(out, native, clock["g_dt"], backend)
        wait(out, 2*clock["D_dt"])
        f.old.append_section(out, native, clock["g_dt"], backend)
        wait(out, clock["D_dt"])
    elif s["group"] == "P5_T":
        for k in range(5):
            slot = QuantumCircuit(5, 9)
            f.old.joint_measurement(slot, [0, 1, 2], 4, f.old.WORDS["P"], k)
            f.old.append_section(out, f.old.native_section(slot, backend), timing["body_slot_dt"], backend)
        wait(out, clock["P5_tail_dt"])
    end = QuantumCircuit(5, 9)
    for q, axis in enumerate(s["obs"][0]): f.old.read_basis(end, q, axis)
    for q in range(4): end.measure(q, 5+q)
    f.old.append_section(out, f.old.native_section(end, backend), timing["terminal_dt"], backend)
    return out

def main():
    start = time.perf_counter()
    plan = json.loads((HERE/"FROZEN_PREPARATION_PLAN.json").read_text())
    assert not (HERE/"PREPARATION_RESULTS.json").exists()
    source = PROPOSAL/"kingston_target_readonly_20261026/export_20261026T194253Z/target_local_only.pickle"
    with source.open("rb") as stream: target = pickle.load(stream)
    backend = f.Snapshot(target); f.old.LAYOUT = f.LAYOUT.copy()
    timing = json.loads((PROPOSAL/"kingston_schedule_20261026/COMPILE_RESULTS_V1.json").read_text())["timing"]
    xs = [target["x"][(q,)].duration/target.dt for q in f.LAYOUT[:4]]
    assert all(abs(x-round(x)) < 1e-9 for x in xs)
    assert len(set(round(x) for x in xs)) == 1, "Unequal X durations: STOP, no automatic adaptation."
    g = round(xs[0]); oldbody = 5*timing["body_slot_dt"]
    alignment = math.lcm(target.granularity, target.pulse_alignment, target.acquire_alignment)
    d = alignment*math.ceil((oldbody+target.min_length-2*g)/(4*alignment))
    total = 4*d+2*g; tail = total-oldbody
    assert min(d, 2*d, tail) >= target.min_length
    assert all(t % target.granularity == 0 for t in (d, 2*d, tail))
    clock = dict(dt_seconds=target.dt, g_dt=g, D_dt=d, T_dt=total,
                 P5_original_body_dt=oldbody, P5_tail_dt=tail,
                 alignment=alignment, min_length=target.min_length,
                 echo_sequence_dt=[d,g,2*d,g,d], all_nonancilla_X_simultaneous=True)
    save("EXACT_CLOCK_PLAN.json", clock)
    specs = settings(); circuits = [build(s, backend, timing, clock) for s in specs]
    validate_isa_circuits(circuits, target)
    ideal = []; resources = []
    for s, qc in zip(specs, circuits):
        p = f.exact(f.compact(qc))
        if "obs" in s:
            val = f.mean(p,s); assert abs(val-s["ideal_mean"]) < 1e-11, (s["name"],val)
            if s["group"] == "P5_T": assert all((int(key,2)&31)==0 for key in p)
        else:
            val = p.get(f'{s["pattern"] << 5:09b}',0.); assert abs(val-1) < 1e-11
        ideal.append(dict(name=s["name"], exact_ideal_mean_or_cal_probability=val))
        span,_ = f.old.elapsed(qc,backend)
        resources.append(dict(name=s["name"],duration_dt=span,duration_seconds=span*target.dt,
                              operations=dict(qc.count_ops())))
    with (HERE/"ECHO27_NOT_AUTHORIZED.qpy").open("xb") as stream: qpy.dump(circuits, stream)
    save("EXACT_IDEAL.json", dict(origin="EXACT_IDEAL_PROBABILITIES_NOT_DATA",records=ideal))

    # Only new controls: static local-Z cancellation and a ZZ negative control.
    # Pulse windows are ideal instantaneous here; no finite-pulse noise claim.
    ind = np.arange(16); bits = np.array([1-2*((ind>>j)&1) for j in range(4)])
    angles = np.array([.31,-.47,.62,.23]); phase = angles@bits/2
    full = np.diag(np.exp(-1j*phase)); quarter = np.diag(np.exp(-.25j*phase))
    half = np.diag(np.exp(-.5j*phase)); G = np.eye(16)[ind^15]
    echo = quarter@G@half@G@quarter
    zz = .37*bits[0]*bits[1]/2
    zzfull = np.diag(np.exp(-1j*zz))
    zzquarter = np.diag(np.exp(-.25j*zz)); zzhalf = np.diag(np.exp(-.5j*zz))
    zzecho = zzquarter@G@zzhalf@G@zzquarter
    err = float(np.max(np.abs(echo-np.eye(16))))
    zzerr = float(np.max(np.abs(zzecho-zzfull)))
    assert err < 1e-12 and zzerr < 1e-12
    assert np.max(np.abs(full-np.eye(16))) > .1
    assert np.max(np.abs(zzecho-np.eye(16))) > .1
    save("EXACT_ECHO_CONTROLS.json",dict(status="PASS_IDEAL_STATIC_Z_AND_ZZ_NEGATIVE_CONTROL",
        local_Z_identity_max_error=err, ZZ_survival_max_error=zzerr,
        static_Z_angles=angles.tolist(), ZZ_angle=.37,
        limitations="Instantaneous error-free pulses and fixed Hamiltonians only. Not new theorem, not hardware data; ZZ not cancelled."))

    # Cached-device engineering only; never used for changing the frozen design.
    local = Target(num_qubits=5,dt=target.dt,
                   qubit_properties=[target.qubit_properties[q] for q in f.LAYOUT])
    for name in target.operation_names:
        props = {tuple(f.INV[q] for q in qs):prop for qs,prop in target[name].items()
                 if qs is not None and all(q in f.INV for q in qs)}
        if props: local.add_instruction(target.operation_from_name(name),props)
    sim = AerSimulator.from_backend(f.Snapshot(local),method="density_matrix",
                                   max_parallel_threads=1,max_memory_mb=256)
    run = sim.run([f.compact(qc) for qc in circuits],shots=256,
                  seed_simulator=plan["cached_noise_seed"]).result()
    assert run.success
    noisy = []
    for i,s in enumerate(specs):
        counts=run.get_counts(i); p={k.replace(" ",""):v/256 for k,v in counts.items()}
        noisy.append(dict(name=s["name"],synthetic_counts=counts,
                          raw_signed_mean=f.mean(p,s) if "obs" in s else None))
    save("CACHED_NOISE_ENGINEERING.json",dict(origin="LOCAL_SIMULATION_NOT_IBM_DATA",records=noisy,
        seed=plan["cached_noise_seed"],shots_per_setting=256,
        limitations="Cached Markovian gate/readout/relaxation model; no calibrated quasistatic Z, leakage, crosstalk or drift; not hardware power forecast."))

    rng=np.random.default_rng(plan["block_order_seed"])
    blocks=[dict(block=i,order=[specs[j]["name"] for j in rng.permutation(27)]) for i in range(4)]
    initialization=max(target["reset"][(q,)].duration for q in f.LAYOUT)
    execution=sum(256*(r["duration_seconds"]+.00025+initialization) for r in resources)
    estimates={str(n):execution+2*n for n in (1,4,13,27)}
    assert max(estimates.values()) < 70
    manifest=dict(status="OFFLINE_PREPARED_NOT_AUTHORIZED",settings=specs,blocks=blocks,resources=resources,
        layout=f.LAYOUT,clock=clock,shots_per_setting=256,shots_per_block=6912,total_shots=27648,
        rep_delay_seconds=.00025,conservative_INIT_seconds=initialization,
        modeled_execution_seconds_per_block=execution,modeled_usage_per_job_by_subjobs=estimates,
        proposed_jobs=4,proposed_per_job_cap_seconds=70,proposed_aggregate_requested_cap_seconds=280,
        circuit_file="ECHO27_NOT_AUTHORIZED.qpy",timing=timing,
        target="READONLY_CACHED_20261026; exact used-instruction live compatibility check only after approval",
        runtime_DD=False,runtime_twirling=False,init_qubits=True)
    save("MEASUREMENT_MANIFEST_NOT_AUTHORIZED.json",manifest)
    result=dict(status="PASS_NEW27_OFFLINE_PREPARATION_NOT_AUTHORIZED",new_exact_settings=27,
        local_engineering_shots=6912,new_real_shots=0,new_real_jobs=0,account_connections=0,
        backend_refreshes=0,quota_queries=0,clock=clock,modeled_seconds_per_job=estimates,
        seconds=time.perf_counter()-start,python=sys.version,qiskit=qiskit.__version__,aer=qiskit_aer.__version__,
        runtime=qiskit_ibm_runtime.__version__,numpy=np.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save("PREPARATION_RESULTS.json",result); print(json.dumps(result,indent=2))

if __name__ == "__main__": main()

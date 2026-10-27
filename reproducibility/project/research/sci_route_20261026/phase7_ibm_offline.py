"""Local files and FakeTorino only. No service, credentials, refresh or QPU.

Reuses the already validated joint parity instrument. The512 noise shots are
engineering samples of an old snapshot, not experiments or production data.
"""
from pathlib import Path
from itertools import product
import json,sys,time
import numpy as np
import qiskit,qiskit_aer,qiskit_ibm_runtime
from qiskit import QuantumCircuit,transpile,qpy
from qiskit.quantum_info import Statevector,Operator
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from ibm_offline_parity import joint_measurement,parity_unitary
HERE=Path(__file__).resolve().parent
OUT=HERE/'phase7_ibm_offline'
WORDS=dict(P='XXX',Q='XYX',R='XZX')
HISTORIES=['PQPRQP','PRQPQP']
COEFF=[[0,1,1,1],[1,1,0,0]]

def prepare(qc):
    qc.h(3);qc.cx(3,2)
    qc.h(0);qc.cx(2,1);qc.h(1);qc.h(2)

def decode_bell(qc):
    qc.h(1);qc.h(2);qc.cx(2,1);qc.h(0)
    qc.cx(3,2);qc.h(3)

def circuit(word=None):
    qc=QuantumCircuit(5,7,name=word or 'no_MCM_SPAM')
    prepare(qc)
    if word:
        for j,a in enumerate(word[1:]):
            joint_measurement(qc,[0,1,2],4,WORDS[a],j)
        # Both all-positive branches implement S_log^dagger. Fixed inverse.
        qc.rxx(np.pi/2,0,2)
    decode_bell(qc)
    qc.measure(2,5);qc.measure(3,6)
    return qc

def score(counts,coeff):
    n=sum(counts.values());ok=0
    for text,count in counts.items():
        b=[int(a) for a in text.replace(' ','')[::-1]]
        parity=sum(c*x for c,x in zip(coeff,b[:4]))%2
        if b[5]==0 and b[6]==parity:ok+=count
    return {'successes':ok,'shots':n,'bell_projector_fidelity':ok/n}

def exact_branches():
    qc=QuantumCircuit(5);prepare(qc)
    init=Statevector.from_instruction(qc)
    unitaries={a:Operator(parity_unitary(w)) for a,w in WORDS.items()}
    errors=[]
    for word,coeff in zip(HISTORIES,COEFF):
        for bits in product([0,1],repeat=5):
            sv=init;prob=1.
            for a,b in zip(word[1:],bits):
                sv=sv.evolve(unitaries[a],qargs=[0,1,2,4])
                data=sv.data[16*b:16*(b+1)]
                p=float(np.vdot(data,data).real);prob*=p
                assert abs(p-.5)<1e-12
                # Exactly project ancilla and reset it, no Born sampling.
                sv=Statevector(np.r_[data/np.sqrt(p),np.zeros(16)])
            final=QuantumCircuit(5);final.rxx(np.pi/2,0,2)
            decode_bell(final);sv=sv.evolve(final)
            parity=sum(c*b for c,b in zip(coeff,bits[:4]))%2
            success=sum(abs(amp)**2 for i,amp in enumerate(sv.data)
                        if ((i>>2)&1)==0 and ((i>>3)&1)==parity)
            error=abs(success-1);errors.append(error)
            assert error<1e-12 and abs(prob-1/32)<1e-12
    return dict(branches=len(errors),max_bell_error=max(errors))

def main():
    assert not OUT.exists();OUT.mkdir();start=time.perf_counter()
    exact=exact_branches()
    fake=FakeTorino()  # Cached package snapshot only. Do not call refresh.
    layout=[16,3,5,2,4];seed=2026102607;shots=512
    circuits=[circuit(w) for w in HISTORIES]+[circuit()]
    compiled=[transpile(qc,backend=fake,initial_layout=layout,
                routing_method='basic',optimization_level=1,seed_transpiler=seed)
                for qc in circuits]
    resources=[]
    for qc in compiled:
        active=set();serial=0.;missing=[]
        for ins in qc.data:
            if ins.operation.name=='barrier':continue
            qs=tuple(qc.find_bit(q).index for q in ins.qubits)
            active.update(qs)
            prop=fake.target[ins.operation.name].get(qs)
            if prop is None or prop.duration is None:missing.append([ins.operation.name,list(qs)])
            else:serial+=prop.duration
        assert len(active)<=5,active
        resources.append(dict(name=qc.name,depth=qc.depth(),ops=dict(qc.count_ops()),
                   active_physical_qubits=sorted(active),serial_known_duration_us=serial*1e6,
                   missing_durations=missing,duration_caveat='Sum of cached instruction durations; NOT a schedule or QPU usage estimate'))
    for name,qs in [('logical.qpy',circuits),('compiled_historical.qpy',compiled)]:
        with (OUT/name).open('wb') as f:qpy.dump(qs,f)
    for qc in circuits:(OUT/(qc.name+'.txt')).write_text(str(qc.draw(output='text',fold=120)))
    ideal=AerSimulator(method='density_matrix',max_parallel_threads=1,max_memory_mb=256)
    noise=AerSimulator.from_backend(fake,method='density_matrix',max_parallel_threads=1,
                                   max_memory_mb=256,enable_truncation=True)
    ideal_result=ideal.run([transpile(qc,ideal,seed_transpiler=seed) for qc in circuits],
                           shots=shots,seed_simulator=seed).result()
    noisy_result=noise.run(compiled,shots=shots,seed_simulator=seed).result()
    assert ideal_result.success and noisy_result.success
    records=[]
    for j,qc in enumerate(circuits):
        coeff=COEFF[j] if j<2 else [0]*4
        a=ideal_result.get_counts(j);b=noisy_result.get_counts(j)
        assert score(a,coeff)['successes']==shots
        records.append(dict(name=qc.name,coefficients=coeff,ideal_counts=a,
                            cached_snapshot_counts=b,ideal=score(a,coeff),
                            cached_snapshot_noise=score(b,coeff)))
    summary=dict(status='PASS',engineering_only=True,exact_new_branch_checks=exact,
                 qiskit=qiskit.__version__,aer=qiskit_aer.__version__,runtime=qiskit_ibm_runtime.__version__,
                 historical_snapshot=str(fake.properties().last_update_date),seed=seed,shots_per_circuit=shots,
                 resources=resources,records=records,seconds=time.perf_counter()-start,
                 real_hardware_jobs=0,account_connections=0,production_trajectories=0,
                 warning='Classical cached-snapshot noise only. Not current IBM data, no experimental discovery, no QPU budget estimate.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='records'},indent=2))

if __name__=='__main__':main()

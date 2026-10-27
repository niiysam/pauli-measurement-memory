"""Offline only: exact three-body instrument + historical IBM-target compile.

Never constructs a RuntimeService, refreshes a snapshot, or loads credentials.
This is an engineering check, not an MIPT or novelty experiment.
"""
from pathlib import Path
from itertools import product
from collections import Counter
import json
import hashlib
import sys
import numpy as np
import qiskit
import qiskit_aer
import qiskit_ibm_runtime
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.quantum_info import Operator, Pauli
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/criticality_ibm_20261026/ibm_offline'


def basis(qc, qubits, word, inverse=False):
    for q,a in zip(qubits,word):
        if a=='X':qc.h(q)
        elif a=='Y':
            if inverse:qc.h(q);qc.s(q)
            else:qc.sdg(q);qc.h(q)
        elif a!='Z':raise ValueError(a)


def parity_unitary(word):
    qc=QuantumCircuit(4)
    basis(qc,range(3),word)
    for q in range(3):qc.cx(q,3)
    basis(qc,range(3),word,True)
    return qc


def joint_measurement(qc, data, ancilla, word, record):
    # Only ancilla is reset/measured: three separate data measurements are wrong.
    qc.reset(ancilla)
    basis(qc,data,word)
    for q in data:qc.cx(q,ancilla)
    basis(qc,data,word,True)
    qc.measure(ancilla,record)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    exact=[]
    for word in map(''.join,product('XYZ',repeat=3)):
        U=Operator(parity_unitary(word)).data
        P=Pauli(word[::-1]).to_matrix()  # q0 is rightmost in Qiskit labels.
        errors=[]
        for b,sign in [(0,1),(1,-1)]:
            K=U[8*b:8*(b+1),:8]
            target=(np.eye(8)+sign*P)/2
            errors.append(float(np.max(np.abs(K-target))))
            assert np.allclose(K,target,rtol=0,atol=1e-12)
            assert np.allclose(K@K,K,rtol=0,atol=1e-12)
        exact.append(dict(word=word,K_plus_error=errors[0],K_minus_error=errors[1]))
    # GHZ is a +1 eigenstate of XXX. The joint projector must preserve its ZZI/IZZ.
    qc=QuantumCircuit(4,5)
    qc.h(0);qc.cx(0,1);qc.cx(0,2)
    joint_measurement(qc,[0,1,2],3,'XXX',0)
    joint_measurement(qc,[0,1,2],3,'XXX',1)
    for j in range(3):qc.measure(j,2+j)
    seed=2026102603;shots=1024
    local=AerSimulator(method='density_matrix',max_parallel_threads=1,max_memory_mb=512)
    ideal=local.run(transpile(qc,local,seed_transpiler=seed),shots=shots,seed_simulator=seed).result()
    ideal_counts=ideal.get_counts()
    assert set(ideal_counts)<= {'00000','11100'}
    fake=FakeTorino()  # Loads package data only, never calls refresh/service.
    adjacency={q:set() for q in range(fake.num_qubits)}
    for a,b in fake.coupling_map.get_edges():adjacency[a].add(b);adjacency[b].add(a)
    center=min(q for q in adjacency if len(adjacency[q])>=3)
    layout=sorted(adjacency[center])[:3]+[center]
    compiled=transpile(qc,backend=fake,initial_layout=layout,optimization_level=1,seed_transpiler=seed)
    # Bound memory explicitly; Aer truncates idle physical wires before simulation.
    noisy=AerSimulator.from_backend(fake,method='density_matrix',max_parallel_threads=1,max_memory_mb=512,enable_truncation=True)
    result=noisy.run(compiled,shots=shots,seed_simulator=seed).result()
    assert result.success
    noisy_counts=result.get_counts()
    def statistics(counts):
        # Classical order c4 c3 c2 c1 c0, two ancilla records then three data bits.
        means={k:0. for k in ['ancilla_first_plus','ancilla_second_plus','ZZ01','ZZ12','all_ideal_support']}
        for text,count in counts.items():
            bits=[int(a) for a in text.replace(' ','')[::-1]];p=count/shots
            means['ancilla_first_plus']+=p*(bits[0]==0)
            means['ancilla_second_plus']+=p*(bits[1]==0)
            means['ZZ01']+=p*(-1)**(bits[2]+bits[3])
            means['ZZ12']+=p*(-1)**(bits[3]+bits[4])
            means['all_ideal_support']+=p*(text.replace(' ','') in ('00000','11100'))
        return means
    # Conservative serial sum of listed historical instruction durations, not billing.
    duration=0.;missing=Counter()
    for item in compiled.data:
        name=item.operation.name
        qubits=tuple(compiled.find_bit(q).index for q in item.qubits)
        if name=='barrier':continue
        try:value=fake.target[name][qubits].duration
        except (KeyError,AttributeError):value=None
        if value is None:missing[name]+=1
        else:duration+=float(value)
    for name,circuit in [('joint_measurement_ghz',qc),('fake_torino_compiled',compiled)]:
        with (OUT/(name+'.qpy')).open('wb') as f:qpy.dump(circuit,f)
        (OUT/(name+'.txt')).write_text(str(circuit.draw(output='text',fold=120))+'\n')
    metadata=result.results[0].metadata
    summary=dict(status='PASS',scope='Offline instrument/compile/noise smoke test only; no temporal-memory physics inference.',
        exact_instruments=exact,max_K_error=max(max(x['K_plus_error'],x['K_minus_error']) for x in exact),
        versions=dict(python=sys.version,qiskit=qiskit.__version__,aer=qiskit_aer.__version__,runtime=qiskit_ibm_runtime.__version__),
        seeds=dict(transpiler=seed,ideal_simulator=seed,noisy_simulator=seed),shots_per_local_circuit=shots,
        ideal_counts=ideal_counts,noisy_counts=noisy_counts,ideal_statistics=statistics(ideal_counts),noisy_statistics=statistics(noisy_counts),
        snapshot=dict(name=fake.name,last_update=str(fake.properties().last_update_date),num_qubits=fake.num_qubits,layout=layout),
        compiled=dict(num_qubits=compiled.num_qubits,depth=compiled.depth(),operations=dict(compiled.count_ops()),
            serial_historical_duration_seconds=duration,missing_duration_operations=dict(missing),
            warning='Not actual scheduled duration or QPU usage. Historical snapshot, no live availability/calibration; repetition delay and execution overhead excluded.'),
        local_simulator=dict(method=metadata.get('method'),active_qubits=metadata.get('num_qubits'),max_memory_mb=512),
        qpu_jobs_submitted=0,account_connections=0,service_instantiations=0,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['status','max_K_error','snapshot','compiled','ideal_statistics','noisy_statistics','qpu_jobs_submitted']},indent=2))


if __name__=='__main__':main()

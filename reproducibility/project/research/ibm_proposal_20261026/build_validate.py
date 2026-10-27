"""Offline five-qubit proposal. No service, credentials, live target or QPU.

Reuses the validated genuine joint-Pauli instrument, but validates the new
clean-auxiliary encoder and direct subsystem-Pauli witness independently.
All FakeTorino data are historical package data; never call refresh().
"""
from pathlib import Path
from itertools import product
import hashlib,json,sys,time
import numpy as np
import qiskit,qiskit_aer,qiskit_ibm_runtime
from qiskit import QuantumCircuit,transpile,qpy
from qiskit.quantum_info import Statevector,Operator,Pauli
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import ASAPScheduleAnalysis,PadDelay
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime.fake_provider import FakeTorino
from qiskit_ibm_runtime.options import SamplerOptions

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'research/sci_route_20261026'))
from ibm_offline_parity import joint_measurement,parity_unitary
from phase7_ibm_offline import prepare as old_prepare

WORDS=dict(P='XXX',Q='XYX',R='XZX')
HIST=['PRQPQP','PQPRQP'];COEFF=[[1,1,0,0],[0,1,1,1]]
LAYOUT=[5,16,3,2,4]
# tuple: axes on data0,data1,data2,reference; support; scalar sign
W=[('XZ YX'.replace(' ',''),[0,1,2,3],1),
   ('ZZZY',[1,2,3],-1),('XZXZ',[0,2,3],1)]
IDLE=[('ZZZX',[1,2,3],1),('XZYY',[0,1,2,3],1),W[2]]
PHASE=[('ZZZX',[1,2,3],1),('XZYY',[0,1,2,3],-1)]

def encoder(qc,entangled=True):
    qc.h(3)
    if entangled:qc.cx(3,2)
    qc.h(0)
    # Ancilla starts |0>: clean bridge implements CX(data2,data1).
    qc.cx(2,4);qc.cx(4,1);qc.cx(2,4)
    qc.h(1);qc.h(2)

def read_basis(qc,q,axis):
    if axis=='X':qc.h(q)
    elif axis=='Y':qc.sdg(q);qc.h(q)
    elif axis!='Z':raise ValueError(axis)

def specifications():
    specs=[]
    for h in range(2):
        for k,obs in enumerate(W):
            specs.append(dict(name=f'H{h}_W{k}',kind='main',h=h,obs=obs,hardware_shots_per_block=512))
    for k,obs in enumerate(IDLE):
        specs.append(dict(name=f'IDLE_W{k}',kind='idle',h=None,obs=obs,hardware_shots_per_block=256))
    for k,obs in enumerate(W):
        specs.append(dict(name=f'NEG_W{k}',kind='negative',h=0,obs=obs,hardware_shots_per_block=256))
    for h in range(2):
        for k,obs in enumerate(PHASE):
            specs.append(dict(name=f'H{h}_D{k}',kind='phase',h=h,obs=obs,hardware_shots_per_block=128))
    for x in range(16):
        specs.append(dict(name=f'CAL_{x:04b}',kind='calibration',pattern=x,hardware_shots_per_block=256))
    assert len(specs)==32 and sum(x['hardware_shots_per_block'] for x in specs)*8==73728
    return specs

def logical_circuit(spec):
    qc=QuantumCircuit(5,9,name=spec['name'])
    if spec['kind']=='calibration':
        for q in range(4):
            if (spec['pattern']>>q)&1:qc.x(q)
    else:
        encoder(qc,entangled=spec['kind']!='negative')
        if spec['kind']!='idle':
            for k,a in enumerate(HIST[spec['h']][1:]):
                qc.barrier(range(5));joint_measurement(qc,[0,1,2],4,WORDS[a],k)
        qc.barrier(range(5))
        for q,axis in enumerate(spec['obs'][0]):read_basis(qc,q,axis)
    for q in range(4):qc.measure(q,5+q)
    return qc

def score(counts,spec,mask=None):
    n=sum(counts.values())
    if spec['kind']=='calibration':
        terminal={}
        for key,v in counts.items():
            bits=[int(c) for c in key.replace(' ','')[::-1]]
            x=sum(bits[5+q]<<q for q in range(4))
            terminal[str(x)]=terminal.get(str(x),0)+v
        return dict(shots=n,terminal_counts=terminal)
    axes,support,sgn=spec['obs'];total=0
    for key,v in counts.items():
        bits=[int(c) for c in key.replace(' ','')[::-1]]
        parity=sum(bits[5+q] for q in support)
        if spec['h'] is not None and spec['name'][-1]!='2':
            # W2 is logical Z. All other science observables are transverse.
            use=COEFF[spec['h']].copy()
            if mask is not None:use[mask]=0
            parity+=sum(a*b for a,b in zip(use,bits[:4]))
        total+=v*sgn*(-1)**parity
    return dict(shots=n,score_mean=total/n,plus_count=(n+total)//2)

def exact_new_checks():
    new=QuantumCircuit(5);encoder(new)
    old=QuantumCircuit(5);old_prepare(old)
    error=1-abs(np.vdot(Statevector.from_instruction(new).data,Statevector.from_instruction(old).data))**2
    assert abs(error)<1e-12
    unitaries={a:Operator(parity_unitary(w)) for a,w in WORDS.items()}
    worst=0.;zero_phase=0.;cases=0
    for h in range(2):
        for bits in product([0,1],repeat=5):
            sv=Statevector.from_instruction(new);prob=1
            for a,b in zip(HIST[h][1:],bits):
                sv=sv.evolve(unitaries[a],qargs=[0,1,2,4])
                v=sv.data[16*b:16*(b+1)];pp=np.vdot(v,v).real
                prob*=pp;sv=Statevector(np.r_[v/np.sqrt(pp),np.zeros(16)])
            assert abs(prob-1/32)<1e-12
            parity=sum(c*b for c,b in zip(COEFF[h],bits[:4]))%2
            for k,(axes,support,sgn) in enumerate(W+PHASE):
                paulis=['I']*5
                for q in support:paulis[q]=axes[q]
                value=float(sv.expectation_value(Pauli(''.join(paulis[::-1]))).real)*sgn
                if k!=2:value*=(-1)**parity
                target=1 if k<3 else 0
                worst=max(worst,abs(value-target))
                if k>=3:zero_phase=max(zero_phase,abs(value))
                assert abs(value-target)<1e-12
            cases+=1
    # The full physical logical Pauli algebra, without a code-space assumption.
    XL=Pauli('IZZII').to_matrix()  # not used: explicit strings below avoid endian ambiguity
    def op(chars):return Pauli(''.join(chars[::-1])).to_matrix()
    x=op(['I','Z','Z','I','I']);z=op(['X','I','X','I','I'])
    y=-op(['X','Z','Y','I','I'])
    assert np.max(np.abs(1j*x@z-y))<1e-12
    return dict(clean_encoder_infidelity=float(error),new_direct_witness_branches=cases,
                max_expectation_error=worst,phase_zero_error=zero_phase,
                logical_subsystem_algebra='PASS; no data decode or reference-data gate after encoding')

def op_duration_dt(op,qs,fake):
    if op.name=='barrier':return 0
    if op.name=='delay':
        assert op.unit=='dt';return int(op.params[0])
    prop=fake.target[op.name].get(tuple(qs))
    if prop is None or prop.duration is None:raise ValueError((op.name,qs,'missing duration'))
    raw=prop.duration/fake.dt
    assert abs(raw-round(raw))<1e-5,(op.name,raw)
    return int(round(raw))

def elapsed(qc,fake):
    tq={q:0 for q in LAYOUT};tc={i:0 for i in range(9)}
    for ins in qc.data:
        qs=[qc.find_bit(q).index for q in ins.qubits];cs=[qc.find_bit(c).index for c in ins.clbits]
        start=max([tq[q] for q in qs]+[tc[c] for c in cs]+[0])
        end=start+op_duration_dt(ins.operation,qs,fake)
        for q in qs:tq[q]=end
        for c in cs:tc[c]=end
    return max(tq.values()),tq

def native_section(qc,fake):
    native=transpile(qc,backend=fake,initial_layout=LAYOUT,routing_method='none',
                     optimization_level=1,seed_transpiler=2026102611)
    scheduled=PassManager([ASAPScheduleAnalysis(target=fake.target),PadDelay(target=fake.target)]).run(native)
    out=QuantumCircuit(fake.num_qubits,9)
    for ins in scheduled.data:
        qs=[scheduled.find_bit(q).index for q in ins.qubits]
        cs=[scheduled.find_bit(c).index for c in ins.clbits]
        if any(q not in LAYOUT for q in qs):
            assert ins.operation.name=='delay';continue
        out.append(ins.operation,qs,cs)
    return out

def append_section(dest,section,total_dt,fake):
    span,tq=elapsed(section,fake);assert span<=total_dt
    dest.compose(section,inplace=True)
    for q in LAYOUT:
        if tq[q]<total_dt:dest.delay(total_dt-tq[q],q,unit='dt')
    dest.barrier(LAYOUT)

def historical_compile(specs,fake):
    preps=[]
    for ent in [True,False]:
        qc=QuantumCircuit(5,9);encoder(qc,ent);preps.append(native_section(qc,fake))
    kernels={}
    for a,k in product('PQR',range(5)):
        qc=QuantumCircuit(5,9);joint_measurement(qc,[0,1,2],4,WORDS[a],k)
        kernels[a,k]=native_section(qc,fake)
    ends={}
    for spec in specs:
        if spec['kind']=='calibration':continue
        qc=QuantumCircuit(5,9)
        for q,a in enumerate(spec['obs'][0]):read_basis(qc,q,a)
        for q in range(4):qc.measure(q,5+q)
        ends[spec['name']]=native_section(qc,fake)
    prep_dt=max(elapsed(x,fake)[0] for x in preps)
    slot_dt=max(elapsed(x,fake)[0] for x in kernels.values())+int(round(1e-6/fake.dt))
    end_dt=max(elapsed(x,fake)[0] for x in ends.values())
    compiled=[];resources=[]
    for spec in specs:
        if spec['kind']=='calibration':qc=native_section(logical_circuit(spec),fake)
        else:
            qc=QuantumCircuit(fake.num_qubits,9,name=spec['name'])
            append_section(qc,preps[int(spec['kind']=='negative')],prep_dt,fake)
            for k in range(5):
                if spec['kind']=='idle':section=QuantumCircuit(fake.num_qubits,9)
                else:section=kernels[HIST[spec['h']][k+1],k]
                append_section(qc,section,slot_dt,fake)
            append_section(qc,ends[spec['name']],end_dt,fake)
            assert elapsed(qc,fake)[0]==prep_dt+5*slot_dt+end_dt
        qc.name=spec['name'];compiled.append(qc)
        active=sorted({qc.find_bit(q).index for ins in qc.data for q in ins.qubits})
        assert set(active)<=set(LAYOUT)
        resources.append(dict(name=qc.name,depth=qc.depth(),operations=dict(qc.count_ops()),
           active=active,duration_seconds=elapsed(qc,fake)[0]*fake.dt))
    return compiled,resources,dict(prep_dt=prep_dt,body_slot_dt=slot_dt,terminal_dt=end_dt,
        total_science_seconds=(prep_dt+5*slot_dt+end_dt)*fake.dt,dt=fake.dt,
        guard_seconds=1e-6,scope='Explicit padded historical-target template, NOT current device schedule or billing')

def main():
    out=HERE/'local_validation';out.mkdir(exist_ok=False)
    start=time.perf_counter();specs=specifications();exact=exact_new_checks()
    fake=FakeTorino()
    compiled,res,timing=historical_compile(specs,fake)
    logical=[logical_circuit(s) for s in specs]
    with (out/'logical32.qpy').open('wb') as f:qpy.dump(logical,f)
    with (out/'historical_scheduled32.qpy').open('wb') as f:qpy.dump(compiled,f)
    # The plan intentionally never creates a network primitive or RuntimeService.
    opts=SamplerOptions(max_execution_time=30,execution={'rep_delay':.00025},
       twirling={'enable_gates':False,'enable_measure':False},dynamical_decoupling={'enable':False})
    seed=2026102612;shots=512
    ideal=AerSimulator(method='density_matrix',max_parallel_threads=1,max_memory_mb=512)
    noisy=AerSimulator.from_backend(fake,method='density_matrix',max_parallel_threads=1,
                                   max_memory_mb=512,enable_truncation=True)
    ir=ideal.run(compiled,shots=shots,seed_simulator=seed).result()
    nr=noisy.run(compiled,shots=shots,seed_simulator=seed).result()
    assert ir.success and nr.success
    rows=[]
    for i,s in enumerate(specs):
        ic=ir.get_counts(i);nc=nr.get_counts(i)
        a=score(ic,s);b=score(nc,s)
        if s['kind']=='main' or s['kind']=='idle':assert a['score_mean']==1
        if s['kind']=='calibration':assert a['terminal_counts']=={str(s['pattern']):shots}
        rows.append(dict(spec=s,ideal=a,cached_noise=b,ideal_counts=ic,cached_noise_counts=nc))
    def fidelity(prefix,which):
        rr=[x for x in rows if x['spec']['name'].startswith(prefix+'_W')]
        assert len(rr)==3
        return (1+sum(x[which]['score_mean'] for x in rr))/4
    fids={s:{w:fidelity(s,w) for w in ['ideal','cached_noise']} for s in ['H0','H1','IDLE','NEG']}
    assert abs(fids['NEG']['ideal']-.25)<.06
    order=[]
    for b in range(8):
        order.append(np.random.default_rng(2026102611+b).permutation(32).tolist())
    result=dict(status='PASS_OFFLINE_DESIGN_ONLY',exact=exact,resources=res,timing=timing,
        versions=dict(python=sys.version,qiskit=qiskit.__version__,aer=qiskit_aer.__version__,runtime=qiskit_ibm_runtime.__version__),
        snapshot=dict(name=fake.name,date=str(fake.properties().last_update_date),layout=LAYOUT,
                      rep_delay_default=fake.default_rep_delay),
        noise_scope='Cached calibration gate/readout/thermal model including built-in custom relaxation pass; does not model realistic MCM correlated backaction/leakage/contextSPAM.',
        custom_noise_passes=[type(x).__name__ for x in noisy.options.noise_model._custom_noise_passes],
        local_shots_per_circuit=shots,seed=seed,rows=rows,fidelities=fids,block_orders=order,
        runtime_options_validated_locally=dict(max_execution_time=opts.max_execution_time,rep_delay=opts.execution.rep_delay),
        wall_seconds=time.perf_counter()-start,account_connections=0,live_backend_queries=0,QPU=0,
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','exact','timing','fidelities','custom_noise_passes','wall_seconds','QPU']},indent=2))

if __name__=='__main__':main()

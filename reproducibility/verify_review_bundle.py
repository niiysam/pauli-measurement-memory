"""Portable reader and immutable real-record checks; no quantum simulation."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from collections import Counter
import importlib.util,json,hashlib,importlib.metadata

HERE=Path(__file__).resolve().parent
index=json.loads((HERE/'SOURCE_INDEX.json').read_text())
count=0
for row in index['hardware_files']:
    path=HERE.parent/row['copy']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==row['copy_sha256']
    raw=json.loads(path.read_text())
    for setting in raw['records']:
        assert Counter(setting['bitstrings_c8_to_c0'])==setting['counts']
        assert len(setting['bitstrings_c8_to_c0'])==setting['shots']
        count+=setting['shots']
assert count==151552
checks=[]
for name,reader,inp,out in [('echo27','analyze_echo27.py','ECHO27_READER_INPUT.json','ECHO27_RESULTS.json'),
                           ('diagnostic49','analyze_new49.py','NEW49_READER_INPUT.json','NEW49_RESULTS.json')]:
    path=HERE/name
    spec=importlib.util.spec_from_file_location('portable_'+name,path/reader)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    obtained=module.analyze(json.loads((path/inp).read_text()))
    expected=json.loads((path/out).read_text())
    assert obtained==expected,name
    checks.append(name)
versions={}
for name in ['numpy','scipy','stim','qiskit','qiskit-aer','qiskit-ibm-runtime']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions[name]='notinstalled_in_this_reader_runtime'
result=dict(status='PASS_PORTABLE_RECENT_READERS_AND_ALL16_REAL_RECORD_FILES',rawshots=count,
            matching_frozen_readers=checks,quantum_simulations=0,QPU_jobs=0,versions=versions)
print(json.dumps(result,indent=2))

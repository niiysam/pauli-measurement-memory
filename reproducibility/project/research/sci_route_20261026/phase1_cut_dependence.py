"""Finite exact negative control: adjacent graph cuts are not independent."""
from pathlib import Path
from itertools import product
from fractions import Fraction
import hashlib
import json
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.stabilizer import Pauli
from src.ensemble import local_word
from research.sci_route_20261026.phase1_exact import add,mul,power,value

HERE=Path(__file__).resolve().parent
OUT=HERE/'phase1_cut_dependence_results.json'
if OUT.exists():raise FileExistsError(OUT)
protected={s:hashlib.sha256((HERE/s).read_bytes()).hexdigest()
           for s in ('exact_results.json','phase1_results.json','phase1_frozen.json')}
start=time.monotonic()
counts=[0]*10
for c,e,f,g,h,i,j,k,l in product('XYZ',repeat=9):
    words=[('X','X',c),('X',e,f),(g,h,i),(j,k,'X'),(l,'X','X')]
    ps=[Pauli.word(local_word(9,b,w)) for b,w in enumerate(words)]
    if all(not ps[a].anti(ps[b]) for a,b in ((0,2),(1,2),(1,3),(2,3),(2,4))):
        counts[sum(s!='Z' for s in (c,e,f,g,h,i,j,k,l))]+=1
expected=add(power([1,0,0,2],3),power([0,2,2,2],3))
assert counts==expected,(counts,expected)
records=[]
for q in (Fraction(13,100),Fraction(1,3)):
    pi=(q,q,1-2*q);u2=sum(x*x for x in pi);u3=sum(x**3 for x in pi)
    rho=u3*u3+(u2-u3)**2
    joint=u3**3+(u2-u3)**3
    assert value(counts,q)==joint
    assert joint!=rho*rho
    records.append(dict(q=str(q),joint=str(joint),product_marginals=str(rho*rho),
                        covariance=str(joint-rho*rho)))
after={s:hashlib.sha256((HERE/s).read_bytes()).hexdigest() for s in protected}
assert after==protected
result=dict(status='PASS',plan=json.loads((HERE/'phase1_cut_dependence_frozen.json').read_text()),
            configurations=3**9,coefficients=counts,predicted_coefficients=expected,
            rational_cases=records,elapsed_seconds=time.monotonic()-start,
            protected_before=protected,protected_after=after)
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))

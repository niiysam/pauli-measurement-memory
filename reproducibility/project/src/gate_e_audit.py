"""Gate E existing-data review only; never mutates archived inputs."""
from itertools import combinations
import hashlib
import io
import json
import unittest
import numpy as np
from scipy.interpolate import PchipInterpolator,CubicSpline
from scipy.optimize import linprog
from scipy.stats import t
from src.reproduce import ROOT,dump,write_csv,environment
from src.stabilizer import Pauli,rank
from src.history_controls import pauli_table
from src.gate_d_analysis import difference

OUT=ROOT/'results'/'gate_e'
SIZES=[12,16,20,24]
QS=[.11,.12,.13,.14]


def archives():
    result={}
    for name in ('gate_a_plus','gate_b','gate_c','gate_c2','gate_d'):
        lines=(ROOT/'results'/name/'FILE_HASHES.sha256').read_text().splitlines()
        for line in lines:
            h,rel=line.split('  ',1)
            assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
        result[name]=len(lines)
    return result


def structural(L):
    rows=[sum(1<<((j+k)%L) for k in range(3)) for j in range(L)]
    full=[p.x|(p.z<<L) for row in pauli_table(L) for p in row]
    return dict(L=L,residue=L%3,quarter=L//4,ZZZ_rank=rank(rows),
                ZZZ_kernel=L-rank(rows),full_pauli_span=rank(full),
                pauli_commutant_dimension=2*L-rank(full))


def load_confirm():
    rows=[]
    for path in sorted((ROOT/'results/gate_d/confirm').glob('rows_*.json')):
        rows+=json.loads(path.read_text())
    return {(r['L'],r['q'],r['c'],r['rep']):r for r in rows if r['initial']=='zero'}


def endpoint_interval(a,b,alpha=.05/48):
    return difference(a,b,alpha)


def interpolation_roots(x,y,method):
    if method=='linear':
        return [float(x[i]-y[i]*(x[i+1]-x[i])/(y[i+1]-y[i]))
                for i in range(len(x)-1) if y[i]*y[i+1]<0]
    f=PchipInterpolator(x,y) if method=='pchip' else CubicSpline(x,y,bc_type='natural')
    return [float(r) for r in f.roots(extrapolate=False) if x[0]<r<x[-1]]


def feasible(x,lo,hi,residue=None):
    # Three tiny linear feasibility problems, not critical exponent fits.
    X=np.column_stack([np.ones(len(x)),x])
    if residue is not None:X=np.column_stack([X,residue])
    bounds=[(None,None),(0,None)]+([(None,None)] if residue is not None else [])
    result=linprog(np.zeros(X.shape[1]),A_ub=np.vstack([X,-X]),b_ub=np.r_[hi,-lo],bounds=bounds,method='highs')
    return bool(result.success)


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    hashes=archives(); log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT/'tests')))
    (OUT/'tests_existing.txt').write_text(log.getvalue());assert tests.wasSuccessful()
    structures=[structural(L) for L in [12,16,20,24,28,32]]
    write_csv(OUT/'structure.csv',structures)
    idx=load_confirm()
    def a(L,q,c,w=3,obs='I3'):
        return np.array([idx[L,q,c,r][f'w{w}_{obs}'] for r in range(32)])
    # Resample complete parent records: both c and every window use same
    # parent indices within an (L,q) cell; different L/q cells resample independently.
    B=4000; rng=np.random.default_rng(2026102425)
    boot={}
    for L in SIZES:
        for q in QS:
            draw=rng.integers(32,size=(B,32))
            for c in (0,.6):
                for w in (1,2,3):boot[L,q,c,w]=a(L,q,c,w)[draw].mean(axis=1)
    endpoints=[]; roots=[]; interpolation=[]
    oldscreen=json.loads((ROOT/'results/gate_d/confirm/convergence.json').read_text())['cells']
    good={(r['L'],r['q'],r['c']):r['all_fixed_pass'] for r in oldscreen}
    for c in (0,.6):
        for small,large in combinations(SIZES,2):
            band={}
            for q in QS:
                e=endpoint_interval(a(large,q,c),a(small,q,c))
                band[q]=e
                endpoints.append(dict(c=c,small=small,large=large,q=q,same_residue=small%3==large%3,**e))
            for w in (1,2,3):
                y=np.array([a(large,q,c,w).mean()-a(small,q,c,w).mean() for q in QS])
                for method in ('linear','pchip','natural_cubic'):
                    rr=interpolation_roots(np.array(QS),y,method)
                    interpolation.append(dict(c=c,small=small,large=large,window=w,method=method,
                        diagnostic_roots=rr,valid_crossings=False,
                        note='Interpolation alone never certifies a crossing'))
                for lo,hi in zip(QS[:-1],QS[1:]):
                    x0=boot[large,lo,c,w]-boot[small,lo,c,w]
                    x1=boot[large,hi,c,w]-boot[small,hi,c,w]
                    crosses=x0*x1<0
                    # Missing roots remain missing; no conditional root CI.
                    opposite=(band[lo]['lower']>0 and band[hi]['upper']<0) or (band[lo]['upper']<0 and band[hi]['lower']>0)
                    converge=all(good[L,q,c] for L in (small,large) for q in (lo,hi))
                    roots.append(dict(c=c,small=small,large=large,window=w,qlo=lo,qhi=hi,
                        bootstrap_root_fraction=float(crosses.mean()),endpoint_family48=bool(opposite),
                        converged_endpoints=converge,valid_crossing=bool(w==3 and opposite and converge)))
    write_csv(OUT/'old_endpoint_recheck.csv',endpoints)
    write_csv(OUT/'hierarchical_bootstrap.csv',roots)
    dump(OUT/'interpolation_sensitivity.json',interpolation)
    # Confidence-band feasibility of exact finite-grid forms only.
    laws=[]
    for q in QS:
        for c in (0,.6):
            values=[a(L,q,c,obs='S') for L in SIZES]
            means=np.array([v.mean() for v in values])
            half=np.array([t.isf(.05/(2*4),31)*v.std(ddof=1)/np.sqrt(32) for v in values])
            lower,upper=means-half,means+half
            all_good=all(good[L,q,c] for L in SIZES)
            constant=bool(max(lower)<=min(upper))
            laws.append(dict(q=q,c=c,law='constant',residue_term=False,compatible=constant,all_converged=all_good))
            for law,x in (('log',np.log2(SIZES)),('linear',np.array(SIZES))):
                for residue in (False,True):
                    ok=feasible(x,lower,upper,np.array([int(L%3==0) for L in SIZES]) if residue else None)
                    laws.append(dict(q=q,c=c,law=law,residue_term=residue,compatible=ok,all_converged=all_good))
    write_csv(OUT/'old_finite_grid_laws.csv',laws)
    dump(OUT/'existing_data_review.json',dict(status='COMPLETE',archived_entries=hashes,tests=tests.testsRun,
        bootstrap_replicates=B,bootstrap_seed=2026102425,parent_unit='independent trajectory, paired c and windows',
        confirmed_crossings=sum(r['valid_crossing'] for r in roots),structures=structures,
        same_residue_pairs=[[12,24]],
        decision='A same-residue 16/28 and 20/32 comparison at q=.12,.13,.14 targets an identified confound. It cannot retroactively repair old convergence failures or certify a thermodynamic transition.',
        environment=environment()))
    print(json.dumps(dict(tests=tests.testsRun,confirmed_crossings=sum(r['valid_crossing'] for r in roots),
                         same_residue_pairs=[[12,24]],structure=structures),indent=2))


if __name__=='__main__':run()

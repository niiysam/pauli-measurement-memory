"""Bounded exact line-graph coverage classification of saved capacity records.

No stabilizer trajectories are run; all capacities are read from round2.
Clique partitions and root incidence are implemented here independently.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import json
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'phase3_graph_results.json'


def edges_of(omega):
    return [(i,j) for i in range(len(omega)) for j in range(i+1,len(omega))
            if omega[i]>>j&1]


def binary_rank(rows):
    pivots={}
    for v in rows:
        while v:
            bit=v.bit_length()-1
            if bit in pivots:v^=pivots[bit]
            else:pivots[bit]=v;break
    return len(pivots)


def root_of(n,partition):
    memberships=[[i for i,c in enumerate(partition) if v in c] for v in range(n)]
    next_vertex=len(partition)
    root=[]
    for memberships_v in memberships:
        endpoints=list(memberships_v)
        while len(endpoints)<2:
            endpoints.append(next_vertex);next_vertex+=1
        assert len(endpoints)==2
        root.append(tuple(endpoints))
    assert len(set(tuple(sorted(e)) for e in root))==n
    cols=[(1<<a)|(1<<b) for a,b in root]
    return dict(root_edges=root,root_vertices=next_vertex,
                incidence_rank=binary_rank(cols),independent=binary_rank(cols)==n)


def roots(omega):
    n=len(omega);edges=edges_of(omega);edge_index={e:k for k,e in enumerate(edges)}
    cliques=[]
    for k in range(2,n+1):
        for c in combinations(range(n),k):
            ce=list(combinations(c,2))
            if all(e in edge_index for e in ce):
                cliques.append((c,sum(1<<edge_index[e] for e in ce)))
    covers=[];visited=0
    def visit(remaining,counts,part):
        nonlocal visited
        visited+=1
        if visited>20000:raise RuntimeError('Frozen partition-search budget')
        if not remaining:
            r=root_of(n,part)
            # Direct validation of the constructed root against target graph.
            observed=[(i,j) for i,j in combinations(range(n),2)
                      if set(r['root_edges'][i])&set(r['root_edges'][j])]
            assert observed==edges
            covers.append(dict(cliques=part,**r));return
        first=remaining&-remaining
        for clique,mask in cliques:
            if mask&first and mask&remaining==mask and all(counts[v]<2 for v in clique):
                nxt=list(counts)
                for v in clique:nxt[v]+=1
                visit(remaining^mask,nxt,part+[clique])
    visit((1<<len(edges))-1,[0]*n,[])
    forest=next((c for c in covers if c['independent']),None)
    return dict(is_line_graph=bool(covers),independent_bilinears=forest is not None,
                root_count=len(covers),search_nodes=visited,
                root_witness=forest or (covers[0] if covers else None))


def claw(omega):
    n=len(omega)
    for center in range(n):
        neighbors=[i for i in range(n) if omega[center]>>i&1]
        for leaves in combinations(neighbors,3):
            if all(not(omega[i]>>j&1) for i,j in combinations(leaves,2)):
                return [center,*leaves]
    return None


def matrix(n,edges):
    rows=[0]*n
    for a,b in edges:rows[a]|=1<<b;rows[b]|=1<<a
    return rows


def main():
    if OUT.exists():raise FileExistsError(OUT)
    start=time.monotonic()
    controls={
        'triangle':roots(matrix(3,[(0,1),(0,2),(1,2)])),
        'claw':roots(matrix(4,[(0,1),(1,2),(1,3)])),
        'square':roots(matrix(4,[(0,1),(0,2),(1,3),(2,3)]))}
    assert controls['triangle']['independent_bilinears']
    assert not controls['claw']['is_line_graph']
    assert controls['square']['is_line_graph'] and not controls['square']['independent_bilinears']
    old=json.loads((HERE/'phase2_results.json').read_text())
    saved=sorted(old['primary_records'],key=lambda x:(x['n'],x['edge_bit_code']))
    protected=list(json.loads((HERE/'phase2_counterexample_results.json').read_text())['protected_before'])
    protected += [str((HERE/f).relative_to(ROOT)) for f in
        ('phase2_counterexample.py','phase2_counterexample_frozen.json','phase2_counterexample_results.json')]
    before={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in protected}
    records=[]
    for rec in saved:
        result=dict(n=rec['n'],edge_bit_code=rec['edge_bit_code'],omega=rec['omega'],
                    delta=rec['delta'],a=rec['a'],rstar=rec['rstar'],
                    stored_word=rec['word'],**roots(rec['omega']),induced_claw=claw(rec['omega']))
        if result['induced_claw'] is not None:assert not result['is_line_graph']
        records.append(result)
        if time.monotonic()-start>30:raise TimeoutError('Frozen classification time cap')
    positive=[r for r in records if r['delta']>0]
    strict=[r for r in positive if not r['is_line_graph']]
    weak=[r for r in positive if not r['independent_bilinears']]
    selected=(strict or weak or [None])[0]
    counts={name:dict(Counter((r['is_line_graph'],r['independent_bilinears']) for r in seq))
            for name,seq in [('all',records),('delta_positive',positive)]}
    counts={name:{str(k):v for k,v in tally.items()} for name,tally in counts.items()}
    after={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in before}
    assert before==after
    output=dict(status='ALGEBRAIC_OBSTRUCTION_SELECTED' if selected else 'NO_OBSTRUCTION_IN_FROZEN_SET',
        plan=json.loads((HERE/'phase3_frozen.json').read_text()),controls=controls,counts=counts,
        selected=selected,records=records,protected_before=before,protected_after=after,
        elapsed_seconds=time.monotonic()-start,new_production=0,QPU_jobs=0)
    OUT.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({k:output[k] for k in ('status','counts','selected','elapsed_seconds')},indent=2))


if __name__=='__main__':main()

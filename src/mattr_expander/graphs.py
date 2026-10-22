"""Deterministic degree-matched graphs; sparse spectral diagnostics only."""
import hashlib
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
import networkx as nx


def ring(n,degree=8):
    if degree%2 or n<=degree:raise ValueError('Even degree smaller than N required.')
    offsets=np.r_[np.arange(1,degree//2+1),-np.arange(1,degree//2+1)]
    return np.sort((np.arange(n)[:,None]+offsets[None,:])%n,axis=1).astype(np.int32)


def canonical(edges,n,degree):
    rows=[[] for _ in range(n)]
    for a,b in edges:
        rows[a].append(b);rows[b].append(a)
    if any(len(r)!=degree for r in rows):raise ValueError('Degree mismatch.')
    return np.array([sorted(r) for r in rows],dtype=np.int32)


def generate(family,n=32491,degree=8,seed=1701):
    start=time.perf_counter();attempts=0;swaps=0
    if family=='G0':neighbors=ring(n,degree)
    elif family=='G1':
        neighbors=ring(n,degree)
        edges=[(a,int(b)) for a,row in enumerate(neighbors) for b in row if a<b]
        adjacency=[set(map(int,row)) for row in neighbors]
        rng=np.random.default_rng(seed)
        budget=len(edges)//4
        while swaps<budget and attempts<100*budget:
            attempts+=1;i,j=map(int,rng.integers(len(edges),size=2))
            if i==j:continue
            a,b=edges[i];c,d=edges[j]
            if len({a,b,c,d})!=4:continue
            if rng.integers(2):c,d=d,c
            if c in adjacency[a] or d in adjacency[b]:continue
            for x,y in ((a,b),(c,d)):adjacency[x].remove(y);adjacency[y].remove(x)
            for x,y in ((a,c),(b,d)):adjacency[x].add(y);adjacency[y].add(x)
            edges[i]=tuple(sorted((a,c)));edges[j]=tuple(sorted((b,d)));swaps+=1
        if swaps!=budget:raise ValueError('Registered edge-swap budget not completed.')
        neighbors=np.array([sorted(row) for row in adjacency],dtype=np.int32)
    elif family=='G2':
        graph=nx.random_regular_graph(degree,n,seed=seed)
        neighbors=canonical(graph.edges(),n,degree)
    else:raise ValueError('Unknown registered graph family.')
    return neighbors,{'family':family,'graph_seed':seed,'generation_seconds':time.perf_counter()-start,
                      'successful_swaps':swaps,'swap_attempts':attempts}


def adjacency(neighbors):
    n,d=neighbors.shape
    return sp.csr_matrix((np.ones(n*d,dtype=np.float64),neighbors.ravel(),np.arange(0,n*d+1,d)),shape=(n,n))


def audit(neighbors):
    n,d=neighbors.shape
    if neighbors.min()<0 or neighbors.max()>=n:raise ValueError('Invalid vertex.')
    if np.any(neighbors==np.arange(n)[:,None]) or np.any(np.diff(neighbors,axis=1)<=0):
        raise ValueError('Self-loop, duplicate or unsorted neighbors.')
    a=adjacency(neighbors)
    if (a!=a.T).nnz:raise ValueError('Graph is not undirected.')
    components=int(connected_components(a,directed=False,return_labels=False))
    if components!=1:raise ValueError('Registered graph is disconnected; no replacement seed permitted.')
    return {'vertices':n,'edges':n*d//2,'degree_min':d,'degree_max':d,'components':components,
            'degree_histogram':{str(d):n},'csr_bytes':int(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes),
            'neighbor_int32_bytes':int(neighbors.nbytes),'gpu_neighbor_int64_bytes':int(n*d*8),
            'adjacency_sha256':hashlib.sha256(np.asarray(neighbors,dtype='<i4').tobytes()).hexdigest()}


def spectrum(neighbors,family):
    start=time.perf_counter();n,d=neighbors.shape
    if family=='G0':
        frequency=np.arange(n,dtype=np.float64)
        values=sum(2*np.cos(2*np.pi*k*frequency/n) for k in range(1,d//2+1))/d
        gap=float(1-np.sort(values)[-2]);magnitude=float(np.max(np.abs(values[1:])))
        residual=0.;method='analytic real circulant spectrum (all N frequencies)'
    else:
        p=adjacency(neighbors)/d
        # Fixed local start vector does not consume any training RNG.
        v0=np.random.default_rng(1701).normal(size=n)
        vals,vecs=eigsh(p,k=3,which='LA',tol=1e-7,maxiter=6000,ncv=40,v0=v0)
        low,lowvec=eigsh(p,k=1,which='SA',tol=1e-7,maxiter=6000,ncv=40,v0=v0)
        gap=float(1-np.sort(vals)[-2]);magnitude=float(max(abs(np.sort(vals)[-2]),abs(low[0])))
        residual=float(max(np.linalg.norm(p@vecs-vecs*vals,axis=0).max(),np.linalg.norm(p@lowvec-lowvec*low,axis=0).max()))
        if abs(vals.max()-1)>1e-5 or residual>1e-5:raise ValueError('Spectral residual check failed.')
        method='sparse symmetric ARPACK; LA k=3, SA k=1, tol1e-7, maxiter6000, ncv40'
    return {'laplacian_gap':gap,'nontrivial_adjacency_magnitude':magnitude,'spectral_residual':residual,
            'spectral_seconds':time.perf_counter()-start,'spectral_method':method}


def permute(neighbors,seed=2701):
    """Relabel auxiliary vertices only; never reorder the real MAttr score vector."""
    n=len(neighbors);permutation=np.random.default_rng(seed).permutation(n)
    result=np.empty_like(neighbors)
    result[permutation]=permutation[neighbors]
    return np.sort(result,axis=1)

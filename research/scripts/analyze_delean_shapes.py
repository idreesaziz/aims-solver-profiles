"""Frozen-data shape diagnostics. No official inference API is modified."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,time,hashlib,csv,importlib.metadata
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.delean_shape import (fit_rasch,graph_association,family_center,make_strata,
                                permute_strata,persistence_statistics)


def save(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(c,out):
    base=ROOT/c['source'];x=np.load(base/'features.npy');y=np.load(base/'outcomes.npy')
    g=np.load(base/'groups.npy');f=np.load(base/'families.npy')
    group_rows={}
    for i,group in enumerate(g):group_rows.setdefault(str(group),[]).append(i)
    keep=np.array([ids[0] for ids in group_rows.values() if len(set(f[ids]))==1])
    x,y,f,g=x[keep],y[keep],f[keep],g[keep]
    for name,data in [('x',x),('y',y),('families',f),('groups',g),('snapshot_indices',keep)]:
        np.save(out/(name+'.npy'),data,allow_pickle=False)
    prepared=json.loads((base/'prepared.json').read_text())
    provenance=dict(config=c,questions=len(x),models=y.shape[1],pilot_models=prepared['pilot_model_columns'],
        grouping='First source-order representative per underlying stem; cross-benchmark stem groups excluded entirely.',
        versions={z:importlib.metadata.version(z) for z in ['numpy','scipy','scikit-learn','ripser']},
        hashes={str(p.relative_to(ROOT)):digest(p) for p in [base/'features.npy',base/'outcomes.npy',base/'groups.npy',base/'families.npy',Path(__file__),ROOT/'models/delean_shape.py',ROOT/'config_delean_shape_v1.json']})
    save(out/'prepared.json',provenance)
    return x,y,f,g,provenance


def nearest(x,metric,maxk,out):
    target=out/('neighbors_'+metric+'.npz')
    if target.exists():
        with np.load(target) as data:return data['indices'],data['distances']
    indices=np.empty((len(x),maxk),dtype=np.int32);distances=np.empty((len(x),maxk),dtype=np.float32)
    for start in range(0,len(x),64):
        ids=np.arange(start,min(start+64,len(x)))
        d=cdist(x[ids],x,metric);d[np.arange(len(ids)),ids]=np.inf
        nn=np.argpartition(d,maxk-1,axis=1)[:,:maxk]
        order=np.argsort(np.take_along_axis(d,nn,axis=1),axis=1,kind='stable')
        nn=np.take_along_axis(nn,order,axis=1)
        indices[ids]=nn;distances[ids]=np.take_along_axis(d,nn,axis=1)
    assert not np.any(indices==np.arange(len(x))[:,None])
    np.savez_compressed(target,indices=indices,distances=distances)
    return indices,distances


def geometry(c,out,report,x,y,f,rng):
    print('GEOMETRY: fitting descriptive ability/item-difficulty control',flush=True)
    p,ability,item=fit_rasch(y,f)
    np.savez_compressed(out/'descriptive_rasch.npz',probabilities=p,ability=ability,item=item)
    residual=y.astype(np.float32)-p;raw=family_center(y,f)
    anchors=np.concatenate([rng.choice(np.flatnonzero(f==z),min(c['anchors_per_family'],np.sum(f==z)),replace=False) for z in np.unique(f)])
    weights=np.zeros(len(anchors))
    for z in np.unique(f):weights[f[anchors]==z]=1/(len(np.unique(f))*np.sum(f[anchors]==z))
    graphs={}
    for metric in c['metrics']:
        nn,_=nearest(x,metric,max(c['neighbors']),out)
        for k in c['neighbors']:graphs[f'{metric}_k{k}']=nn[anchors,:k]
        # Additional within/cross-benchmark neighborhoods use the entire candidate bank.
        for cross in [False,True]:
            graph=np.empty((len(anchors),20),dtype=int)
            for start in range(0,len(anchors),64):
                ids=anchors[start:start+64];d=cdist(x[ids],x,metric)
                same=f[ids,None]==f[None,:]
                d[same if cross else ~same]=np.inf;d[np.arange(len(ids)),ids]=np.inf
                graph[start:start+len(ids)]=np.argpartition(d,19,axis=1)[:,:20]
            graphs[f'{metric}_'+('cross' if cross else 'within')+'_k20']=graph
    observed={name:dict(raw=graph_association(raw,anchors,graph,weights),
                        residual=graph_association(residual,anchors,graph,weights)) for name,graph in graphs.items()}
    nulls={name:{kind:[] for kind in ['family_shuffle','difficulty_shuffle','rasch_simulation']} for name in graphs}
    family_strata=[np.flatnonzero(f==z) for z in np.unique(f)]
    difficulty_strata=make_strata(f,item,20)
    for rep in range(c['null_repetitions']):
        for kind in ['family_shuffle','difficulty_shuffle','rasch_simulation']:
            if kind=='rasch_simulation':yn=(rng.random(y.shape)<p).astype(np.float32)
            else:yn=y[permute_strata(family_strata if kind=='family_shuffle' else difficulty_strata,len(y),rng)].astype(np.float32)
            rn=yn-p
            for name,graph in graphs.items():nulls[name][kind].append(graph_association(rn,anchors,graph,weights))
        if rep%10==0:print('GEOMETRY null',rep,flush=True)
    summary={};rows=[]
    for name in graphs:
        result={kind:float(np.nanmean(v)) for kind,v in observed[name].items()}
        for kind,array in nulls[name].items():
            array=np.asarray(array);macro=np.nanmean(array,axis=1)
            result[kind]=dict(mean=float(macro.mean()),range_95=np.quantile(macro,[.025,.975]).tolist(),
                monte_carlo_p=float((1+np.sum(macro>=result['residual']))/(len(macro)+1)))
        summary[name]=result
        for m in range(y.shape[1]):rows.append(dict(graph=name,model_column=m,raw=float(observed[name]['raw'][m]),residual=float(observed[name]['residual'][m])))
    # Feature collisions are descriptive; empirical cell rates are NOT held-out probabilities.
    _,inverse,counts=np.unique(x,axis=0,return_inverse=True,return_counts=True)
    sums=np.zeros((len(counts),y.shape[1]));np.add.at(sums,inverse,y)
    ambiguous=(sums>0)&(sums<counts[:,None])
    collisions=dict(unique_profiles=len(counts),questions=len(x),questions_in_repeated_profiles=int(counts[counts>1].sum()),
        median_ambiguous_repeated_cells=float(np.median(ambiguous[counts>1].mean(0))),
        optimistic_in_sample_majority_accuracy=float(np.maximum(sums,counts[:,None]-sums).sum(0).mean()/len(y)),
        warning='Cell-majority accuracy uses all labels, is optimistic, and is not an estimated Bayes limit or held-out accuracy.')
    save(report/'geometry.json',dict(results=summary,collisions=collisions,anchors=len(anchors),
        interpretation='Association is neighbor residual product divided by anchor residual variance, family balanced. Rasch and difficulty strata use all outcomes only for descriptive controls; they never enter held-out predictors. Null ranges are not effect confidence intervals.'))
    with (report/'geometry_per_model.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(out/'anchors.npz',indices=anchors,weights=weights)
    return p,residual


def spectral(c,out,report,x,y,f,p,residual,rng):
    print('SPECTRAL: normalized graph eigenvectors',flush=True)
    nn,d=nearest(x,'euclidean',max(c['neighbors']),out);nn=nn[:,:20];d=d[:,:20]
    widths=np.maximum(d[:,-1],1e-4)
    affinity=np.exp(-d*d/(widths[:,None]**2))
    w=sp.csr_matrix((affinity.ravel(),(np.repeat(np.arange(len(x)),20),nn.ravel())),shape=(len(x),len(x)))
    w=(w+w.T)*.5
    components,component_ids=connected_components(w)
    degree=np.asarray(w.sum(1)).ravel();root=np.sqrt(degree)
    normalized=sp.diags(1/root)@w@sp.diags(1/root)
    eigenvalues,eigenvectors=eigsh(normalized,k=c['spectral_modes']+min(components,10),which='LA',
        tol=1e-5,maxiter=3000,v0=rng.normal(size=len(x)))
    order=np.argsort(eigenvalues)[::-1];eigenvalues=eigenvalues[order];eigenvectors=eigenvectors[:,order]
    valid=eigenvalues<1-1e-7;eigenvalues=eigenvalues[valid][:c['spectral_modes']];u=eigenvectors[:,valid][:,:c['spectral_modes']]
    np.savez_compressed(out/'spectral_basis.npz',eigenvalues=eigenvalues,eigenvectors=u,degree=degree,component_ids=component_ids)
    answer={};coefficients={}
    for name,signal in [('raw',family_center(y,f)),('residual',residual)]:
        z=root[:,None]*signal;coeff=u.T@z;coefficients[name]=coeff
        total=(z*z).sum(0)
        observed={str(k):float(np.mean((coeff[:k]**2).sum(0)/total)) for k in [8,24,48]}
        null=[]
        for rep in range(30):
            yn=(rng.random(y.shape)<p).astype(np.float32)
            rn=yn-p if name=='residual' else family_center(yn,f)
            zn=root[:,None]*rn;cn=u.T@zn
            null.append([(cn[:k]**2).sum(0).mean()/(zn*zn).sum(0).mean() for k in [8,24,48]])
        null=np.asarray(null)
        answer[name]=dict(energy_fraction=observed,rasch_null_mean={str(k):float(null[:,j].mean()) for j,k in enumerate([8,24,48])},
                         rasch_null_range_95={str(k):np.quantile(null[:,j],[.025,.975]).tolist() for j,k in enumerate([8,24,48])})
    np.savez_compressed(out/'model_spectral_coefficients.npz',**coefficients)
    # Compare residual model-pair relations across question halves and benchmark families.
    similarities={};pairs=np.triu_indices(y.shape[1],1)
    for family in np.unique(f):
        ids=np.flatnonzero(f==family)
        r=residual[ids].astype(float);r-=r.mean(0)
        norm=np.sqrt((r*r).sum(0));similarities[str(family)]=(r.T@r)/np.maximum(norm[:,None]*norm[None,:],1e-9)
    relations=[]
    for i,a in enumerate(sorted(similarities)):
        for b in sorted(similarities)[i+1:]:relations.append(dict(a=a,b=b,pair_similarity_spearman=float(spearmanr(similarities[a][pairs],similarities[b][pairs]).statistic)))
    # This residual covariance has row constraints from fitting item effects; do not
    # interpret negative mean correlation as antagonistic capabilities.
    covariance=np.cov(residual,rowvar=False)
    values=np.maximum(np.linalg.eigvalsh(covariance),0)
    effective=float(values.sum()**2/np.sum(values**2))
    null_ranks=[]
    for rep in range(20):
        rn=(rng.random(y.shape)<p).astype(float)-p
        v=np.maximum(np.linalg.eigvalsh(np.cov(rn,rowvar=False)),0)
        null_ranks.append(float(v.sum()**2/np.sum(v*v)))
    save(report/'spectral.json',dict(components=int(components),nonconstant_modes=len(eigenvalues),
        signal_energy=answer,residual_covariance_effective_rank=effective,rasch_null_rank_mean=float(np.mean(null_ranks)),
        cross_family_model_similarity=relations,
        warning='All-label descriptive spectral projections, not held-out explained variance. Item-effect fitting couples residuals; negative correlations are not evidence of opposing skills.'))
    return u


def topology(c,out,report,x,y,f,p,provenance,rng):
    print('TOPOLOGY: persistent H0/H1, fixed 24-model cohort',flush=True)
    landmarks=np.concatenate([rng.choice(np.flatnonzero(f==z),min(c['topology_landmarks_per_family'],np.sum(f==z)),replace=False) for z in np.unique(f)])
    strata=make_strata(f[landmarks],p[landmarks].mean(1),4)
    rows=[]
    for model in provenance['pilot_models']:
        labels=y[landmarks,model]
        observed=persistence_statistics(x[landmarks[labels==1]])
        null=[]
        for rep in range(c['topology_null_repetitions']):
            permutation=permute_strata(strata,len(landmarks),rng)
            null.append(persistence_statistics(x[landmarks[labels[permutation]==1]]))
        row=dict(model_column=model,**observed)
        for key in ['h0_max','h1_max','h1_total']:
            values=np.array([v[key] for v in null]);row[key+'_null_mean']=float(values.mean())
            row[key+'_null_low']=float(np.quantile(values,.025));row[key+'_null_high']=float(np.quantile(values,.975))
            row[key+'_p']=float((1+np.sum(values>=observed[key]))/(len(values)+1))
        rows.append(row)
        print('TOPOLOGY model',model,flush=True)
    save(report/'topology.json',dict(landmarks=len(landmarks),models=len(rows),rows=rows,
        control='Within-benchmark and 4 difficulty strata, shuffle observed outcomes: same success counts in each stratum. Deduplicate exact vectors before Rips complexes.',
        limitations='Finite subsample, 20 permutations, unadjusted exploratory tests; feature metric and sparse sampling can create holes. No result establishes cognitive topology.'))
    with (report/'topology_per_model.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    c=json.loads((ROOT/'config_delean_shape_v1.json').read_text());out=ROOT/c['output'];report=ROOT/c['report']
    out.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    started=time.time();rng=np.random.default_rng(c['seed'])
    x,y,f,g,provenance=prepare(c,out)
    for stage in ['geometry','spectral','topology']:
        save(out/'status.json',dict(stage=stage,elapsed_seconds=time.time()-started))
        if stage=='geometry':p,residual=geometry(c,out,report,x,y,f,rng)
        elif stage=='spectral':spectral(c,out,report,x,y,f,p,residual,rng)
        else:topology(c,out,report,x,y,f,p,provenance,rng)
    save(out/'shape_completion.json',dict(stage='complete',elapsed_seconds=time.time()-started,
        peak_rss_mb=__import__('psutil').Process().memory_info().peak_wset/2**20))
    print('SHAPE DIAGNOSTICS COMPLETE',time.time()-started,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

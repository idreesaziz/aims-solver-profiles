"""Fit/export source-only 160-tree profile forests for portable inference.

Uses the same 11,722 source rows, 192 outputs and fit hyperparameters as the
benchmark-cold experiments, now with all source benchmarks present. No target
or protected labels are loaded. Exports plain arrays, not sklearn pickles.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,time
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeRegressor
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import demand_residual_adaptation_v48 as cache
ROOT=REPO/'artifacts/final_v74'


def portable_predict(forest,x):
    x=np.asarray(x,dtype=np.float32);out=np.zeros((len(x),192))
    for root in forest['roots']:
        node=np.full(len(x),root,dtype=np.int32)
        while True:
            active=forest['left'][node]>=0
            if not active.any():break
            at=node[active];go=x[np.flatnonzero(active),forest['feature'][at]]<=forest['threshold'][at]
            node[active]=np.where(go,forest['left'][at],forest['right'][at])
        out+=forest['values'][forest['leaf'][node]]
    return out/len(forest['roots'])


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    ix=np.load(cache.BASE/'indices_and_weights.npz')
    x=np.asarray(np.load(cache.BASE/'features.npy',mmap_mode='r')[:len(ix['fit'])],np.float32)
    y=np.asarray(np.load(cache.BASE/'historical_training_labels.npy',mmap_mode='r'),np.float64)
    w=ix['weights'].copy();w*=len(w)/w.sum()
    assert x.shape==(11722,70) and y.shape==(11722,192)
    assert set(ix['historical']).isdisjoint(ix['held'])
    assert np.isfinite(x).all() and np.isin(y,[0,1]).all()
    seed=2026102946;seeds=np.random.RandomState(seed).randint(np.iinfo(np.int32).max,size=160)
    plan=dict(rows=11722,outputs=192,trees=160,seed=seed,min_samples_leaf=12,max_features=1.,
        fit_role='source train only; all source benchmark families; 24 target configurations and all development/protected outcomes excluded',
        historical_columns=ix['historical'].tolist(),excluded_target_columns=ix['held'].tolist(),
        files={str(p):cache.sha(p) for p in [cache.BASE/'features.npy',cache.BASE/'historical_training_labels.npy',cache.BASE/'indices_and_weights.npz',Path(__file__),REPO/'models/manual_text_features_v1.py']})
    cache.save(ROOT/'fit_plan.json',plan)
    for bank,columns in [('structural52',slice(18,70)),('combined70',slice(0,70))]:
        target=ROOT/(bank+'.npz');receipt=ROOT/(bank+'_receipt.json')
        if target.exists() and receipt.exists():
            old=cache.read(receipt);assert cache.sha(target)==old['sha256'];print('CACHED',bank,flush=True);continue
        xx=np.ascontiguousarray(x[:,columns]);probe=xx[np.linspace(0,len(xx)-1,129,dtype=int)]
        fields={k:[] for k in ['left','right','feature','threshold','leaf','values']};roots=[];nodes=leaves=0
        expected=np.zeros((len(probe),192));began=time.monotonic()
        for j,s in enumerate(seeds):
            sample=np.random.RandomState(int(s)).randint(0,len(xx),len(xx),dtype=np.int32)
            tree=DecisionTreeRegressor(min_samples_leaf=12,max_features=1.,random_state=int(s))
            tree.fit(xx,y,sample_weight=w*np.bincount(sample,minlength=len(xx)))
            expected+=tree.predict(probe)
            t=tree.tree_;isleaf=t.children_left<0
            left=t.children_left.copy();right=t.children_right.copy();left[~isleaf]+=nodes;right[~isleaf]+=nodes
            leaf=np.full(t.node_count,-1,dtype=np.int32);leaf[isleaf]=leaves+np.arange(isleaf.sum())
            fields['left'].append(left.astype(np.int32));fields['right'].append(right.astype(np.int32))
            fields['feature'].append(t.feature.astype(np.int16));fields['threshold'].append(t.threshold.copy())
            fields['leaf'].append(leaf);fields['values'].append(t.value[isleaf,:,0].astype(np.float32))
            roots.append(nodes);nodes+=t.node_count;leaves+=int(isleaf.sum())
            if (j+1)%20==0:print('FIT',bank,j+1,160,round(time.monotonic()-began,1),flush=True)
        forest={k:np.concatenate(v,axis=0) for k,v in fields.items()};forest['roots']=np.array(roots,np.int32)
        got=portable_predict(forest,probe);err=float(np.max(abs(got-expected/160)))
        assert err<1e-7,err
        np.savez_compressed(target,**forest)
        # Loading with allow_pickle=False checks the exact portable artifact.
        with np.load(target,allow_pickle=False) as check:
            assert np.max(abs(portable_predict(check,probe)-got))==0.
        cache.save(receipt,dict(bank=bank,features=xx.shape[1],nodes=nodes,leaves=leaves,trees=160,
            seconds=time.monotonic()-began,max_export_probability_error=err,sha256=cache.sha(target),plan_sha256=cache.sha(ROOT/'fit_plan.json')))
        print('EXPORTED',bank,err,target.stat().st_size,flush=True)
        del forest,fields


if __name__=='__main__':main()

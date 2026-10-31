"""Secondary support-only CV control declared before v48 outer scoring."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from scripts import demand_residual_adaptation_v48 as base
OUT=base.ROOT/'support_cv_secondary'
GRID=[4.,16.,64.,256.,4096.]
BUDGETS=[0,1,3,7,15,31]
ALC=np.array([.1,.2,.2,.2,.2,.1])


def choose_from_support(xs,xq,p0s,p0q,ys,kind,fallback):
    """Features and p0 are fixed. Query labels are not an input."""
    n=len(xs);models=ys.shape[1]
    if n==0:return np.broadcast_to(p0q[:,None],(len(p0q),models)).copy(),np.full(models,-1,dtype=int)
    ds=np.c_[np.ones(n),xs];dq=np.c_[np.ones(len(xq)),xq]
    residual=ys-p0s[:,None]
    if n<7:
        if fallback is None:return np.broadcast_to(p0q[:,None],(len(p0q),models)).copy(),np.full(models,-1,dtype=int)
        g,l=fallback;penalty=np.r_[g,np.repeat(l,xs.shape[1])]
        coef=np.linalg.solve(ds.T@ds+np.diag(penalty),ds.T@residual)
        return np.clip(p0q[:,None]+dq@coef,0,1),np.full(models,-2,dtype=int)
    # A zero-correction candidate is included. Each candidate's LOO score uses support only.
    risks=[np.square(ys-p0s[:,None]).mean(0)];parameters=[None]
    for g in GRID:
        for local in ([1.] if xs.shape[1]==0 else GRID):
            penalty=np.r_[g,np.repeat(local,xs.shape[1])]
            inverse=np.linalg.solve(ds.T@ds+np.diag(penalty),ds.T)
            hat=ds@inverse
            fitted=hat@residual
            denominator=np.maximum(1-np.diag(hat),1e-8)
            error_loo=(residual-fitted)/denominator[:,None]
            prediction_loo=np.clip(ys-error_loo,0,1)
            risks.append(np.square(prediction_loo-ys).mean(0));parameters.append((g,local))
    chosen=np.argmin(np.asarray(risks),axis=0)
    output=np.broadcast_to(p0q[:,None],(len(p0q),models)).copy()
    for candidate in np.unique(chosen):
        if candidate==0:continue
        g,l=parameters[candidate];penalty=np.r_[g,np.repeat(l,xs.shape[1])]
        models_at=np.flatnonzero(chosen==candidate)
        coefficient=np.linalg.solve(ds.T@ds+np.diag(penalty),ds.T@residual[:,models_at])
        output[:,models_at]=np.clip(p0q[:,None]+dq@coefficient,0,1)
    return output,chosen


def prepare():
    OUT.mkdir(exist_ok=True)
    path=OUT/'plan.json'
    if path.exists():
        p=base.read(path);assert p['script_sha256']==base.sha(__file__);return p
    assert not (base.ROOT/'prediction_freeze.json').exists(),'Declare before main outer scoring.'
    p=dict(script_sha256=base.sha(__file__),methods=['support_cv_intercept','support_cv_probability_slope','support_cv_learned_demand'],
       global_penalties=GRID,local_penalties=GRID,minimum_cv_answers=7,include_null_correction=True,
       early_budget_fallback='Source-selected frozen intercept/slope/learned parameters; no future labels.',
       learned_map_choice='Lowest SOURCE validation ALC among the four frozen learned maps. Does not inspect outer outcomes.',
       support_cv='Exact ridge leave-one-out predictions, clipped probability Brier; choose regularization separately per subject using current support only.',
       scope='Secondary adaptive-regularization control added after source simulations showed strong shrinkage, before any current-run outer scores.',
       limitations=['Support CV may overfit31answers; no risk guarantee.', 'Source-selected learned map is tested even if source preferred no correction.'],
       new_model_fitting='Only small ridge adaptation from currently allowed support answers.',protected_test_used=False,paid_calls=0)
    base.save(path,p);return p


def smoke():
    rng=np.random.default_rng(488)
    x=rng.normal(size=(31,4));q=rng.normal(size=(19,4));ps=rng.uniform(.1,.9,31);pq=rng.uniform(.1,.9,19);ys=rng.integers(0,2,(31,3))
    a,_=choose_from_support(x,q,ps,pq,ys,'demand',(64.,16.))
    perm=rng.permutation(31);b,_=choose_from_support(x[perm],q,ps[perm],pq,ys[perm],'demand',(64.,16.))
    assert np.max(abs(a-b))<1e-12
    ds=np.c_[np.ones(31),x];r=ys-ps[:,None];penalty=np.diag([16.,64.,64.,64.,64.])
    inverse=np.linalg.solve(ds.T@ds+penalty,ds.T);hat=ds@inverse
    fast=ys-(r-hat@r)/(1-np.diag(hat))[:,None]
    brute=np.empty_like(fast)
    for i in range(31):
        keep=np.arange(31)!=i
        beta=np.linalg.solve(ds[keep].T@ds[keep]+penalty,ds[keep].T@r[keep])
        brute[i]=ps[i]+ds[i]@beta
    assert np.max(abs(fast-brute))<1e-12
    print('SUPPORT CV SMOKE PASS: analytic LOO equals31explicit refits; permutation invariant',flush=True)


def run():
    plan=prepare();assert (base.ROOT/'prediction_freeze.json').exists()
    idx=np.load(base.BASE/'indices_and_weights.npz');nf=len(idx['fit'])
    # All source models/choices are already frozen. Outer labels enter only scoring/support.
    y=np.load(base.SWEEP/'development_outcomes.npy')
    allx=np.load(base.BASE/'features.npy',mmap_mode='r');records=[];selection_counts={}
    for protocol in base.PROTOCOLS:
        folder=base.ROOT/protocol;selected=base.read(folder/'selection_frozen.json');split=np.load(folder/'source_split.npz')
        source_positions,directory,_,devpos=base.protocol_meta(protocol,idx)
        expert=np.clip(np.load(directory/'predictions.npy'),1e-4,1-1e-4);p0=expert.mean(1)
        raw=np.array(allx[nf+devpos,:18],copy=True)
        if protocol!='mixed':raw[np.load(base.COLD/'missing_development_demands.npy')[devpos]]=np.median(allx[source_positions,:18],axis=0)
        d=np.clip((raw-split['demand_mean'])/split['demand_scale'],-5,5)
        target=y[devpos];family=idx['families'][nf:][devpos];groups=idx['groups'][nf:][devpos]
        if protocol=='mixed':
            from scripts.coordinate_brier_ablation_v42 import support_orders
            orders=support_orders(family,groups)
        else:
            z=np.load(base.COLD/protocol/'orders.npz');orders=list(zip(z['support'],z['evaluation']))
        learned=min(selected['learned'],key=lambda z:z['source_validation']['alc'])
        learned_cfg=learned['config']
        model_features={
            'support_cv_intercept':np.empty((len(d),0)),
            'support_cv_probability_slope':((p0-float(split['probability_mean']))/float(split['probability_scale']))[:,None],
            'support_cv_learned_demand':base.prediction_features(folder,d,learned_cfg,p0,expert)}
        fallback={}
        for method,key in [('support_cv_intercept','intercept'),('support_cv_probability_slope','probability_slope')]:
            cfg=selected['fixed'][key]['config']
            if cfg['kind']=='none':fallback[method]=None
            elif method=='support_cv_probability_slope' and cfg['kind']!='probability_slope':
                fallback[method]=(cfg['global_penalty'],1e12)
            else:fallback[method]=(cfg['global_penalty'],cfg['local_penalty'])
        fallback['support_cv_learned_demand']=(learned_cfg['global_penalty'],learned_cfg['local_penalty'])
        for method,u in model_features.items():
            count=np.zeros(26,dtype=int)
            for seed,(support,evaluation) in enumerate(orders):
                for budget in BUDGETS:
                    chosen=support[:budget]
                    prediction,choice=choose_from_support(u[chosen],u[evaluation],p0[chosen],p0[evaluation],target[chosen],method,fallback[method])
                    if budget==31:count+=np.bincount(choice,minlength=26)
                    error=np.square(prediction-target[evaluation])
                    for f in np.unique(family):
                        mask=family[evaluation]==f
                        if not mask.any():continue
                        for model,value in zip(idx['held'],error[mask].mean(0)):
                            records.append((protocol,method,str(f),seed,budget,int(model),float(value)))
            selection_counts[protocol+'::'+method]=count.tolist()
        print('SUPPORT_CV',protocol,flush=True)
    with (OUT/'per_model.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['protocol','method','family','seed','budget','model','brier']);w.writerows(records)
    summaries={}
    for group in ('mixed','cold_primary','cold_all_secondary'):
        subset=[r for r in records if (r[0]=='mixed' if group=='mixed' else r[0]!='mixed') and (group!='cold_primary' or r[2]!='MuSR')]
        summaries[group]=[]
        for method in plan['methods']:
            rows=[r for r in subset if r[1]==method]
            curve={str(b):float(np.mean([r[6] for r in rows if r[4]==b])) for b in BUDGETS}
            summaries[group].append(dict(method=method,brier31=curve['31'],alc=float(np.array(list(curve.values()))@ALC),curve=curve))
    base.save(OUT/'result.json',dict(completed=True,plan=plan,summaries=summaries,penalty_selections=selection_counts))
    for group,rows in summaries.items():
        for r in rows:print(group,r['method'],r['brier31'],r['alc'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['smoke','prepare','run']);a=p.parse_args()
    with threadpool_limits(1):
        if a.action=='smoke':smoke()
        elif a.action=='prepare':prepare()
        else:run()

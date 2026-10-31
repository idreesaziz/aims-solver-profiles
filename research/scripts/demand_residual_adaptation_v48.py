"""Source-selected residual adaptation on cached forests; CPU-only experiment."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from models import demand_residual_adaptation_v48 as alg

ROOT=Path('D:/AIMS/demand_residual_adaptation_v48')
BASE=Path('D:/AIMS/demand_expert_transfer_v45')
COLD=Path('D:/AIMS/model_scorecard_prior_v46')
SWEEP=Path('D:/AIMS/core_subset_finetune_v44')
CORE=['BBH','GPQA','MATH','MMLU-Pro','MuSR']
PROTOCOLS=['mixed']+CORE
SEED=2026102948
STEPS=800
PENALTIES=[1.,4.,16.,64.,256.,1024.,4096.]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save_array(path,x):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('wb') as f:np.save(f,x)
    tmp.replace(path)


def status(**kwargs):
    save(ROOT/'status.json',dict(time=time.time(),**kwargs))


def plan_prepare():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'plan.json').exists():
        plan=read(ROOT/'plan.json')
        for f,digest in plan['frozen_files'].items():assert sha(f)==digest,f
        return plan
    idx=np.load(BASE/'indices_and_weights.npz')
    assert len(idx['historical'])==192 and len(idx['held'])==24
    assert not set(idx['historical'])&set(idx['held'])
    files=[Path(__file__),REPO/'models/demand_residual_adaptation_v48.py',BASE/'plan.json',
        BASE/'indices_and_weights.npz',BASE/'features.npy',BASE/'historical_training_labels.npy',
        BASE/'demand_manual/predictions.npy',COLD/'plan.json',REPO/'scripts/demand_expert_transfer_v45.py',
        REPO/'scripts/core_subset_finetune_v44.py']
    files += [COLD/f/'predictions.npy' for f in CORE]
    plan=dict(frozen_files={str(p):sha(p) for p in files},protocols=PROTOCOLS,steps=STEPS,batch=8,
         learned_ranks=[4,8],learned_seeds=[42,49],validation_every=100,
         historical=idx['historical'].tolist(),held=idx['held'].tolist(),
         controls=['none','bayes','brier_mean','brier_strong','intercept','probability_slope','demand_linear','demand_rbf','rf_covariance'],
         source_selection='32historical configurations + disjoint source question groups; ALC only.',
         group_oob_minimum=8,meta_validation_pretraining_limit='RF splits were pretrained using all192historical configurations. Meta-validation is not a wholly model-disjoint RF refit; outer24configurations never enter source fitting.',
         protected_test_used=False,paid_calls=0,gpu_used=False,root_model_modified=False,
         target_scorecards_used=False,prior_experiments_preserved=True)
    save(ROOT/'plan.json',plan)
    return plan


def protocol_meta(name,idx):
    n=len(idx['fit'])
    if name=='mixed':
        positions=np.arange(n);directory=BASE/'demand_manual';forest_seed=2026102549
        devpos=np.arange(len(idx['development']))
    else:
        positions=np.asarray(read(COLD/name/'fit_receipt.json')['source_positions'])
        directory=COLD/name;forest_seed=2026102946
        devpos=np.load(COLD/name/'orders.npz')['development_positions']
        assert np.all(idx['families'][positions]!=name)
    return positions,directory,forest_seed,devpos


def group_oob(name,idx):
    import joblib
    folder=ROOT/name;folder.mkdir(exist_ok=True)
    positions,directory,forest_seed,devpos=protocol_meta(name,idx)
    if (folder/'oob_predictions.npy').exists():return
    data=np.load(BASE/'features.npy',mmap_mode='r')
    x=np.asarray(data[positions],dtype=np.float32)
    groups=idx['groups'][positions]
    unique,inverse=np.unique(groups,return_inverse=True)
    seeds=np.random.RandomState(forest_seed).randint(np.iinfo(np.int32).max,size=160)
    total=np.zeros((len(x),192));count=np.zeros(len(x),dtype=np.int32);begin=0
    if (folder/'oob_resume.npz').exists():
        resume=np.load(folder/'oob_resume.npz');total=resume['total'];count=resume['count'];begin=int(resume['completed'])
    for at in range(begin,160):
        sample=np.random.RandomState(int(seeds[at])).randint(0,len(x),len(x),dtype=np.int32)
        group_inbag=np.bincount(inverse[sample],minlength=len(unique))>0
        oob=np.flatnonzero(~group_inbag[inverse])
        assert set(groups[oob]).isdisjoint(groups[sample])
        tree=joblib.load(directory/f'tree_{at:03d}.joblib')
        assert tree.n_outputs_==192
        total[oob]+=tree.predict(x[oob]);count[oob]+=1
        del tree
        if (at+1)%20==0:
            resume_path=folder/'oob_resume.npz'
            with resume_path.with_suffix('.tmp').open('wb') as f:
                np.savez(f,total=total,count=count,completed=at+1)
            resume_path.with_suffix('.tmp').replace(resume_path)
            status(stage='group_oob',protocol=name,trees=at+1,total_trees=160)
            print('OOB',name,at+1,flush=True)
    good=count>=8
    prediction=np.full_like(total,np.nan)
    prediction[good]=total[good]/count[good,None]
    save_array(folder/'oob_predictions.npy',prediction)
    save_array(folder/'oob_counts.npy',count)
    save(folder/'oob_receipt.json',dict(source_positions=positions.tolist(),development_positions=devpos.tolist(),
         valid_rows=int(good.sum()),excluded_low_oob_rows=int((~good).sum()),minimum_count=int(count.min()),
         group_bootstrap_exclusion_checked_every_tree=True,tree_directory=str(directory),forest_seed=forest_seed,
         oob_sha256=sha(folder/'oob_predictions.npy'),count_sha256=sha(folder/'oob_counts.npy')))
    del x,total,prediction
    gc.collect()


def source_prepare(name,idx):
    folder=ROOT/name
    if (folder/'source_split.npz').exists():return
    positions=protocol_meta(name,idx)[0]
    counts=np.load(folder/'oob_counts.npy')
    good=counts>=8
    families=idx['families'][positions];groups=idx['groups'][positions]
    validation_groups=set()
    rng=np.random.default_rng(SEED)
    for family in CORE:
        candidate=np.unique(groups[good&(families==family)])
        if not len(candidate):continue
        nval=max(48,int(np.ceil(.2*len(candidate))))
        assert len(candidate)-nval>=63
        val=rng.choice(candidate,nval,replace=False)
        validation_groups.update(val.tolist())
    validation_mask=good&np.isin(groups,list(validation_groups))
    train_mask=good&~np.isin(groups,list(validation_groups))
    assert set(groups[train_mask]).isdisjoint(groups[validation_mask])
    val_models=np.sort(np.random.default_rng(SEED+1).choice(192,32,replace=False))
    train_models=np.setdiff1d(np.arange(192),val_models)
    raw=np.load(BASE/'features.npy',mmap_mode='r')[positions,:18]
    w=idx['weights'][positions]*train_mask
    # Equal family/task weighting inherited from the baseline, renormalized after filtering.
    w=w/w.sum()
    mean=w@raw;scale=np.maximum(np.sqrt(w@np.square(raw-mean)),.05)
    d=np.clip((raw-mean)/scale,-5,5).astype(np.float32)
    anchors=[]
    for family in np.unique(families):
        candidates=np.flatnonzero(train_mask&(families==family))
        anchors.extend(rng.choice(candidates,128,replace=len(candidates)<128))
    anchors=np.asarray(anchors[:512],dtype=np.int32)
    # Do not read target-model histories: this array has only the historical192columns.
    y=np.asarray(np.load(BASE/'historical_training_labels.npy',mmap_mode='r')[positions],dtype=np.float32)
    p=np.load(folder/'oob_predictions.npy',mmap_mode='r')
    base=np.clip(np.asarray(p[train_mask]),1e-4,1-1e-4).mean(1)
    np.savez(folder/'source_split.npz',source_positions=positions,train_mask=train_mask,validation_mask=validation_mask,
        train_models=train_models,validation_models=val_models,demand_mean=mean,demand_scale=scale,
        anchor_positions=anchors,probability_mean=base.mean(),probability_scale=max(base.std(),.05))
    save_array(folder/'source_demands.npy',d)
    save(folder/'source_split_receipt.json',dict(meta_training_configurations=160,meta_validation_configurations=32,
          training_question_groups=len(set(groups[train_mask])),validation_question_groups=len(set(groups[validation_mask])),
          source_rows=len(positions),source_historical_labels_only=True,source_demands_only_for_learned_map=True))


class Source:
    def __init__(self,name,idx):
        self.name=name;self.folder=ROOT/name;self.idx=idx
        self.split=np.load(self.folder/'source_split.npz')
        self.positions=self.split['source_positions']
        self.x=np.load(self.folder/'source_demands.npy')
        self.p=np.load(self.folder/'oob_predictions.npy',mmap_mode='r')
        self.y=np.asarray(np.load(BASE/'historical_training_labels.npy',mmap_mode='r')[self.positions],dtype=np.float32)
        self.groups=idx['groups'][self.positions];self.families=idx['families'][self.positions]
        self.pools={}
        for role in ('train','validation'):
            mask=self.split[role+'_mask']
            pools={}
            for family in CORE:
                rows=np.flatnonzero(mask&(self.families==family))
                if not len(rows):continue
                mapping={}
                for row in rows:mapping.setdefault(str(self.groups[row]),[]).append(int(row))
                pools[family]=list(mapping.values())
            self.pools[role]=pools

    def episode(self,rng,role,h=None,family=None):
        if h is None:h=int(rng.choice(self.split[role+'_models']))
        pools=self.pools[role]
        support=[];query=[]
        if self.name=='mixed':
            fs=list(pools);extra=int(rng.integers(len(fs)));used=set()
            for j,f in enumerate(fs):
                n=6+int(j==extra);nq=8 if role=='train' else 4
                eligible=[k for k,pool in enumerate(pools[f]) if self.groups[pool[0]] not in used]
                choices=rng.choice(eligible,n+nq,replace=False)
                rows=[int(rng.choice(pools[f][g])) for g in choices]
                used.update(self.groups[rows].tolist())
                support.extend(rows[:n]);query.extend(rows[n:])
            support=rng.permutation(support).tolist()
        else:
            f=family if family is not None else str(rng.choice(list(pools)))
            nq=32 if role=='train' else 16
            choices=rng.choice(len(pools[f]),31+nq,replace=False)
            rows=[int(rng.choice(pools[f][g])) for g in choices]
            support=rows[:31];query=rows[31:]
        s=np.array(support);q=np.array(query)
        assert len(s)==31 and set(self.groups[s]).isdisjoint(self.groups[q])
        return s,q,h

    def batch(self,episodes,experts=False):
        s=np.stack([e[0] for e in episodes]);q=np.stack([e[1] for e in episodes]);h=np.array([e[2] for e in episodes])
        ps=np.clip(np.asarray(self.p[s]),1e-4,1-1e-4)
        pq=np.clip(np.asarray(self.p[q]),1e-4,1-1e-4)
        mask=np.arange(192)[None,:]!=h[:,None]
        ps=ps[np.broadcast_to(mask[:,None,:],ps.shape)].reshape(len(s),s.shape[1],191)
        pq=pq[np.broadcast_to(mask[:,None,:],pq.shape)].reshape(len(q),q.shape[1],191)
        out=dict(s=s,q=q,h=h,xs=self.x[s],xq=self.x[q],base_s=ps.mean(2),base_q=pq.mean(2),
                 labels=self.y[s,h[:,None]].astype(float),truth=self.y[q,h[:,None]].astype(float))
        if experts:out.update(ps=ps,pq=pq)
        return out

    def validation(self):
        path=self.folder/'validation_episodes.npz'
        if path.exists():
            z=np.load(path)
            return self.batch(list(zip(z['s'],z['q'],z['h'])),experts=True)
        rng=np.random.default_rng(SEED+7);episodes=[]
        for h in self.split['validation_models']:
            if self.name=='mixed':
                episodes.extend([self.episode(rng,'validation',h=int(h)) for _ in range(2)])
            else:
                episodes.extend([self.episode(rng,'validation',h=int(h),family=f) for f in self.pools['validation']])
        np.savez(path,s=np.stack([e[0] for e in episodes]),q=np.stack([e[1] for e in episodes]),h=np.array([e[2] for e in episodes]))
        return self.batch(episodes,experts=True)


def prediction_curve(batch,cfg,features=None):
    name=cfg['kind'];predictions={}
    for b in alg.BUDGETS:
        if name=='none':p=batch['base_q']
        elif name in ('bayes','brier_mean','brier_strong'):
            p=alg.expert_prediction(batch['ps'],batch['pq'],batch['labels'],b,name)
        elif name=='demand_rbf':
            if b==0:p=batch['base_q']
            else:
                ks,kq=alg.rbf_blocks(batch['xs'][:,:b],batch['xq'],cfg['length'])
                p=alg.ridge_kernel(ks,kq,batch['base_s'],batch['base_q'],batch['labels'],b,cfg['global_penalty'],cfg['local_penalty'])
        else:
            us,uq=features
            p=alg.ridge_features(us,uq,batch['base_s'],batch['base_q'],batch['labels'],b,cfg['global_penalty'],cfg['local_penalty'])
        assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
        predictions[b]=p
    return predictions


def metric(batch,predictions):
    curve={str(b):float(np.square(predictions[b]-batch['truth']).mean()) for b in alg.BUDGETS}
    return dict(alc=float(np.array(list(curve.values()))@alg.ALC),brier31=curve['31'],curve=curve)


def configured_features(source,batch,cfg):
    kind=cfg['kind']
    if kind=='intercept':return batch['xs'][:,:,:0],batch['xq'][:,:,:0]
    if kind=='probability_slope':
        center=float(source.split['probability_mean']);scale=float(source.split['probability_scale'])
        return ((batch['base_s']-center)/scale)[:,:,None],((batch['base_q']-center)/scale)[:,:,None]
    if kind=='demand_linear':return batch['xs']/np.sqrt(18),batch['xq']/np.sqrt(18)
    if kind=='rf_covariance':
        us=batch['ps']-batch['ps'].mean(2,keepdims=True)
        uq=batch['pq']-batch['pq'].mean(2,keepdims=True)
        denominator=np.sqrt(max(float(cfg['variance_scale']),1e-8)*us.shape[2])
        return us/denominator,uq/denominator
    if kind=='learned':
        state=np.load(source.folder/cfg['state_file'])
        d={k:state[k] for k in state.files}
        center=alg.feature_center(source.x[source.split['anchor_positions']],d)
        return alg.numpy_features(batch['xs'],d,center),alg.numpy_features(batch['xq'],d,center)
    return None


def fixed_controls(source,batch):
    path=source.folder/'fixed_selection.json'
    if path.exists():return read(path)
    selected={};trace=[]
    def score(label,cfg):
        value=metric(batch,prediction_curve(batch,cfg,configured_features(source,batch,cfg)))
        row=dict(method=label,config=cfg,source_validation=value)
        trace.append(row)
        return row
    for label in ('none','bayes','brier_mean','brier_strong'):selected[label]=score(label,dict(kind=label))
    candidates=[selected['none']]+[score('intercept',dict(kind='intercept',global_penalty=g,local_penalty=1.)) for g in PENALTIES]
    selected['intercept']=min(candidates,key=lambda r:r['source_validation']['alc'])
    g=selected['intercept']['config'].get('global_penalty',4096.)
    for label in ('probability_slope','demand_linear'):
        candidates=[selected['intercept']]+[score(label,dict(kind=label,global_penalty=g,local_penalty=l)) for l in PENALTIES[:6]]
        selected[label]=min(candidates,key=lambda r:r['source_validation']['alc'])
    candidates=[selected['intercept']]
    for length in (.5,1.,2.):
        for local in (4.,16.,64.):
            candidates.append(score('demand_rbf',dict(kind='demand_rbf',length=length,global_penalty=g,local_penalty=local)))
    selected['demand_rbf']=min(candidates,key=lambda r:r['source_validation']['alc'])
    # Variance scale depends only on available historical forecasts, not target labels.
    valid=np.flatnonzero(source.split['train_mask'])
    sample=valid[::max(1,len(valid)//2000)]
    variance=float(np.var(np.asarray(source.p[sample]),axis=1).mean())
    candidates=[selected['intercept']]+[score('rf_covariance',dict(kind='rf_covariance',variance_scale=variance,global_penalty=g,local_penalty=l)) for l in (1.,4.,16.,64.)]
    selected['rf_covariance']=min(candidates,key=lambda r:r['source_validation']['alc'])
    output=dict(selected=selected,trace=trace)
    save(path,output)
    print('FIXED',source.name,{k:round(v['source_validation']['alc'],6) for k,v in selected.items()},flush=True)
    return output


def learn(source,validation,rank,seed,global_penalty):
    import torch
    torch.set_num_threads(1)
    label=f'learned_r{rank}_s{seed}'
    result_path=source.folder/(label+'.json')
    if result_path.exists():return read(result_path)
    torch.manual_seed(seed);rng=np.random.default_rng(SEED+seed+rank)
    model=alg.make_network(rank)
    with torch.no_grad():model.log_global.fill_(np.log(global_penalty))
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.001)
    anchors=torch.tensor(source.x[source.split['anchor_positions']],dtype=torch.float32)
    best=float('inf');best_step=0;history=[];start=time.monotonic();actual_update=False
    initial=model.net[0].weight.detach().clone()
    state_file=label+'.npz'
    def validate(step):
        nonlocal best,best_step
        model.eval()
        state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
        center=alg.feature_center(source.x[source.split['anchor_positions']],state)
        us=alg.numpy_features(validation['xs'],state,center);uq=alg.numpy_features(validation['xq'],state,center)
        lg,ll=[float(v.detach()) for v in model.penalties()]
        cfg=dict(kind='learned',rank=rank,seed=seed,state_file=state_file,global_penalty=lg,local_penalty=ll)
        predictions={b:alg.ridge_features(us,uq,validation['base_s'],validation['base_q'],validation['labels'],b,lg,ll) for b in alg.BUDGETS}
        score=metric(validation,predictions)
        # Verify actual network and analytical NumPy readout agree at a complete context.
        with torch.no_grad():
            tensor=lambda k:torch.tensor(validation[k][:4],dtype=torch.float32)
            p=model.predict(tensor('xs'),tensor('xq'),anchors,tensor('base_s'),tensor('base_q'),tensor('labels'),31).numpy()
        parity=float(np.max(abs(p-predictions[31][:4])))
        assert parity<2e-5,parity
        row=dict(step=step,**score,torch_numpy_max_error=parity,feature_std=float(np.std(uq)),global_penalty=lg,local_penalty=ll)
        history.append(row)
        if score['alc']<best:
            best=score['alc'];best_step=step
            np.savez(source.folder/state_file,**state)
            save(result_path.with_suffix('.best.json'),dict(method=label,config=cfg,source_validation=score,best_step=step))
        status(stage='meta_training',protocol=source.name,method=label,step=step,total_steps=STEPS,source_validation_alc=score['alc'],best_source_alc=best)
        print('META',source.name,label,step,round(score['alc'],6),flush=True)
        model.train()
    validate(0)
    for step in range(1,STEPS+1):
        episodes=[source.episode(rng,'train') for _ in range(8)]
        batch=source.batch(episodes)
        b=int(rng.choice([1,3,7,15,31],p=np.array([.2,.2,.2,.2,.1])/.9))
        tensor=lambda k:torch.tensor(batch[k],dtype=torch.float32)
        pred=model.predict(tensor('xs'),tensor('xq'),anchors,tensor('base_s'),tensor('base_q'),tensor('labels'),b)
        loss=torch.square(pred-tensor('truth')).mean()
        assert torch.isfinite(loss)
        optimizer.zero_grad();loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step==1:
            actual_update=not torch.equal(initial,model.net[0].weight.detach())
            assert actual_update
        if step%100==0:validate(step)
    result=read(result_path.with_suffix('.best.json'))
    result.update(history=history,elapsed_seconds=time.monotonic()-start,actual_weight_update_verified=actual_update)
    save(result_path,result)
    return result


def source_run(name,idx,steps=None):
    group_oob(name,idx);source_prepare(name,idx)
    source=Source(name,idx);validation=source.validation()
    controls=fixed_controls(source,validation)
    learned=[]
    for rank in (4,8):
        for seed in (42,49):
            g=controls['selected']['intercept']['config'].get('global_penalty',256.)
            learned.append(learn(source,validation,rank,seed,g))
    candidates=[controls['selected'][n] for n in ('none','demand_linear','demand_rbf')]+learned
    chosen=min(candidates,key=lambda r:r['source_validation']['alc'])
    save(source.folder/'selection_frozen.json',dict(fixed=controls['selected'],learned=learned,selected_demand=chosen,
        source_split_sha256=sha(source.folder/'source_split.npz'),validation_episode_sha256=sha(source.folder/'validation_episodes.npz'),
        models={r['config']['state_file']:sha(source.folder/r['config']['state_file']) for r in learned}))
    del source,validation
    gc.collect()


def prediction_features(folder,demand,cfg,probability,experts):
    split=np.load(folder/'source_split.npz');kind=cfg['kind']
    if kind=='intercept':return demand[:,:0]
    if kind=='probability_slope':return ((probability-float(split['probability_mean']))/float(split['probability_scale']))[:,None]
    if kind=='demand_linear':return demand/np.sqrt(18)
    if kind=='rf_covariance':
        values=experts-experts.mean(1,keepdims=True)
        return values/np.sqrt(max(cfg['variance_scale'],1e-8)*experts.shape[1])
    if kind=='learned':
        z=np.load(folder/cfg['state_file']);state={k:z[k] for k in z.files}
        source=np.load(folder/'source_demands.npy',mmap_mode='r')
        center=alg.feature_center(source[split['anchor_positions']],state)
        return alg.numpy_features(demand,state,center)
    return None


def evaluate(plan,idx):
    # All model choices and source fits have been sealed; outer labels enter only now.
    frozen=dict(plan_sha256=sha(ROOT/'plan.json'),sources={p:sha(ROOT/p/'selection_frozen.json') for p in PROTOCOLS})
    save(ROOT/'prediction_freeze.json',frozen)
    outcomes=np.load(SWEEP/'development_outcomes.npy')
    raw_features=np.load(BASE/'features.npy',mmap_mode='r')[len(idx['fit']):,:18]
    family_dev=idx['families'][len(idx['fit']):];groups_dev=idx['groups'][len(idx['fit']):]
    records=[];initial_errors=[]
    for name in PROTOCOLS:
        folder=ROOT/name;selection=read(folder/'selection_frozen.json')
        split=np.load(folder/'source_split.npz');devpos=protocol_meta(name,idx)[3]
        expert=np.clip(np.load((BASE/'demand_manual' if name=='mixed' else COLD/name)/'predictions.npy'),1e-4,1-1e-4)
        assert len(expert)==len(devpos)
        p0=expert.mean(1);truth=outcomes[devpos]
        demand_raw=np.array(raw_features[devpos],copy=True)
        if name!='mixed':
            missing=np.load(COLD/'missing_development_demands.npy')[devpos]
            source_positions=split['source_positions']
            source_raw=np.load(BASE/'features.npy',mmap_mode='r')[source_positions,:18]
            demand_raw[missing]=np.median(source_raw,axis=0)
        demand=np.clip((demand_raw-split['demand_mean'])/split['demand_scale'],-5,5)
        f=family_dev[devpos];groups=groups_dev[devpos]
        if name=='mixed':
            from scripts.coordinate_brier_ablation_v42 import support_orders
            orders=support_orders(f,groups)
        else:
            z=np.load(COLD/name/'orders.npz');orders=list(zip(z['support'],z['evaluation']))
        chosen={k:v['config'] for k,v in selection['fixed'].items()}
        chosen.update({v['method']:v['config'] for v in selection['learned']})
        chosen['selected_demand']=selection['selected_demand']['config']
        chosen['selected_demand_shuffled']=selection['selected_demand']['config']
        # Label-free demand permutation within benchmark and baseline-probability deciles.
        permutation=np.arange(len(demand));rng=np.random.default_rng(SEED+99)
        for family in np.unique(f):
            rows=np.flatnonzero(f==family)
            order=rows[np.argsort(p0[rows],kind='stable')]
            for chunk in np.array_split(order,10):
                if len(chunk)>1:permutation[chunk]=rng.permutation(chunk)
        save_array(folder/'evaluation_demand_permutation.npy',permutation)
        for method,cfg in chosen.items():
            d=demand[permutation] if method.endswith('_shuffled') else demand
            features=prediction_features(folder,d,cfg,p0,expert)
            for seed,(support,evaluation) in enumerate(orders):
                # One episode per target configuration; no query labels enter the readout.
                batch=dict(base_s=np.broadcast_to(p0[support],(24,31)),base_q=np.broadcast_to(p0[evaluation],(24,len(evaluation))),
                    labels=truth[support].T,truth=truth[evaluation].T,
                    xs=np.broadcast_to(d[support],(24,31,18)),xq=np.broadcast_to(d[evaluation],(24,len(evaluation),18)))
                if cfg['kind'] in ('bayes','brier_mean','brier_strong'):
                    batch.update(ps=np.broadcast_to(expert[support],(24,31,192)),pq=np.broadcast_to(expert[evaluation],(24,len(evaluation),192)))
                feature_pair=None if features is None else (np.broadcast_to(features[support],(24,31,features.shape[1])),np.broadcast_to(features[evaluation],(24,len(evaluation),features.shape[1])))
                predictions=prediction_curve(batch,cfg,feature_pair)
                for budget,p in predictions.items():
                    error=np.square(p-truth[evaluation].T)
                    for family in np.unique(f):
                        mask=f[evaluation]==family
                        if not mask.any():continue
                        per_model=error[:,mask].mean(1)
                        for h,value in zip(idx['held'],per_model):records.append((name,method,family,seed,budget,int(h),float(value)))
        print('EVALUATED',name,flush=True)
    with (ROOT/'per_model.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['protocol','method','family','seed','budget','model','brier']);w.writerows(records)
    summaries={}
    for group in ('mixed','cold_primary','cold_all_secondary'):
        chosen=[r for r in records if (r[0]=='mixed' if group=='mixed' else r[0]!='mixed') and (group!='cold_primary' or r[2]!='MuSR')]
        rows=[]
        for method in sorted(set(r[1] for r in chosen)):
            relevant=[r for r in chosen if r[1]==method]
            curve={str(b):float(np.mean([r[6] for r in relevant if r[4]==b])) for b in alg.BUDGETS}
            per={family:float(np.mean([r[6] for r in relevant if r[2]==family and r[4]==31])) for family in sorted(set(r[2] for r in relevant))}
            rows.append(dict(method=method,brier31=curve['31'],alc=float(np.array(list(curve.values()))@alg.ALC),curve=curve,per_family_brier31=per))
        summaries[group]=rows
    old=read(BASE/'result.json')
    expected=next(s for s in old['summaries'] if s['cohort']=='full_development3241' and s['method']=='demand_manual_bayes_mixture')
    got=next(s for s in summaries['mixed'] if s['method']=='bayes')
    assert abs(got['brier31']-expected['brier31'])<1e-12 and abs(got['alc']-expected['alc'])<1e-12
    cold_expected=next(s for s in read(COLD/'result.json')['primary'] if s['method']=='uniform_bayes')
    got=next(s for s in summaries['cold_primary'] if s['method']=='bayes')
    assert abs(got['brier31']-cold_expected['brier31'])<1e-12 and abs(got['alc']-cold_expected['alc'])<1e-12
    save(ROOT/'result.json',dict(completed=True,summaries=summaries,plan=plan,baseline_exactly_reproduced=True))
    status(stage='complete',records=len(records))
    for group,rows in summaries.items():
        for r in rows:print('RESULT',group,r['method'],r['brier31'],r['alc'],flush=True)


def smoke():
    import torch
    torch.set_num_threads(1)
    rng=np.random.default_rng(48)
    xs=rng.normal(size=(3,31,4));xq=rng.normal(size=(3,12,4))
    bs=rng.uniform(.1,.9,(3,31));bq=rng.uniform(.1,.9,(3,12));ys=rng.integers(0,2,(3,31))
    ks=xs@xs.transpose(0,2,1);kq=xq@xs.transpose(0,2,1)
    for b in alg.BUDGETS:
        a=alg.ridge_features(xs,xq,bs,bq,ys,b,16.,4.)
        z=alg.ridge_kernel(ks,kq,bs,bq,ys,b,16.,4.)
        assert np.max(abs(a-z))<1e-12
    order=rng.permutation(31)
    a=alg.ridge_features(xs,xq,bs,bq,ys,31,16.,4.)
    z=alg.ridge_features(xs[:,order],xq,bs[:,order],bq,ys[:,order],31,16.,4.)
    assert np.max(abs(a-z))<1e-12
    model=alg.make_network(4);anchor=torch.randn(32,18);s=torch.randn(3,31,18);q=torch.randn(3,12,18)
    tb=torch.tensor(bs,dtype=torch.float32);tq=torch.tensor(bq,dtype=torch.float32);ty=torch.tensor(ys,dtype=torch.float32)
    before=model.net[0].weight.detach().clone();optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    p=model.predict(s,q,anchor,tb,tq,ty,31);loss=torch.square(p-torch.rand_like(p)).mean();loss.backward()
    assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
    optimizer.step();assert not torch.equal(before,model.net[0].weight.detach())
    state={k:v.detach().numpy().copy() for k,v in model.state_dict().items()}
    center=alg.feature_center(anchor.numpy(),state)
    us=alg.numpy_features(s.numpy(),state,center);uq=alg.numpy_features(q.numpy(),state,center)
    lg,ll=[float(v.detach()) for v in model.penalties()]
    expected=alg.ridge_features(us,uq,bs,bq,ys,31,lg,ll)
    actual=model.predict(s,q,anchor,tb,tq,ty,31).detach().numpy()
    assert np.max(abs(expected-actual))<2e-5
    # Whole-group OOB excludes a sibling even when that sibling row was not drawn.
    groups=np.array([0,0,1,2,2]);sample=np.array([0,2,2,2,0])
    present=np.bincount(groups[sample],minlength=3)>0
    assert np.array_equal(np.flatnonzero(~present[groups]),[3,4])
    print('SMOKE PASS: primal/dual parity, support permutation invariance, finite gradients/actual update, NumPy forward parity, group OOB',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['smoke','prepare','run']);args=parser.parse_args()
    import psutil
    psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    if args.action=='smoke':
        with threadpool_limits(1):smoke()
        return
    ROOT.mkdir(parents=True,exist_ok=True)
    import msvcrt
    with (ROOT/'worker.lock').open('a+b') as lock:
        if lock.tell()==0:lock.write(b'0');lock.flush()
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        save(ROOT/'worker.json',dict(pid=os.getpid(),started=time.time(),action=args.action))
        try:
            with threadpool_limits(1):
                plan=plan_prepare()
                if args.action=='prepare':return
                if (ROOT/'result.json').exists():print('ALREADY COMPLETE',flush=True);return
                idx=np.load(BASE/'indices_and_weights.npz')
                for name in PROTOCOLS:source_run(name,idx)
                evaluate(plan,idx)
        except BaseException as exc:
            save(ROOT/'error.json',dict(error=repr(exc),traceback=traceback.format_exc()))
            raise


if __name__=='__main__':main()

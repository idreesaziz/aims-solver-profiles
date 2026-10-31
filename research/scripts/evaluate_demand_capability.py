"""Read-only model evaluation plus budget-faithful held-out-model simulation."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,time,csv,math
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.analyze_delean_shapes import save,digest
from scripts.evaluate_delean_shape_predictors import mapped_splits
from models.individual_behavior import score
from models.delean_classification_metrics import classification_metrics,mean_defined


def offset_update(mean,covariance,features,offset,outcome):
    p=float(expit(offset+features@mean));v=covariance@features
    new=covariance-(p*(1-p)/(1+p*(1-p)*(features@v)))*np.outer(v,v)
    return mean+new@features*(outcome-p),(new+new.T)/2


def representation(path):
    with np.load(path) as saved:
        q=saved['coordinates'].astype(float);a=saved['ability'].astype(float);u=saved['model_vectors'].astype(float)
    d=u.shape[1];features=np.c_[np.ones(len(q)),q[:,1:]/math.sqrt(d)] if d else np.ones((len(q),1))
    theta=np.c_[a,u];mean=theta.mean(0);emp=np.cov(theta,rowvar=False)
    emp=np.atleast_2d(emp);cov=.75*emp+.25*np.diag(np.diag(emp))+.01*np.eye(len(mean));cov[0,0]=max(cov[0,0],.25)
    scalar=np.zeros_like(cov);scalar[0,0]=cov[0,0]
    return dict(features=features,offset=q[:,0],mean=mean,cov=cov,scalar=scalar,theta=theta)


def main():
    start=time.time();c=json.loads((ROOT/'config_demand_capability_v1.json').read_text());out=ROOT/c['output'];report=ROOT/c['report']
    prepared=json.loads((out/'prepared.json').read_text());historical=np.array(prepared['historical_columns']);held=np.array(prepared['heldout_model_columns']);known=np.array(prepared['known_report_columns'])
    save(out/'evaluation_prepared.json',dict(script_hash=digest(Path(__file__)),config=c,
        target_models=held.tolist(),historical_models=historical.tolist(),prior_policy='Fixed covariance shrinkage; no held-out labels tune parameters.'))
    assert not set(historical)&set(held)
    for filename,expected in prepared['hashes'].items():assert digest(ROOT/filename)==expected,filename
    source=ROOT/c['source'];x=np.load(source/'x.npy');y=np.load(source/'y.npy');f=np.load(source/'families.npy');keep=np.load(source/'snapshot_indices.npy')
    splits=mapped_splits(ROOT/c['split_source'],keep)
    if not all((out/s['name']/'complete.json').exists() for s in splits):raise RuntimeError('All six folds required before aggregate evaluation')
    known_rows=[];banks={}
    for split in splits:
        fold=out/split['name']
        with np.load(fold/'predictions.npz') as saved:
            te=saved['test_indices'];methods=[k for k in saved.files if k!='test_indices']
            for family in np.unique(f[te]):
                mask=f[te]==family;target=y[te[mask]][:,known]
                errors={name:(saved[name][mask]-target)**2 for name in methods}
                banks.setdefault(split['protocol'],[]).append(errors)
                for name in methods:
                    pp=saved[name][mask]
                    for j,model in enumerate(known):
                        w=np.ones(len(target))/len(target)
                        known_rows.append(dict(protocol=split['protocol'],family=str(family),model_column=int(model),method=name,questions=len(target),
                            **score(pp[:,j],target[:,j],w),**classification_metrics(pp[:,j],target[:,j],w)))
    with (report/'known_per_model.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(known_rows[0]));writer.writeheader();writer.writerows(known_rows)
    known_summary=[];rng=np.random.default_rng(c['seed']+600)
    for protocol,pairs in banks.items():
        for name in pairs[0]:
            differences=[r[name]-r['shared_selected_calibrated'] for r in pairs]
            boot=[]
            for rep in range(c['bootstrap_repetitions']):boot.append(float(np.mean([d[rng.integers(0,len(d),len(d))].mean() for d in differences])))
            relevant=[r for r in known_rows if r['protocol']==protocol and r['method']==name]
            metrics={key:mean_defined(relevant,key) for key in ['auc','ece_10_equal_width','precision','recall','f1']}
            metrics.update({key+'_defined_pairs':sum(r[key] is not None for r in relevant) for key in ['precision','recall','f1']})
            known_summary.append(dict(protocol=protocol,method=name,brier=float(np.mean([r[name].mean() for r in pairs])),
                difference_vs_shared=float(np.mean([d.mean() for d in differences])),paired_question_ci=np.quantile(boot,[.025,.975]).tolist(),**metrics))
    save(report/'known.json',dict(results=known_summary,models=known.tolist(),history_models=historical.tolist(),heldout_models=held.tolist()))
    methods=['beta_mean','shared_scalar','bce_scalar','bce_capability','joint_scalar','joint_capability','historical_mixture']
    adaptation_rows=[];error_banks={}
    for split in splits:
        print('ADAPTATION',split['name'],flush=True)
        fold=out/split['name'];selection=json.loads((fold/'selection.json').read_text())['choices']
        reps={name:representation(fold/f"{name}_d{selection[name]['dimension']}_representation.npz") for name in ['shared','bce','joint']}
        source_ids=split['refit'];source_weights=np.zeros(len(source_ids))
        for fam in np.unique(f[source_ids]):source_weights[f[source_ids]==fam]=1/(len(np.unique(f[source_ids]))*np.sum(f[source_ids]==fam))
        source_rate=float(source_weights@y[source_ids][:,historical].mean(1))
        for family in np.unique(f[split['test']]):
            ids=split['test'][f[split['test']]==family]
            losses=np.full((c['adaptation_seeds'],len(ids),len(c['budgets']),len(methods)),np.nan,dtype=np.float32)
            local={int(v):j for j,v in enumerate(ids)}
            for seed in range(c['adaptation_seeds']):
                rng=np.random.default_rng(c['seed']+seed+sum(map(ord,split['name']+str(family))))
                order=rng.permutation(ids);acquisition=order[:len(ids)//2];evaluation=order[len(ids)//2:];support=acquisition[:31]
                assert len(support)==31 and not set(acquisition)&set(evaluation)
                ev=np.array([local[int(v)] for v in evaluation]);running=np.zeros((len(evaluation),len(c['budgets']),len(methods)))
                p_hist=expit(reps['bce']['offset'][evaluation,None]+reps['bce']['features'][evaluation]@reps['bce']['theta'].T)
                for model in held:
                    means={};covs={}
                    for name in methods[1:-1]:
                        base=name.split('_')[0];r=reps[base];means[name]=r['mean'].copy();covs[name]=r['scalar' if name.endswith('scalar') else 'cov'].copy()
                    logweights=np.full(len(historical),-np.log(len(historical)));successes=0;target=y[evaluation,model]
                    for budget in range(32):
                        if budget in c['budgets']:
                            j=c['budgets'].index(budget)
                            probabilities={}
                            for name in means:
                                r=reps[name.split('_')[0]]
                                probabilities[name]=expit(r['offset'][evaluation]+r['features'][evaluation]@means[name])
                            probabilities['beta_mean']=np.full(len(evaluation),(2*source_rate+successes)/(2+budget))
                            probabilities['historical_mixture']=p_hist@np.exp(logweights-logsumexp(logweights))
                            for k,name in enumerate(methods):
                                error=(probabilities[name]-target)**2;running[:,j,k]+=error/len(held)
                                adaptation_rows.append(dict(protocol=split['protocol'],family=str(family),model_column=int(model),seed=seed,budget=budget,method=name,brier=float(error.mean()),evaluation_questions=len(evaluation)))
                        if budget==31:break
                        i=support[budget];outcome=int(y[i,model]);successes+=outcome
                        for name in means:
                            r=reps[name.split('_')[0]];means[name],covs[name]=offset_update(means[name],covs[name],r['features'][i],r['offset'][i],outcome)
                        r=reps['bce'];pr=np.clip(expit(r['offset'][i]+r['theta']@r['features'][i]),1e-4,1-1e-4)
                        logweights+=np.log(pr if outcome else 1-pr)
                losses[seed,ev]=running
            np.savez_compressed(fold/(str(family)+'_adaptation_errors.npz'),losses=losses,indices=ids)
            error_banks.setdefault(split['protocol'],[]).append(losses)
    with (report/'adaptation_per_model.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(adaptation_rows[0]));writer.writeheader();writer.writerows(adaptation_rows)
    summary=[];rng=np.random.default_rng(c['seed']+800)
    def ci_for(banks,a,b,budget=None):
        diffs=[]
        for bank in banks:
            difference=bank[:,:,:,a]-bank[:,:,:,b]
            if budget is None:difference=(difference*np.array(c['alc_weights'])).sum(2)
            else:difference=difference[:,:,budget]
            diffs.append(difference)
        boot=[]
        for rep in range(c['bootstrap_repetitions']):
            ss=rng.integers(0,c['adaptation_seeds'],c['adaptation_seeds']);values=[]
            for d in diffs:
                q=rng.integers(0,d.shape[1],d.shape[1]);values.append(np.nanmean(d[ss][:,q]))
            boot.append(float(np.mean(values)))
        return float(np.mean([np.nanmean(d) for d in diffs])),np.quantile(boot,[.025,.975]).tolist()
    for protocol,banklist in error_banks.items():
        for k,name in enumerate(methods):
            scores=[float(np.mean([np.nanmean(bank[:,:,j,k]) for bank in banklist])) for j in range(len(c['budgets']))]
            delta,ci=ci_for(banklist,k,methods.index('shared_scalar'))
            summary.append(dict(protocol=protocol,method=name,brier_by_budget=dict(zip(map(str,c['budgets']),scores)),
                brier_alc=float(np.array(c['alc_weights'])@scores),alc_difference_vs_shared=delta,paired_question_seed_ci=ci))
        for a,b in [('bce_capability','bce_scalar'),('joint_capability','joint_scalar')]:
            delta,ci=ci_for(banklist,methods.index(a),methods.index(b))
            summary.append(dict(protocol=protocol,comparison=a+' minus '+b,alc_difference=delta,paired_question_seed_ci=ci))
    save(report/'adaptation.json',dict(results=summary,
        protocol='All 24 target model columns absent from teacher/encoder/tuning/calibration; random target acquisition/evaluation pools 50/50, 5 seeds, budgets 0/1/3/7/15/31. No held-out target outcomes except acquired labels used in prediction. Related fine-tunes remain among 192 historical models.',
        scoring='Official budget locations and normalized trapezoidal ALC weights reproduced locally; not official leaderboard scoring, which can have repeated responses/configurations and different hidden benchmarks.'))
    save(out/'evaluation_complete.json',dict(stage='complete',seconds=time.time()-start,known_records=len(known_rows),adaptation_records=len(adaptation_rows)))
    print('EVALUATION COMPLETE',time.time()-start,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

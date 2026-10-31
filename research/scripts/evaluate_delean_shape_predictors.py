"""Held-out tests of inductive diffusion coordinates and Bayesian profile adaptation."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,time,csv
from pathlib import Path
import numpy as np
from scipy.special import expit,logsumexp
from scipy.spatial.distance import cdist
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.delean_shape import (fit_diffusion_basis,transform_diffusion,fit_logistic_multi,
    logistic_predictions,laplace_update)
from models.individual_behavior import fit_calibration,calibrate
from scripts.analyze_delean_shapes import save,digest


def weights(f):
    w=np.zeros(len(f))
    for family in np.unique(f):w[f==family]=1/(len(np.unique(f))*np.sum(f==family))
    return w


def mapped_splits(base,keep):
    lookup={int(old):new for new,old in enumerate(keep)}
    splits=json.loads((base/'splits.json').read_text())
    result=[]
    for original in splits:
        s=dict(original)
        for role in ['fit','tune','calibrate','refit','test']:
            s[role]=np.array([lookup[i] for i in original[role] if i in lookup],dtype=int)
        for a in ['fit','tune','calibrate']:
            assert not set(s[a])&set(s['test'])
        assert not set(s['refit'])&set(s['calibrate'])
        result.append(s)
    return result


def classification(y,p):
    pred=p>=.5;tp=np.sum(pred&(y==1));fp=np.sum(pred&(y==0));fn=np.sum(~pred&(y==1))
    return dict(precision=float(tp/(tp+fp)) if tp+fp else None,
        recall=float(tp/(tp+fn)) if tp+fn else None,f1=float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else None)


def known_heads(c,out,report,x,y,f,keep,pilot):
    base=ROOT/c['source'];splits=mapped_splits(base,keep);summaries=[]
    for split in splits:
        print('PREDICTORS',split['name'],flush=True)
        checkpoint=out/(split['name']+'_prediction.npz')
        histfile=out/(split['name']+'_historical.npz')
        if checkpoint.exists() and histfile.exists():continue
        tr,tu,ca,rf,te=[split[k] for k in ['fit','tune','calibrate','refit','test']]
        wtr,wtu,wca,wrf=[weights(f[ix]) for ix in [tr,tu,ca,rf]]
        fitbasis=fit_diffusion_basis(x[tr],c['inductive_landmarks'],48,c['seed'])
        zfit=transform_diffusion(fitbasis,x)
        choices={name:dict(loss=np.full(len(pilot),np.inf),dimension=np.zeros(len(pilot),dtype=int),penalty=np.zeros(len(pilot)))
                 for name in ['raw_logistic','diffusion_logistic']}
        tuning=[]
        for name,data,dimensions in [('raw_logistic',x,[18]),('diffusion_logistic',zfit,c['prediction_dimensions'])]:
            for dimension in dimensions:
                for penalty in c['prediction_penalties']:
                    b,optimization=fit_logistic_multi(data[tr,:dimension],y[tr][:,pilot],wtr,penalty)
                    p=logistic_predictions(data[tu,:dimension],b)
                    loss=wtu@((p-y[tu][:,pilot])**2)
                    improved=loss<choices[name]['loss']
                    choices[name]['loss'][improved]=loss[improved]
                    choices[name]['dimension'][improved]=dimension
                    choices[name]['penalty'][improved]=penalty
                    tuning.append(dict(name=name,dimension=dimension,penalty=penalty,mean_tune_brier=float(loss.mean()),**optimization))
        finalbasis=fit_diffusion_basis(x[rf],c['inductive_landmarks'],48,c['seed'])
        z=transform_diffusion(finalbasis,x)
        predictions={};evidence=np.r_[rf,ca]
        rate=weights(f[evidence])@y[evidence][:,pilot]
        own=np.tile(rate,(len(te),1))
        if split['protocol']=='within_benchmark':
            for family in np.unique(f[te]):
                ids=evidence[f[evidence]==family]
                own[f[te]==family]=(y[ids][:,pilot].sum(0)+1)/(len(ids)+2)
        predictions['own_mean']=own
        for name,data in [('raw_logistic',x),('diffusion_logistic',z)]:
            raw=np.zeros((len(te),len(pilot)));calibrated=raw.copy()
            for dimension in np.unique(choices[name]['dimension']):
                for penalty in np.unique(choices[name]['penalty']):
                    cols=np.flatnonzero((choices[name]['dimension']==dimension)&(choices[name]['penalty']==penalty))
                    if not len(cols):continue
                    b,_=fit_logistic_multi(data[rf,:dimension],y[rf][:,np.array(pilot)[cols]],wrf,float(penalty))
                    pc=logistic_predictions(data[ca,:dimension],b);pt=logistic_predictions(data[te,:dimension],b)
                    for local,col in enumerate(cols):
                        parameters=fit_calibration(pc[:,local],y[ca,pilot[col]],wca,.05)
                        raw[:,col]=pt[:,local];calibrated[:,col]=calibrate(pt[:,local],parameters)
            predictions[name]=raw;predictions[name+'_calibrated']=calibrated
        # A local empirical smoother, source-only neighbour labels and fixed shrinkage.
        for name,data in [('raw_knn',x),('diffusion_knn',z[:,:24])]:
            p=np.zeros((len(te),len(pilot)))
            for start in range(0,len(te),128):
                ids=te[start:start+128];dist=cdist(data[ids],data[evidence])
                nn=np.argpartition(dist,19,axis=1)[:,:20]
                p[start:start+len(ids)]=(y[evidence[nn]][:,:,pilot].sum(1)+20*rate)/(20+20)
            predictions[name]=p
        # Historical demand functions for subsequent leave-one-recorded-model adaptation.
        b,optimization=fit_logistic_multi(z[rf,:24],y[rf],wrf,.1)
        pall=logistic_predictions(z[:,:24],b)
        np.savez_compressed(histfile,coefficients=b,z=z[:,:24].astype(np.float32),predictions=pall.astype(np.float32),
            training_rate=(wrf@y[rf]),test_indices=te)
        for name,p in predictions.items():assert p.shape==(len(te),len(pilot)) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),name
        np.savez_compressed(checkpoint,**predictions,test_indices=te)
        save(out/(split['name']+'_tuning.json'),dict(choices={name:{k:v.tolist() for k,v in val.items()} for name,val in choices.items()},
            tuning=tuning,historical_fit=optimization,basis_fit_indices_hash=__import__('hashlib').sha256(rf.tobytes()).hexdigest()))
        print('PREDICTORS complete',split['name'],flush=True)
    # Fixed cohort, all complete folds required; joint question resampling across subjects.
    predictions_by_protocol={};rows=[]
    for split in splits:
        with np.load(out/(split['name']+'_prediction.npz')) as data:
            te=data['test_indices'];names=[name for name in data.files if name!='test_indices']
            for family in np.unique(f[te]):
                mask=f[te]==family;target=y[te[mask]][:,pilot]
                errors={name:(data[name][mask]-target)**2 for name in names}
                predictions_by_protocol.setdefault(split['protocol'],[]).append((family,errors))
                for name in names:
                    pp=data[name][mask]
                    for col,model in enumerate(pilot):
                        rows.append(dict(protocol=split['protocol'],family=str(family),method=name,model_column=model,
                            questions=len(target),brier=float(errors[name][:,col].mean()),
                            auc=float(roc_auc_score(target[:,col],pp[:,col])) if len(np.unique(target[:,col]))==2 else None,
                            **classification(target[:,col],pp[:,col])))
    summary=[];rng=np.random.default_rng(c['seed'])
    for protocol,pairs in predictions_by_protocol.items():
        for name in pairs[0][1]:
            score=float(np.mean([errors[name].mean() for _,errors in pairs]))
            delta=float(np.mean([(errors[name]-errors['own_mean']).mean() for _,errors in pairs]))
            boot=[]
            for rep in range(c['bootstrap_repetitions']):
                boot.append(np.mean([(errors[name]-errors['own_mean'])[rng.integers(0,len(errors[name]),len(errors[name]))].mean() for _,errors in pairs]))
            summary.append(dict(protocol=protocol,method=name,brier=score,delta_vs_own_mean=delta,
                paired_question_ci=np.quantile(boot,[.025,.975]).tolist(),models=len(pilot),pairs=len(pilot)*len(pairs)))
    save(report/'prediction.json',dict(results=summary,
        geometry='Inductive source-only landmark diffusion basis, refitted on source refit questions. New target features affect only their own out-of-sample transform.',
        selection='Dimensions and penalties chosen independently per known model using source tune Brier. Separate calibration labels excluded from final head/basis fitting.',
        scope='Known historical subjects. Equal benchmark means on one representative per stem. Intervals conditional on fitted heads and subjects. Not official Brier ALC.'))
    with (report/'prediction_per_model.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return splits


def adaptation(c,out,report,x,y,f,pilot,splits):
    rows=[]
    for split in splits:
        print('BAYESIAN',split['name'],flush=True)
        with np.load(out/(split['name']+'_historical.npz')) as saved:
            coefficients=saved['coefficients'];z=saved['z'];p=saved['predictions'];rates=saved['training_rate'];te=saved['test_indices']
        features=np.c_[np.ones(len(z)),z]
        for family in np.unique(f[te]):
            ids=te[f[te]==family]
            if len(ids)<40:raise ValueError('Insufficient target questions')
            for seed in range(c['adaptation_seeds']):
                rng=np.random.default_rng(c['seed']+seed+sum(map(ord,split['name']+str(family))))
                order=rng.permutation(ids);support=order[:32];evaluation=order[32:]
                assert not set(support)&set(evaluation)
                pe=p[evaluation].astype(float)
                gram=(pe.T@pe)/len(pe);cross=(pe.T@y[evaluation][:,pilot])/len(pe)
                target_rates=y[evaluation][:,pilot].mean(0)
                for col,model in enumerate(pilot):
                    historical=np.arange(y.shape[1])!=model
                    b=coefficients[:,historical]
                    mean=b.mean(1);empirical=np.cov(b)
                    covariance=.75*empirical+.25*np.diag(np.diag(empirical))+.01*np.eye(len(mean))
                    covariance[0,0]=max(covariance[0,0],.25)
                    prior_rate=float(np.clip(rates[historical].mean(),.001,.999))
                    logweights=np.full(y.shape[1],-np.inf);logweights[historical]=-np.log(historical.sum())
                    successes=0
                    for budget in range(33):
                        if budget in c['adaptation_budgets']:
                            pb=expit(features[evaluation]@mean)
                            gaussian=float(np.mean((pb-y[evaluation,model])**2))
                            mixture_weights=np.exp(logweights-logsumexp(logweights))
                            mixture=float(mixture_weights@gram@mixture_weights-2*mixture_weights@cross[:,col]+target_rates[col])
                            beta=(2*prior_rate+successes)/(2+budget)
                            beta_brier=float(np.mean((beta-y[evaluation,model])**2))
                            for name,value in [('gaussian_capability',gaussian),('historical_mixture',mixture),('beta_mean',beta_brier)]:
                                rows.append(dict(protocol=split['protocol'],family=str(family),model_column=model,seed=seed,
                                    budget=budget,method=name,brier=value,evaluation_questions=len(evaluation)))
                        if budget==32:break
                        i=support[budget];outcome=int(y[i,model]);successes+=outcome
                        mean,covariance=laplace_update(mean,covariance,features[i],outcome)
                        likelihood=np.clip(p[i],1e-4,1-1e-4)
                        logweights+=np.log(likelihood if outcome else 1-likelihood)
    with (report/'adaptation_per_model.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    results=[]
    for protocol in sorted(set(r['protocol'] for r in rows)):
        for method in ['beta_mean','historical_mixture','gaussian_capability']:
            for budget in c['adaptation_budgets']:
                values=[r['brier'] for r in rows if r['protocol']==protocol and r['method']==method and r['budget']==budget]
                results.append(dict(protocol=protocol,method=method,budget=budget,brier=float(np.mean(values)),rows=len(values)))
    save(report/'adaptation.json',dict(results=results,
        protocol='Leave one recorded historical model out of prior; all its source outcomes excluded from prior statistics. Source question labels train historical functions; target benchmark outcomes excluded from those fits. Random 32-question support pool, disjoint common evaluation questions for all budgets, 5 seeds.',
        caveat='Related fine-tunes may remain among 215 historical models. This does not establish transfer to new model families. Gaussian updates are an online Laplace approximation; covariance shrinkage is fixed, not target tuned. No active acquisition or official evaluator.'))


def main():
    started=time.time();c=json.loads((ROOT/'config_delean_shape_v1.json').read_text());out=ROOT/c['output'];report=ROOT/c['report']
    prepared=json.loads((out/'prepared.json').read_text())
    for filename,expected in prepared['hashes'].items():assert digest(ROOT/filename)==expected,filename
    x=np.load(out/'x.npy');y=np.load(out/'y.npy');f=np.load(out/'families.npy');keep=np.load(out/'snapshot_indices.npy');pilot=prepared['pilot_models']
    save(out/'prediction_prepared.json',dict(script_hash=digest(Path(__file__)),config=c,models=pilot,
        methods=['own_mean','raw_logistic','diffusion_logistic','raw_knn','diffusion_knn','gaussian_capability','historical_mixture','beta_mean']))
    splits=known_heads(c,out,report,x,y,f,keep,pilot)
    adaptation(c,out,report,x,y,f,pilot,splits)
    save(out/'prediction_completion.json',dict(stage='complete',seconds=time.time()-started,
        peak_rss_mb=__import__('psutil').Process().memory_info().peak_wset/2**20))
    print('PREDICTION/ADAPTATION COMPLETE',time.time()-started,flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()

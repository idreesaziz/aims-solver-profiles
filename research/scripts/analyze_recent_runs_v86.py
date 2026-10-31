"""Descriptive public-development diagnostics only; never tunes/submits a model."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
import sys,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import broad_prior_audit_v81 as audit
OUT=ROOT/'reports/recent_run_diagnosis_v86';OUT.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2),encoding='utf-8')

def main():
    plan=read(audit.OUT/'plan.json')
    frame=pd.DataFrame(read(audit.OUT/'v81_validation_records.json'))
    weights={0:.1,1:.2,3:.2,7:.2,15:.2,31:.1}
    frame['weighted']=frame.brier*frame.budget.map(weights)
    keys=['benchmark','cohort','seed','pair']
    paired=frame.groupby(keys+['method']).weighted.sum().unstack('method')
    ci={};rng=np.random.default_rng(860)
    for size in [1,3]:
        sub=paired.xs(size,level='cohort');delta=sub.neutral-sub.drift
        groups=np.array([plan['groups'][b] for b in sub.index.get_level_values('benchmark')])
        unique=np.unique(groups);sums=np.array([delta.to_numpy()[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
        picks=rng.integers(len(unique),size=(10000,len(unique)))
        boots=sums[picks].sum(1)/counts[picks].sum(1)
        ci[str(size)]=dict(neutral_minus_drift=float(delta.mean()),interval95=np.quantile(boots,[.025,.975]).tolist(),groups=len(unique),pair_episodes=len(delta),source_wins=int((delta.groupby(level='benchmark').mean()<0).sum()))
    meta,arr=audit.drift._load_art(str(audit.T.OUT/'v81_validation'))
    prior=audit.drift.Predictor(meta,arr)
    diag=[];overlaps=[]
    for number,b in enumerate(plan['validation']):
        for size in [1,3]:
            for seed in [810,811]:
                ep=audit.distinct_subjects(read(audit.OUT/'episodes'/f'{b}_{size}_{seed}.json'))
                for j,pair in enumerate(ep['pairs']):
                    n=np.array(pair['n'],float);k=np.array(pair['k'],float);w=n/n.sum();y=k/n
                    z=np.array([prior.b+prior.zs(pair['subject'])+prior.zi(ep['items'][key]) for key in pair['query']])
                    p=expit(z);mu=float(w@y);mp=float(w@p)
                    variance=float(w@(p-mp)**2);cov=float(w@((p-mp)*(y-mu)));bias=(mp-mu)**2
                    brier=float(w@(y*(1-p)**2+(1-y)*p**2));u=mu*(1-mu)
                    assert abs(brier-(u+variance-2*cov+bias))<1e-12
                    auc=None
                    if k.sum()>0 and (n-k).sum()>0:
                        auc=float(roc_auc_score(np.r_[np.ones(len(k)),np.zeros(len(k))],np.r_[p,p],sample_weight=np.r_[k,n-k]))
                    diag.append(dict(benchmark=b,cohort=size,seed=seed,pair=j,brier=brier,uncertainty=u,mean_bias_squared=bias,prediction_variance=variance,twice_covariance=2*cov,centered_excess=variance-2*cov,query_rate_privileged=mu,mean_prediction=mp,auroc=auc,logit_std=float(np.std(z))))
                    for budget in weights:
                        covered={i for a,other in enumerate(ep['pairs']) if a!=j for i in other['support'][:budget]}
                        mask=np.array([key in covered for key in pair['query']])
                        overlaps.append(dict(benchmark=b,cohort=size,actual_cohort=len(ep['pairs']),seed=seed,pair=j,budget=budget,other_solver_query_coverage=float(w@mask),unweighted_coverage=float(mask.mean())))
        if number%10==0:print('DIAG',number,b,flush=True)
    d=pd.DataFrame(diag);o=pd.DataFrame(overlaps)
    d.to_csv(OUT/'public_prior_decomposition.csv',index=False);o.to_csv(OUT/'public_overlap.csv',index=False)
    columns=['brier','uncertainty','mean_bias_squared','prediction_variance','twice_covariance','centered_excess']
    summary={}
    for size in [1,3]:
        t=d[d.cohort==size]
        summary[str(size)]={**{c:float(t[c].mean()) for c in columns},'pairs':len(t),'negative_covariance_pair_fraction':float((t.twice_covariance<0).mean()),'mixed_outcome_pairs':int(t.auroc.notna().sum()),'mean_auroc_mixed_pairs':float(t.auroc.mean()),'median_auroc_mixed_pairs':float(t.auroc.median()),'mean_logit_std':float(t.logit_std.mean()),'overlap':{str(n):float(o[(o.cohort==size)&(o.budget==n)].other_solver_query_coverage.mean()) for n in weights}}
    official=read(ROOT/'reports/finalization_v82/official_budget_means.json')
    officialcurve={int(k):v for k,v in official['curve'].items()}
    # Pure counterfactual arithmetic, not evaluated replacement predictions.
    headroom={str(n):weights[n]*(officialcurve[n]-.17) for n in weights}
    result=dict(scope='Post-hoc analysis of inspected PUBLIC validation; no new candidate fitted or evaluated. Query-derived rate/bias terms are diagnostics and cannot be deployed.',neutral_vs_drift_cluster_bootstrap=ci,prior_decomposition=summary,hosted_arithmetic=dict(current_alc=official['brier_alc'],replace_only_B0_with_half=official['brier_alc']+weights[0]*(.25-officialcurve[0]),needed_gain_for_point17=official['brier_alc']-.17,weighted_difference_from_flat_point17=headroom),input_sha256={name:hashlib.sha256((audit.OUT/name).read_bytes()).hexdigest() for name in ['v81_validation_records.json','plan.json']},paid_calls=0,training=False,production_changes=False)
    save('result.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()

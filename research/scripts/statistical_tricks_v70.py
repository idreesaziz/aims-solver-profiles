"""Research-guided adaptation screen then sealed fresh-split validation.

Same inspected question bank/configurations. New seeds are fresh randomization,
NOT independent benchmark/model holdouts. No external APIs or paid compute.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import sys
import time
from pathlib import Path
import numpy as np
from scipy.integrate import quad
from scipy.stats import beta as beta_dist
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO))
from models import statistical_adaptation_v70 as alg
from models import correlated_profile_adaptation_v67 as current
from scripts import demand_residual_adaptation_v48 as cache
from scripts import drift_feature_adaptation_v64 as original
ROOT=Path('D:/AIMS/statistical_tricks_v70')
CORE=['BBH','GPQA','MATH','MMLU-Pro']
METHODS=['current']+[f'beta{k:g}' for k in alg.KAPPAS]+['beta_mixture','beta_mixture_no_solver','evidence_bma','evidence_bma_no_postcal']
METHODS += [f'normal_sd{sd:g}_{readout}' for sd in [1.5,3.,6.] for readout in ['map','posterior_mean']]
METHODS += [f'anchored_sd{sd:g}' for sd in [1.5,3.]]
METHODS += [f'factor_rank{rank}_penalty{penalty:g}' for rank in [1,2] for penalty in [1.,4.]]


def smoke():
    rng=np.random.default_rng(700)
    p0=rng.uniform(.05,.95,20)
    supports=[np.array([0,1,2,3,4,5,6]),np.array([1,2,4,6,7,8,9]),np.array([0,3,5,8,9,10,11])]
    observed=[rng.integers(0,2,len(s)).astype(float) for s in supports]
    queries=[np.array([12,13,14]),np.array([12,15,16]),np.array([12,17,18])]
    results=alg.beta_predictions(p0,supports,observed,queries)
    assert all(np.isfinite(np.concatenate(v)).all() for v in results.values())
    # Exact beta posterior predictive versus independent numerical integration.
    maximum=0.
    for prior,k,n,kappa in [(.3,2,3,.5),(.8,0,2,2.),(.1,3,5,8.)]:
        a=kappa*prior+k;b=kappa*(1-prior)+n-k
        ref=quad(lambda p:p*beta_dist.pdf(p,a,b),0,1,epsabs=1e-10)[0]
        value=float(alg.beta_posterior(prior,k,n,kappa));maximum=max(maximum,abs(ref-value))
        assert maximum<1e-8
    fit=alg.logistic_fit(p0,supports,observed)
    gradient_error=alg.factor_predictions(p0,supports,observed,queries,fit,rank=2,penalty=1.,_gradient_check=True)
    assert gradient_error<1e-4
    empty=[s[:0] for s in supports];zero=[y[:0] for y in observed]
    anchored=alg.logistic_fit(p0,empty,zero,item_sd=3.,anchor=True)
    pred=alg.logistic_predict(p0,queries,anchored)['posterior_mean']
    anchor_error=max(float(np.max(abs(p-p0[q]))) for p,q in zip(pred,queries))
    assert anchor_error<1e-9
    return dict(beta_quadrature_max_error=maximum,factor_gradient_error=gradient_error,
                anchored_zero_label_prior_error=anchor_error)


def predict_all(p0,support,labels,queries,requested=None):
    if requested is None:requested=METHODS
    output={};subjects=len(support)
    iid=np.concatenate(support);sid=np.repeat(np.arange(subjects),len(support[0]));y=np.concatenate(labels)
    if 'current' in requested:
        cov=(.1*np.eye(subjects)+.9*np.ones((subjects,subjects)))[None,:,:]
        ps=np.clip(p0[iid],1e-5,1-1e-5)
        fit=current.fit(np.log(ps/(1-ps)),y,sid,iid,cov)
        output['current']=[current.predict(p0[q],q,s,fit)[0] for s,q in enumerate(queries)]
    if any(m.startswith('beta') or m.startswith('evidence') for m in requested):
        output.update({k:v for k,v in alg.beta_predictions(p0,support,labels,queries).items() if k in requested})
    fits={}
    for sd in [1.5,3.,6.]:
        selected=[m for m in requested if m.startswith(f'normal_sd{sd:g}_')]
        if selected or (sd==1.5 and any(m.startswith('factor') for m in requested)):
            fits[sd]=alg.logistic_fit(p0,support,labels,item_sd=sd)
            if selected:
                prediction=alg.logistic_predict(p0,queries,fits[sd])
                for m in selected:output[m]=prediction[m.split('_',2)[2]]
    for sd in [1.5,3.]:
        name=f'anchored_sd{sd:g}'
        if name in requested:
            fit=alg.logistic_fit(p0,support,labels,item_sd=sd,anchor=True)
            output[name]=alg.logistic_predict(p0,queries,fit)['posterior_mean']
    for rank in [1,2]:
        for penalty in [1.,4.]:
            name=f'factor_rank{rank}_penalty{penalty:g}'
            if name in requested:output[name]=alg.factor_predictions(p0,support,labels,queries,fits[1.5],rank,penalty)
    assert set(output)==set(requested),(set(output),set(requested))
    return output


def aggregate(records):
    result={}
    for bank in ['combined70','structural52','constant']:
        cohorts={}
        for size in [3,10]:
            rows=[r for r in records if r['bank']==bank and r['cohort']==size]
            methods={}
            for method in sorted({r['method'] for r in rows}):
                curve=[float(np.mean([r['brier'] for r in rows if r['method']==method and r['budget']==n])) for n in original.BUDGETS]
                methods[method]=dict(brier31=curve[-1],alc=float(original.old.ALC@curve),curve=curve)
            cohorts[str(size)]=methods
        result[bank]=cohorts
    return result


def stage(kind,idx,selected=None):
    truth=np.load(cache.SWEEP/'development_outcomes.npy',mmap_mode='r')
    seeds=range(5) if kind=='screen' else range(100,110)
    records=[];traces={}
    for name in CORE:
        dp,_,groups,_,directory=original.old.orders_for(name,idx)
        banks={'combined70':np.clip(np.load(directory/'predictions.npy'),1e-4,1-1e-4).mean(1),
               'structural52':np.clip(np.load(Path('D:/AIMS/demand_structural_ablation_v65/structural52')/name/'predictions.npy'),1e-4,1-1e-4).mean(1),
               'constant':np.full(len(dp),.5)}
        for size in [3,10]:
            for seed in seeds:
                rng=np.random.default_rng(640200+10000*CORE.index(name)+100*size+seed)
                chosen=rng.permutation(24)[:size]
                pool=rng.permutation(len(dp))[:min(len(dp),max(80,1000//size))]
                ss=[];qq=[]
                for _ in chosen:
                    perm=rng.permutation(pool);ss.append(perm[:31]);qq.append(perm[len(perm)//2:])
                assert all(set(groups[s]).isdisjoint(groups[q]) for s,q in zip(ss,qq))
                for bank,p0 in banks.items():
                    requested=METHODS if kind=='screen' else list(dict.fromkeys(['current',selected[bank]]))
                    for n in original.BUDGETS:
                        support=[s[:n] for s in ss]
                        labels=[np.array(truth[dp[s],k],float) for s,k in zip(support,chosen)]
                        predictions=predict_all(p0,support,labels,qq,requested)
                        for method,pp in predictions.items():
                            for j,k in enumerate(chosen):
                                q=qq[j];p=pp[j]
                                assert np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=1
                                records.append(dict(benchmark=name,cohort=size,seed=seed,budget=n,model=int(idx['held'][k]),
                                    bank=bank,method=method,brier=float(np.mean((p-truth[dp[q],k])**2))))
                                if seed==seeds.start:traces[f'{name}_{size}_{bank}_{method}_{n}_{j}']=p
                print(kind,name,size,seed,flush=True)
                cache.save(ROOT/'status.json',dict(stage=kind,benchmark=name,cohort=size,seed=seed))
    cache.save(ROOT/f'{kind}_records.json',records)
    np.savez(ROOT/f'{kind}_seed_predictions.npz',**traces)
    summaries=aggregate(records);cache.save(ROOT/f'{kind}_result.json',dict(summaries=summaries,records=len(records)))
    return summaries


def run():
    ROOT.mkdir(parents=True,exist_ok=True);assert not (ROOT/'result.json').exists()
    idx=np.load(cache.BASE/'indices_and_weights.npz')
    files=[Path(__file__),Path(alg.__file__),Path(current.__file__),cache.BASE/'indices_and_weights.npz']
    for name in CORE:
        files.extend([cache.COLD/name/'predictions.npy',Path('D:/AIMS/demand_structural_ablation_v65/structural52')/name/'predictions.npy'])
    plan=dict(methods=METHODS,screen_seeds=list(range(5)),validation_seeds=list(range(100,110)),
        selection='ONE method perbank minimizing average ALC across3and10solver cohorts/4benchmarks on5screen seeds. Freeze before freshseed stage. Include current baseline. No selection using freshseed scores.',
        validation_limit='Fresh random splits/cohorts on SAME previously inspected bank/models. Not independent data or official test.',
        methods_detail=['Item-equal global calibration, exact leave-item-out prior shifts, leave-self-out beta-binomial solver calibration, mixture posterior mean.',
                        'Shared concentration Bayesian model averaging overkappa, exactly remove self-label evidence for calibration.',
                        'Convex full-convergence hierarchical MAP vs full-Hessian Laplace predictive mean via48node Gauss-Hermite, itemSD1.5/3/6.',
                        'Mean-preserving inverse logistic-normal prior offset for anchoredposterior prediction.',
                        'Logistic matrix factorization residual rank1/2 withpenalties1/4 andtwoinitializations. Nonconvex,180iterationbudget.'],
        sources=['https://bob-carpenter.github.io/stan-getting-started/stan-getting-started.html',
                 'https://proceedings.mlr.press/v97/trippe19a.html','https://proceedings.mlr.press/v51/ni16.html',
                 'https://jmlr.org/papers/v22/20-589.html','https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf'],
        attribution='Extends and compares TeamDrift andourv68baseline; established statistical methods credited.',
        paid_calls=0,protected_test_used=False,production_modified=False,report_modified=False,
        smoke=smoke(),frozen_files={str(p):cache.sha(p) for p in files})
    cache.save(ROOT/'plan.json',plan)
    cache.save(ROOT/'screen_freeze.json',dict(plan_sha256=cache.sha(ROOT/'plan.json'),time=time.time()))
    screen=stage('screen',idx)
    selected={bank:min(METHODS,key=lambda m:np.mean([screen[bank][str(s)][m]['alc'] for s in [3,10]])) for bank in screen}
    cache.save(ROOT/'selection_frozen.json',dict(selected=selected,criterion=plan['selection'],screen_sha256=cache.sha(ROOT/'screen_result.json'),time=time.time()))
    print('SELECTED',selected,flush=True)
    validation=stage('validation',idx,selected)
    result=dict(completed=True,screen=screen,selected=selected,validation=validation,plan=plan)
    cache.save(ROOT/'result.json',result);cache.save(ROOT/'status.json',dict(stage='complete'))
    for bank,cohorts in validation.items():
        for size,methods in cohorts.items():
            for method,r in methods.items():print('RESULT',bank,size,method,r['brier31'],r['alc'],flush=True)


if __name__=='__main__':
    if '--smoke' in sys.argv:print(smoke())
    else:run()

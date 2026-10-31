"""CPU-only downstream coordinate ablation; no active training mutation.

Uses ONLY current development-validation outcome rows of the24excludedsolvers.
All selections/means/rotations/priors use training questions and historical
solvers. No protected-test labels, new inference, or decoder fitting.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT=Path('D:/AIMS/original_single_window_v41')
OUT=ROOT/'analysis/coordinate_brier_ablation_v42'
CORE=['BBH','GPQA','MATH','MMLU-Pro','MuSR']
BUDGETS=[0,1,3,7,15,31]
ALC=np.array([.1,.2,.2,.2,.2,.1])


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,value):
    temp=path.with_name(path.name+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def update(mean,covariance,x,offset,z):
    probability=expit(offset+mean@x)
    v=covariance@x;w=probability*(1-probability)
    new=covariance-(w/(1+w*(v@x)))[:,None,None]*v[:,:,None]*v[:,None,:]
    mean=mean+(new@x)*(z-probability)[:,None]
    return mean,(new+new.transpose(0,2,1))/2


def make_specs(c,mu,cov,rotation,L):
    specs=[]
    for size in range(5):
        for subset in itertools.combinations(range(4),size):
            traits=[j for j in subset if j>0];omitted=[j for j in [1,2,3] if j not in traits]
            A=np.zeros((1+len(traits),4));A[0,0]=1
            for j in omitted:A[0,j]=c[j]
            for i,j in enumerate(traits):A[i+1,j]=1
            specs.append(dict(name='raw_'+(''.join(str(j+1) for j in subset) or 'none'),kind='raw',coordinates=list(subset),traits=traits,
                mean=A@mu,cov=A@cov@A.T,question_numbers=len(subset)))
    # x=(q_traits-c_traits) L R; q_offset'=q0+(q_traits-c)mu_traits.
    # New solver intercept is ability+c_traits*traits. This is an exact affine
    # reparameterization at fullrank, including intercept/trait covariance.
    B=np.zeros((4,4));B[0]=np.r_[1,c[1:]];B[1:,1:]=rotation.T@np.linalg.inv(L)
    transformed_cov=B@cov@B.T
    transformed_mu=np.r_[mu[0]+c[1:]@mu[1:],np.zeros(3)]
    for rank in range(4):
        specs.append(dict(name=f'rotated_difficulty_plus_{rank}',kind='rotated',rank=rank,
            coordinates=list(range(rank+1)),question_numbers=rank+1,mean=transformed_mu[:rank+1],cov=transformed_cov[:rank+1,:rank+1]))
    return specs


def representation(q,spec,c,mu,rotation,L):
    if spec['kind']=='raw':
        offset=q[:,0] if 0 in spec['coordinates'] else np.full(len(q),c[0])
        return offset,np.c_[np.ones(len(q)),q[:,spec['traits']]]
    centered=q[:,1:]-c[1:]
    offset=q[:,0]+centered@mu[1:]
    return offset,np.c_[np.ones(len(q)),(centered@L@rotation)[:,:spec['rank']]]


def support_orders(families,groups,seeds=10):
    orders=[]
    for seed in range(seeds):
        rng=np.random.default_rng(4242+seed);support=[]
        for i,family in enumerate(CORE):
            ids=np.flatnonzero(families==family)
            n=7 if i==seed%len(CORE) else 6
            assert len(ids)>n
            support.extend(rng.choice(ids,n,replace=False))
        support=rng.permutation(support);evaluation=np.setdiff1d(np.arange(len(families)),support)
        assert len(support)==31 and set(groups[support]).isdisjoint(groups[evaluation])
        orders.append((support,evaluation))
    return orders


def score(q,spec,Y,families,orders,c,mu,rotation,L):
    offset,X=representation(q,spec,c,mu,rotation,L);rows=[]
    for seed,(support,evaluation) in enumerate(orders):
        mean=np.tile(spec['mean'],(Y.shape[1],1));cov=np.broadcast_to(spec['cov'],(Y.shape[1],len(spec['mean']),len(spec['mean']))).copy()
        for budget in range(32):
            if budget in BUDGETS:
                probabilities=expit(offset[evaluation,None]+X[evaluation]@mean.T)
                errors=(probabilities-Y[evaluation])**2
                for family in CORE:
                    k=families[evaluation]==family;assert k.any()
                    for model,brier in enumerate(errors[k].mean(0)):
                        rows.append((seed,budget,family,model,float(brier)))
            if budget<31:
                i=support[budget];mean,cov=update(mean,cov,X[i],offset[i],Y[i])
    return rows


def summarize(rows):
    # Equal benchmark weight and equal held-out model/seed weight.
    curve={b:float(np.mean([r[4] for r in rows if r[1]==b])) for b in BUDGETS}
    per={f:float(np.mean([r[4] for r in rows if r[1]==31 and r[2]==f])) for f in CORE}
    return dict(brier31=curve[31],alc=float(np.array([curve[b] for b in BUDGETS])@ALC),curve=curve,per_family_brier31=per)


def main():
    if OUT.exists():raise RuntimeError('Frozen ablation already exists; do not overwrite')
    OUT.mkdir(parents=True);start=time.monotonic()
    frame=pq.read_table(ROOT/'data/rows.parquet').to_pandas();fit=np.load(ROOT/'data/indices.npz')['fit']
    source=frame.iloc[fit];source=source[source.family.isin(CORE)].sort_values('index').drop_duplicates('group')
    target=np.load('E:/AIMS/text_fingerprint_4d_v1/targets.npy',mmap_mode='r')
    qfit=np.array(target[source['index']],dtype=float)
    weights=np.zeros(len(source))
    for family in CORE:
        k=source.family.to_numpy()==family;weights[k]=1/(len(CORE)*k.sum())
    c=weights@qfit
    legacy=Path('E:/AIMS/masked_answer_encoder_4d_v1')
    theta=np.load(legacy/'within__full_masked_pool__seed0/frozen.npz')['theta'].astype(float);assert theta.shape==(192,4)
    held=np.array(json.loads((legacy/'plan.json').read_text())['heldout_original_models'])
    historical=np.setdiff1d(np.arange(216),held)
    assert np.array_equal(historical,np.array(json.loads((legacy/'identity_audit.json').read_text())['panels'][0]['columns']))
    assert not set(historical)&set(held)
    mu=theta.mean(0);emp=np.cov(theta,rowvar=False);cov=.75*emp+.25*np.diag(np.diag(emp))+.01*np.eye(4);cov[0,0]=max(cov[0,0],.25)
    L=np.linalg.cholesky(cov[1:,1:]);_,singular,vt=np.linalg.svd(np.sqrt(weights)[:,None]*((qfit[:,1:]-c[1:])@L),full_matrices=False);rotation=vt.T
    specs=make_specs(c,mu,cov,rotation,L)
    # Snapshot current text predictions without running any encoder inference.
    run=ROOT/'run'
    for attempt in range(10):
        before=(run/'latest_validation.json').read_bytes();pred=np.load(run/'latest_validation_predictions.npy')
        if before==(run/'latest_validation.json').read_bytes() and (run/'latest_validation_predictions.npy').stat().st_mtime_ns>=(run/'latest_validation.json').stat().st_mtime_ns:break
        time.sleep(.2)
    else:raise RuntimeError('Concurrent validation write; retry')
    current=json.loads(before);plan=json.loads((run/'plan.json').read_text());vi=np.array(plan['validation_global_indices'])
    matched=frame.iloc[vi].copy();matched['prediction_position']=np.arange(len(matched));matched=matched[matched.family.isin(CORE)].sort_values('index').drop_duplicates('group')
    assert matched.role.eq('validation').all() and not set(matched.group)&set(source.group)
    scaling=json.loads((ROOT/'data/target_scaling.json').read_text());student=pred[matched.prediction_position.to_numpy()].astype(float)*scaling['std']+scaling['mean']
    qmatched=np.array(target[matched['index']],dtype=float)
    eligible=np.load(ROOT/'data/eligible_mask.npy');allval=frame[eligible&frame.role.eq('validation').to_numpy()&frame.family.isin(CORE).to_numpy()].sort_values('index').drop_duplicates('group')
    assert allval.role.eq('validation').all() and not set(allval.group)&set(source.group)
    bank=np.load('E:/AIMS/fingerprint_sources_v2/aligned_216/profiles.npy',mmap_mode='r')
    cohorts={
        'matched_true':dict(rows=matched,q=qmatched),
        'matched_text':dict(rows=matched,q=student),
        'larger_true':dict(rows=allval,q=np.array(target[allval['index']],dtype=float))}
    # Commit source-only choices and student predictions BEFORE target outcomes.
    np.savez(OUT/'frozen_source_and_predictions.npz',center=c,prior_mean=mu,prior_covariance=cov,trait_cholesky=L,rotation=rotation,source_singular_values=singular,
        source_global_ids=source['index'].to_numpy(),matched_ids=matched['index'].to_numpy(),student=student,truth=qmatched,larger_ids=allval['index'].to_numpy())
    manifest=dict(source_fit_questions=len(source),matched_questions=len(matched),larger_true_questions=len(allval),text_checkpoint_draws=current['presentations'],
        historical_solvers=historical.tolist(),held_out_solvers=held.tolist(),subsets=[s['name'] for s in specs],
        support='31mixedbenchmark questions:6each plus1rotatingbenchmark;10frozen seeds; evaluation excludes allsupportgroups',
        target_scope='ONLY current eligible development-validation rows. Protected test labels untouched.',
        rotation='Source-only trait-prior whitening + family-balanced source SVD. Fullrankexact; retains shared average-logit difficulty.',
        mode='Inference-time coordinate information ablation; no backbone re-training or proof of training gradient interference',
        frozen_sha256=sha(OUT/'frozen_source_and_predictions.npz'),script_sha256=sha(__file__),protected_test_used=False)
    save(OUT/'plan_frozen_before_outcomes.json',manifest)
    # Exact fullrank reparameterization proof on historical models.
    rawspec=next(s for s in specs if s['name']=='raw_1234');rotspec=next(s for s in specs if s['name']=='rotated_difficulty_plus_3')
    B=np.zeros((4,4));B[0]=np.r_[1,c[1:]];B[1:,1:]=rotation.T@np.linalg.inv(L)
    transformed=theta@B.T;transformed[:,1:]-=mu[1:]@np.linalg.inv(L).T@rotation
    off,X=representation(qmatched,rotspec,c,mu,rotation,L)
    exact=float(np.max(abs(qmatched[:,0,None]+np.c_[np.ones(len(qmatched)),qmatched[:,1:]]@theta.T-(off[:,None]+X@transformed.T))))
    assert exact<1e-10
    summaries=[];saved={};all_records=[]
    with threadpool_limits(1):
        for name,cohort in cohorts.items():
            rows=cohort['rows'];assert rows.source.eq('aligned_216').all()
            Y=np.asarray(bank[np.ix_(rows.row.to_numpy(),held)],dtype=float);assert np.isfinite(Y).all() and np.isin(Y,[0,1]).all()
            families=rows.family.to_numpy();orders=support_orders(families,rows.group.to_numpy());saved[name]={}
            for spec in specs:
                values=score(cohort['q'],spec,Y,families,orders,c,mu,rotation,L);saved[name][spec['name']]=values
                summary=summarize(values);summary.update(cohort=name,method=spec['name'],question_numbers=spec['question_numbers']);summaries.append(summary)
                for seed,budget,family,model,error in values:all_records.append((name,spec['name'],seed,budget,family,int(held[model]),error))
            # Full rotation preserves Gaussian/Laplace adaptation, not just logits.
            a=np.array([r[-1] for r in saved[name]['raw_1234']]);b=np.array([r[-1] for r in saved[name]['rotated_difficulty_plus_3']]);assert np.max(abs(a-b))<1e-10
            print('COHORT_COMPLETE',name,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    # Paired model-cluster bootstrap: conditional on this small question cohort.
    comparisons=[];rng=np.random.default_rng(4242)
    for name,methods in saved.items():
        base=np.array([r[-1] for r in methods['raw_1234'] if r[1]==31]).reshape(10,len(CORE),24)
        for method,values in methods.items():
            val=np.array([r[-1] for r in values if r[1]==31]).reshape(10,len(CORE),24)
            model_delta=(val-base).mean((0,1));boot=np.mean(model_delta[rng.integers(0,24,(2000,24))],axis=1)
            comparisons.append(dict(cohort=name,method=method,delta_brier31=float(model_delta.mean()),conditional_model_bootstrap_interval=np.quantile(boot,[.025,.975]).tolist()))
    # Recoverability is measured WITHIN each core collection, after the exact
    # source rotation, so pooled between-source R2 cannot masquerade as success.
    recoverability=[]
    for family in CORE:
        k=matched.family.to_numpy()==family
        for label,spec in [('raw',rawspec),('rotated',rotspec)]:
            a,x=representation(student[k],spec,c,mu,rotation,L);b,y=representation(qmatched[k],spec,c,mu,rotation,L)
            predq=np.c_[a,x[:,1:]];trueq=np.c_[b,y[:,1:]];mse=((predq-trueq)**2).mean(0);var=trueq.var(0)
            recoverability.append(dict(family=family,basis=label,questions=int(k.sum()),coordinate_mse=mse.tolist(),within_family_r2=(1-mse/var).tolist()))
    import csv
    with (OUT/'per_model.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['cohort','method','seed','budget','family','held_model','brier']);w.writerows(all_records)
    result=dict(completed=True,summaries=summaries,comparisons=comparisons,recoverability=recoverability,exact_full_rotation_logit_max_error=exact,
        manifest=manifest,elapsed_seconds=time.monotonic()-start,training_changed=False,protected_test_used=False,
        caveats=['Previously inspected developmentcohorts, not competition result or untouchedconfirmation.',
            'Matched text cohort has160or fewer questions and31mixed-family support; uncertainty is conditional and postselectioncomparisons exploratory.',
            'True-coordinate cohorts use suppliedhistorical-answer fingerprints, not deployable question-text predictions.',
            'This ablates inference information; removing a coordinate trainingloss may affect the backbone differently and is NOTtested.',
            'No fitting or prediction on protected-test rows; all means/rotations selected from trainingquestions and historicalsolvers.'])
    save(OUT/'result.json',result)
    save(OUT/'verification.json',dict(passed=True,all_evaluation_rows_current_validation=True,source_groups_disjoint=True,held_solvers_excluded_from_prior=True,
        source_choices_frozen_before_outcome_access=True,fullrotation_logit_error=exact,fullrotation_brier_reproduced_all_cohorts=True,
        budgets=BUDGETS,seeds=10,all_model_records=len(all_records),training_changed=False,protected_test_used=False))
    for name in cohorts:
        rows=[s for s in summaries if s['cohort']==name]
        print(name,'BEST',min(rows,key=lambda r:r['brier31']),flush=True)

if __name__=='__main__':main()

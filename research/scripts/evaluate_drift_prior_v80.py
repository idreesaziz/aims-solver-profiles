"""Matched comparison: exact Drift prior/update, same prior/ours, our system.

Uses only the cold_core artifact. NEVER full_public weights on local core tests.
Previously inspected public bank,24related excluded configurations; not official.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,time,json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import reproduce_drift_prior_v80 as training
from scripts import statistical_tricks_v70 as source
from scripts import demand_residual_adaptation_v48 as cache
from models.subject_adaptation_v79 import Adaptation
from models.final_runtime_v74 import Adaptation as Original
from scripts import drift_feature_adaptation_v64 as upstream
OUT=ROOT/'reports/drift_exact_prior_v80'

def main():
    OUT.mkdir(exist_ok=True,parents=True);assert not (OUT/'comparison.json').exists()
    art=training.OUT/'cold_core';receipt=cache.read(art/'fit_receipt.json')
    assert not set(receipt['benchmarks'])&{'mmlupro','fewshot_ttt_bbh','llm_uncertainty_bench','naturalreasoning','robust_reasoning_benchmark'}
    assert receipt['warnflag']==0,receipt['task']
    meta,arr=upstream.drift._load_art(str(art));predictor=upstream.drift.Predictor(meta,arr)
    idx=np.load(cache.BASE/'indices_and_weights.npz');truth=np.load(cache.SWEEP/'development_outcomes.npy',mmap_mode='r')
    raw=cache.read('E:/AIMS/fingerprint_sources_v2/aligned_216/subjects.json')
    subjects=[dict(normalized_name=s['model'],harness='lm-evaluation-harness') for s in raw]
    text=pq.read_table('E:/AIMS/fingerprint_sources_v2/aligned_216/questions.parquet',columns=['text'])['text']
    ability=np.array([predictor.zs(subjects[int(i)]) for i in idx['held']])
    cache.save(OUT/'comparison_plan.json',dict(seeds=[800,801,802],cohorts=[1,3,10],budgets=upstream.BUDGETS,
        methods=['drift_prior_drift_update','drift_prior_our52_update','drift_prior_our70_update','our52','our70'],
        prior_artifact_sha=cache.sha(art/'fit_receipt.json'),script_sha=cache.sha(__file__),
        metadata='Aligned bank model name; harness lm-evaluation-harness; other runtime fields unavailable and empty. Not guaranteed identical to hosted metadata.',
        limitations='Known direct core relatives excluded from broad source fit. Other cross-source item duplicates not certified absent. Inspected public development, not official. Training prior uses public subject identities; no claim of unseen-name training.',
        paid_calls=0,submission_changed=False))
    rows=[];began=time.monotonic()
    for benchmark in source.CORE:
        dp,_,groups,_,directory=source.original.old.orders_for(benchmark,idx)
        texts=[text[int(i)].as_py() for i in idx['development'][dp]]
        z=np.array([meta['b']+predictor.zi({'item_content':t}) for t in texts])
        rf={'our70':np.clip(np.load(directory/'predictions.npy'),1e-4,1-1e-4).mean(1),
            'our52':np.clip(np.load(Path('D:/AIMS/demand_structural_ablation_v65/structural52')/benchmark/'predictions.npy'),1e-4,1-1e-4).mean(1)}
        for size in [1,3,10]:
            for seed in [800,801,802]:
                rng=np.random.default_rng(800000+10000*source.CORE.index(benchmark)+100*size+seed)
                chosen=rng.permutation(24)[:size];pool=rng.permutation(len(dp))[:min(len(dp),max(80,1000//size))]
                ss=[];qq=[]
                for _ in chosen:
                    order=rng.permutation(pool);ss.append(order[:31]);qq.append(order[len(order)//2:])
                assert all(set(groups[s]).isdisjoint(groups[q]) for s,q in zip(ss,qq))
                p0=np.clip(expit(z[None,:]+ability[chosen,None]),1e-5,1-1e-5)
                for n in upstream.BUDGETS:
                    support=[s[:n] for s in ss];ys=[np.asarray(truth[dp[s],k],float) for s,k in zip(support,chosen)]
                    exact=upstream.fit(upstream.rows_for(z,ability[chosen],support,ys))
                    ours52=Adaptation(p0,support,ys,.25,'pseudo4');ours70=Adaptation(p0,support,ys,.5,'gaussian1')
                    controls={m:Original(b,support,ys,.5 if m=='our70' else .25,'gaussian1' if m=='our70' else 'pseudo4') for m,b in rf.items()}
                    for j,k in enumerate(chosen):
                        q=qq[j]
                        predictions={'drift_prior_drift_update':upstream.forecast(z,ability[chosen],exact,np.full(len(q),j),q),
                            'drift_prior_our52_update':ours52.predict(p0[j,q],q,j),'drift_prior_our70_update':ours70.predict(p0[j,q],q,j)}
                        predictions.update({m:control.predict(rf[m][q],q,j) for m,control in controls.items()})
                        for m,p in predictions.items():
                            assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                            rows.append(dict(benchmark=benchmark,cohort=size,seed=seed,budget=n,method=m,model=int(idx['held'][k]),brier=float(np.mean((p-truth[dp[q],k])**2))))
                print('EVAL',benchmark,size,seed,flush=True)
                cache.save(OUT/'comparison_progress.json',dict(benchmark=benchmark,cohort=size,seed=seed,seconds=time.monotonic()-began))
    cache.save(OUT/'comparison_records.json',rows)
    result={}
    for size in [1,3,10]:
        result[str(size)]={}
        for method in sorted({r['method'] for r in rows}):
            curve=[float(np.mean([r['brier'] for r in rows if r['cohort']==size and r['method']==method and r['budget']==n])) for n in upstream.BUDGETS]
            result[str(size)][method]=dict(alc=float(upstream.old.ALC@curve),brier31=curve[-1],curve=curve)
    cache.save(OUT/'comparison.json',dict(results=result,seconds=time.monotonic()-began));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

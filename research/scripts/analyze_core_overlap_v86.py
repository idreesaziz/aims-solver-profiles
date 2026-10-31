"""Recreate the frozen v80 public episode draws; count other-solver overlap."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import evaluate_drift_prior_v80 as e
def main():
    idx=np.load(e.cache.BASE/'indices_and_weights.npz');rows=[]
    for benchmark in e.source.CORE:
        dp,_,groups,_,directory=e.source.original.old.orders_for(benchmark,idx)
        for size in [1,3,10]:
            for seed in [800,801,802]:
                rng=np.random.default_rng(800000+10000*e.source.CORE.index(benchmark)+100*size+seed)
                chosen=rng.permutation(24)[:size];pool=rng.permutation(len(dp))[:min(len(dp),max(80,1000//size))]
                ss=[];qq=[]
                for _ in chosen:
                    order=rng.permutation(pool);ss.append(order[:31]);qq.append(order[len(order)//2:])
                for n in e.upstream.BUDGETS:
                    for j,q in enumerate(qq):
                        other={int(i) for k,s in enumerate(ss) if k!=j for i in s[:n]}
                        rows.append(dict(benchmark=benchmark,cohort=size,seed=seed,budget=n,coverage=float(np.mean([int(i) in other for i in q])),pool=len(pool)))
    import pandas as pd
    out=ROOT/'reports/recent_run_diagnosis_v86';d=pd.DataFrame(rows);d.to_csv(out/'core_overlap.csv',index=False)
    result={str(c):{str(n):float(g.coverage.mean()) for n,g in d[d.cohort==c].groupby('budget')} for c in [1,3,10]}
    (out/'core_overlap.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()

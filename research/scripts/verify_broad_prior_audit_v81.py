"""Independent aggregation, split checks and paired source-group bootstrap."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.demand_residual_adaptation_v48 import read,save,sha
OUT=ROOT/'reports/broad_prior_audit_v81'

def main():
    p=read(OUT/'plan.json');r=read(OUT/'result.json');sel=read(OUT/'selection_frozen.json')
    assert sel['selected']==r['selected'] and sel['screen_sha']==sha(OUT/'v81_screen_records.json')
    for stage in ['screen','validation']:
        tr=p['train_'+stage];te=p[stage]
        assert not {p['groups'][b] for b in tr}&{p['groups'][b] for b in te}
        receipt=read(Path('E:/AIMS/drift_exact_prior_v80')/('v81_'+stage)/'fit_receipt.json')
        assert receipt['benchmarks']==tr and receipt['warnflag']==0
    maximum=0.;summaries={};frames={}
    for stage in ['screen','validation']:
        d=pd.DataFrame(read(OUT/f'v81_{stage}_records.json'));frames[stage]=d
        assert not d.duplicated(['benchmark','cohort','seed','budget','method','pair']).any()
        neutral=d[(d.budget==0)&(d.method=='neutral')].brier.to_numpy()
        assert len(neutral)>0 and np.max(abs(neutral-.25))<1e-12
        summary={}
        for cohort in [1,3]:
            summary[str(cohort)]={}
            for m in sorted(d.method.unique()):
                curve=d[(d.cohort==cohort)&(d.method==m)].groupby('budget').brier.mean().reindex([0,1,3,7,15,31]).to_numpy()
                value=dict(early=float(curve[:3].sum()/3),alc=float((curve*np.array([.1,.2,.2,.2,.2,.1])).sum()))
                if stage=='screen' and m!='ours':value.pop('alc')
                summary[str(cohort)][m]=value
                for key in value:maximum=max(maximum,abs(value[key]-r[stage][str(cohort)][m][key]))
        summaries[stage]=summary
    assert maximum<1e-12,maximum
    winner=min(p['methods'],key=lambda m:np.mean([summaries['screen'][str(c)][m]['early'] for c in [1,3]]))
    assert winner==sel['selected']
    # Conditional paired uncertainty: benchmark sources within a common repo are
    # resampled together; seeds/pairs remain paired across the candidate/control.
    d=frames['validation'];bootstrap={};rng=np.random.default_rng(811)
    for cohort in [1,3]:
        a=d[(d.cohort==cohort)&d.budget.isin([0,1,3])].groupby(['benchmark','seed','pair','method']).brier.mean().unstack('method')
        groups=np.array([p['groups'][b] for b in a.index.get_level_values('benchmark')]);unique=np.unique(groups)
        for method in a.columns:
            delta=(a[method]-a['ours']).to_numpy();sums=np.array([delta[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
            draws=rng.integers(0,len(unique),size=(2000,len(unique)));means=sums[draws].sum(1)/counts[draws].sum(1)
            bootstrap[f'{cohort}:{method}']=dict(delta=float(delta.mean()),interval95=np.quantile(means,[.025,.975]).tolist(),source_groups=len(unique))
    bootstrap_alc={}
    for cohort in [1,3]:
        work=d[d.cohort==cohort].copy()
        work['weighted']=work.brier*work.budget.map({0:.1,1:.2,3:.2,7:.2,15:.2,31:.1})
        a=work.groupby(['benchmark','seed','pair','method']).weighted.sum().unstack('method')
        groups=np.array([p['groups'][b] for b in a.index.get_level_values('benchmark')]);unique=np.unique(groups)
        for method in a.columns:
            delta=(a[method]-a['ours']).to_numpy();sums=np.array([delta[groups==g].sum() for g in unique]);counts=np.array([(groups==g).sum() for g in unique])
            draws=rng.integers(0,len(unique),size=(2000,len(unique)));means=sums[draws].sum(1)/counts[draws].sum(1)
            bootstrap_alc[f'{cohort}:{method}']=dict(delta=float(delta.mean()),interval95=np.quantile(means,[.025,.975]).tolist(),source_groups=len(unique))
    for path in (OUT/'episodes').glob('*.json'):
        e=read(path)
        # Legacy cached proposals may include a duplicate; scoring must apply
        # the explicit visible-identity deduplication before constructing fits.
        from scripts.broad_prior_audit_v81 import distinct_subjects
        e=distinct_subjects(e)
        keys=[tuple(sorted(pair['subject'].items())) for pair in e['pairs']]
        assert len(keys)==len(set(keys))
        for pair in e['pairs']:
            support={' '.join(e['items'][i]['item_content'].split()) for i in pair['support']}
            query={' '.join(e['items'][i]['item_content'].split()) for i in pair['query']}
            assert support.isdisjoint(query) and len(pair['support'])==31
            assert len(pair['query'])==len(pair['n'])==len(pair['k'])
            assert all(n>0 and 0<=k<=n for k,n in zip(pair['k'],pair['n']))
    save(OUT/'independent_verification.json',dict(passed=True,aggregate_max_error=maximum,selection_verified=True,source_groups_disjoint=True,own_support_query_text_disjoint=True,validation_bootstrap=bootstrap,validation_alc_bootstrap=bootstrap_alc))
    print(json.dumps(bootstrap,indent=2))
if __name__=='__main__':main()

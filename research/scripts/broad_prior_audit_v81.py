"""Broad source-group-held-out comparison of starting priors; fixed adaptation.

No paid annotations: portable52-feature forest, exact-recipe Drift prior, fixed
probability blends, softened forest and 0.5. Early Brier drives selection; six
budgets and ALC are also reported. Separate source groups for screening/testing.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import sys,json,hashlib,re,time,gc
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.special import expit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts import reproduce_drift_prior_v80 as T
from scripts.demand_residual_adaptation_v48 import save,read,sha
from scripts.export_final_forests_v74 import portable_predict
from scripts.drift_feature_adaptation_v64 import drift
from models.final_runtime_v74 import Forest,local_features
from models.subject_adaptation_v79 import Adaptation
OUT=ROOT/'reports/broad_prior_audit_v81'
METHODS=['ours','drift','blend25','blend50','blend75','soft_ours','neutral']
BUDGETS=[0,1,3,7,15,31];WEIGHTS=np.array([.1,.2,.2,.2,.2,.1])
EXCLUDE={'mmlupro','mmlu','fewshot_ttt_bbh','llm_uncertainty_bench','naturalreasoning','robust_reasoning_benchmark','babilong','scbench_long'}

def group(b):
    meta=pq.read_table(T.DATA/b/'formatted_tables/benchmarks.parquet').to_pylist()[0]
    url=str(meta.get('source_url') or '').lower();match=re.search(r'github\.com/([^/]+/[^/#?]+)',url)
    if b.startswith('medhelm_'):return 'family:medhelm'
    if b.startswith('igakuqa'):return 'family:igakuqa'
    if b.startswith('helm_'):return 'family:helm'
    if 'swebench' in b or b.startswith('swe_') or b=='multi_swebench':return 'family:swe'
    return 'repo:'+match.group(1).removesuffix('.git') if match else 'benchmark:'+b

def plan():
    if (OUT/'plan.json').exists():return read(OUT/'plan.json')
    groups={b:group(b) for b in T.SOURCE['benchmark_names']}
    excluded_groups={groups[b] for b in EXCLUDE if b in groups}
    # Source metadata only; no outcome rates used to choose the split.
    eligible=[]
    for b,g in groups.items():
        if g in excluded_groups:continue
        f=T.load(b);counts=np.bincount(f['si'],minlength=f['S'].shape[0])
        if np.any(counts>=80):eligible.append(b)
    unique=sorted({groups[b] for b in eligible});rng=np.random.default_rng(810);rng.shuffle(unique)
    assignments={g:i%3 for i,g in enumerate(unique)}
    screen=[b for b in eligible if assignments[groups[b]]==0]
    validation=[b for b in eligible if assignments[groups[b]]==1]
    train_screen=[b for b in groups if groups[b] not in {groups[x] for x in screen}]
    train_validation=[b for b in groups if groups[b] not in {groups[x] for x in validation}]
    assert not {groups[b] for b in screen}&{groups[b] for b in validation}
    for tr,te in [(train_screen,screen),(train_validation,validation)]:assert not {groups[b] for b in tr}&{groups[b] for b in te}
    p=dict(methods=METHODS,screen=screen,validation=validation,train_screen=train_screen,train_validation=train_validation,groups=groups,
        seed=810,episode_seeds=[810,811],cohorts=[1,3],budgets=BUDGETS,
        selection='Minimize pair-equal mean Brier across0/1/3, averaged equally across cohort settings. Freeze before validation. No selection by official feedback.',
        fixed_update='Subject-specific v74 extension,kappa.25,pseudo4,solver ridge16,Brier-stack penalty1; identical for all priors.',
        scope='API-free52 prior only; no18demand annotations on broad sources. Exclude direct core relatives and two expensive long-context banks. Skip inputs over100000chars without truncation.',
        caveats='Whole source-group exclusion, not globally certified question dedup across unrelated repositories. Previously inspected public release, not protected/official test. Drift may see validation subject identities on other sources.',
        scripts_sha=sha(__file__),forest_sha=sha(ROOT/'artifacts/final_v74/structural52.npz'),paid_calls=0,submission_changed=False)
    save(OUT/'plan.json',p);return p

def candidates_for(b,size,seed,f):
    counts=np.bincount(f['si'],minlength=f['S'].shape[0]);eligible=np.flatnonzero(counts>=80)
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'{b}|{size}|{seed}'.encode()).digest()[:8],'big'))
    chosen=rng.permutation(eligible)[:size];candidates={}
    for sid in chosen:
        at=np.flatnonzero(f['si']==sid);at=rng.permutation(at)[:max(80,1000//len(chosen))];candidates[int(sid)]=at
    return rng,candidates


BANK={}
def bank_inputs(b,f):
    if BANK.get('benchmark')==b:return BANK['items'],BANK['source_subjects'],BANK['tools']
    needed=set();wanted_sids=set()
    for size in [1,3]:
        for seed in [810,811]:
            _,candidates=candidates_for(b,size,seed,f)
            wanted_sids.update(candidates)
            needed.update(int(i) for at in candidates.values() for i in f['ii'][at])
    all_ids=set();observed_subjects=set()
    # Recover upstream's sorted observed-item selection, not raw table row order.
    for batch in pq.ParquetFile(T.DATA/b/'formatted_tables/responses.parquet').iter_batches(batch_size=65536,columns=['subject_id','item_id','response']):
        frame=batch.to_pandas().dropna(subset=['response']);all_ids.update(frame.item_id.unique());observed_subjects.update(frame.subject_id.unique())
    all_ids=sorted(all_ids);r=np.random.default_rng(int.from_bytes(b.encode()[:4].ljust(4,b'0'),'big'))
    selected=np.sort(r.permutation(len(all_ids))[:10000]);lookup={all_ids[selected[i]]:i for i in needed}
    source_subjects=sorted(observed_subjects);wanted_subjects={source_subjects[s] for s in wanted_sids};tools={}
    for batch in pq.ParquetFile(T.DATA/b/'formatted_tables/responses.parquet').iter_batches(batch_size=65536,columns=['subject_id','item_id','response','interactors']):
        frame=batch.to_pandas().dropna(subset=['response']);frame=frame[frame.subject_id.isin(wanted_subjects)&frame.item_id.isin(lookup)]
        for row in frame.itertuples(index=False):
            # Exactly first non-null tool metadata, as in original aggregation.
            key=(row.subject_id,lookup[row.item_id])
            if key not in tools and row.interactors is not None and not (isinstance(row.interactors,float) and np.isnan(row.interactors)):
                tools[key]=str(row.interactors)
    items={}
    for batch in pq.ParquetFile(T.DATA/b/'formatted_tables/items.parquet').iter_batches(batch_size=64,columns=['item_id','content','item_features']):
        for row in batch.to_pylist():
            j=lookup.get(row['item_id'])
            if j is None or j in items:continue
            text=T.clean(row['content'],10**12)
            if len(text)>100000:continue
            items[j]=dict(item_content=text,item_features=T.clean(row['item_features'],10**12))
    BANK.clear();BANK.update(benchmark=b,items=items,source_subjects=source_subjects,tools=tools)
    return items,source_subjects,tools


def episode(b,size,seed):
    path=OUT/'episodes'/f'{b}_{size}_{seed}.json'
    if path.exists():return distinct_subjects(read(path))
    f=T.load(b);rng,candidates=candidates_for(b,size,seed,f)
    items,source_subjects,tools=bank_inputs(b,f)
    subjects=read(T.OUT/'features'/b/'subjects.json');pairs=[]
    # Per-pair text groups keep duplicated wording out of both support and query.
    for sid,at in candidates.items():
        available=[int(i) for i in at if int(f['ii'][i]) in items]
        grouped={}
        for i in available:
            iid=int(f['ii'][i]);key=' '.join(items[iid]['item_content'].split())
            grouped.setdefault(key,[]).append(i)
        keys=list(grouped);rng.shuffle(keys)
        if len(keys)<80:continue
        half=len(keys)//2;support_keys=keys[:half];query_keys=keys[half:]
        support=[grouped[k][0] for k in support_keys[:31]];queries=[i for k in query_keys for i in grouped[k]]
        assert set(support_keys).isdisjoint(query_keys)
        pairs.append(dict(subject=subjects[sid],source_sid=sid,support=[int(f['ii'][i]) for i in support],
            labels=[int(rng.random()<float(f['k'][i])/float(f['n'][i])) for i in support],
            query=[int(f['ii'][i]) for i in queries],n=[int(f['n'][i]) for i in queries],k=[int(f['k'][i]) for i in queries]))
    variants={}
    for pair in pairs:
        sid=source_subjects[pair.pop('source_sid')]
        for key in ['support','query']:
            converted=[]
            for i in pair[key]:
                tool=tools.get((sid,i),'');variant=f'{i}:{hashlib.sha256(tool.encode()).hexdigest()}'
                variants[variant]=dict(items[i],interactors=tool);converted.append(variant)
            pair[key]=converted
    out=dict(benchmark=b,cohort=size,seed=seed,pairs=pairs,items=variants)
    out=distinct_subjects(out);save(path,out);return out

def distinct_subjects(episode):
    """Source IDs cannot buy extra observations for the same visible subject."""
    seen=set();pairs=[]
    for pair in episode['pairs']:
        key=tuple(sorted(pair['subject'].items()))
        if key in seen:continue
        seen.add(key);pairs.append(pair)
    episode['pairs']=pairs
    return episode

def summarize(rows,screen=False):
    out={}
    for size in [1,3]:
        out[str(size)]={}
        for m in sorted({r['method'] for r in rows}):
            curve=[float(np.mean([r['brier'] for r in rows if r['cohort']==size and r['method']==m and r['budget']==n])) if not(screen and m!='ours' and n>=7) else None for n in BUDGETS]
            out[str(size)][m]=dict(early=float(np.mean(curve[:3])),alc=None if None in curve else float(WEIGHTS@curve),curve=curve)
    return out

def evaluate(benchmarks,tag,methods):
    artifact=T.OUT/tag;meta,arr=drift._load_art(str(artifact));dp=drift.Predictor(meta,arr)
    forest=Forest(ROOT/'artifacts/final_v74/structural52.npz');rows=[]
    for b in benchmarks:
        checkpoint=OUT/'chunks'/tag/(b+'.json')
        if checkpoint.exists():
            stored=read(checkpoint)
            compatible=[sha(__file__)]+read(OUT/'plan.json').get('compatible_full_budget_script_shas',[])
            assert stored['methods']==methods and stored['evaluation_sha'] in compatible
            rows.extend(stored['rows']);print('CACHED',tag,b,len(rows),flush=True);continue
        start=len(rows)
        for size in [1,3]:
            for seed in [810,811]:
                ep=episode(b,size,seed)
                if not ep['pairs']:continue
                ids=sorted(ep['items']);pos={i:j for j,i in enumerate(ids)};items=[ep['items'][i] for i in ids]
                x=np.array([local_features(i['item_content']) for i in items],np.float32)
                rf=np.clip(portable_predict(forest.arrays,x),1e-4,1-1e-4).mean(1)
                zs=np.array([dp.zs(p['subject']) for p in ep['pairs']]);zi=np.array([meta['b']+dp.zi(i) for i in items])
                historical=expit(zs[:,None]+zi[None,:]);ours=np.broadcast_to(rf,historical.shape)
                priors={'ours':ours,'drift':historical,'neutral':np.full_like(ours,.5),'soft_ours':.5*ours+.25}
                for weight in [.25,.5,.75]:priors[f'blend{int(weight*100)}']=(1-weight)*ours+weight*historical
                for n in BUDGETS:
                    support=[np.array([pos[i] for i in p['support'][:n]],int) for p in ep['pairs']]
                    yy=[np.array(p['labels'][:n],float) for p in ep['pairs']]
                    for m in methods:
                        # Selection uses only0/1/3. Avoid costly unused late fits;
                        # validation still evaluates every method at all budgets.
                        if tag=='v81_screen' and n>=7 and m!='ours':continue
                        p0=priors[m];fitted=Adaptation(p0,support,yy,.25,'pseudo4')
                        for j,pair in enumerate(ep['pairs']):
                            q=np.array([pos[i] for i in pair['query']],int);pred=fitted.predict(p0[j,q],q,j)
                            nn=np.array(pair['n'],float);kk=np.array(pair['k'],float)
                            assert np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all()
                            error=float((kk*(1-pred)**2+(nn-kk)*pred**2).sum()/nn.sum())
                            rows.append(dict(benchmark=b,cohort=size,actual_cohort=len(ep['pairs']),seed=seed,budget=n,method=m,pair=j,brier=error))
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        save(checkpoint,dict(methods=methods,evaluation_sha=sha(__file__),rows=rows[start:]))
        save(OUT/f'{tag}_progress.json',dict(benchmark=b,records=len(rows)));print('EVAL',tag,b,len(rows),flush=True)
    save(OUT/f'{tag}_records.json',rows);return summarize(rows,screen=(tag=='v81_screen'))

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'episodes').mkdir(exist_ok=True)
    assert not (OUT/'result.json').exists();p=plan();began=time.monotonic()
    # Training source lists fixed before any evaluation, same released recipe.
    T.fit_stream(p['train_screen'],'v81_screen')
    screen=evaluate(p['screen'],'v81_screen',METHODS);save(OUT/'screen_summary.json',screen)
    selected=min(METHODS,key=lambda m:np.mean([screen[str(c)][m]['early'] for c in [1,3]]))
    save(OUT/'selection_frozen.json',dict(selected=selected,screen_sha=sha(OUT/'v81_screen_records.json')))
    T.fit_stream(p['train_validation'],'v81_validation')
    validation=evaluate(p['validation'],'v81_validation',list(dict.fromkeys(['ours',selected,'drift','neutral'])))
    save(OUT/'result.json',dict(selected=selected,screen=screen,validation=validation,seconds=time.monotonic()-began))
    print('DONE',json.dumps(validation),flush=True)
if __name__=='__main__':main()

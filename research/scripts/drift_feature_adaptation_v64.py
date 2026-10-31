"""Team Drift's unmodified v2 MAP update on our cached forest priors.

No API, no model training, no production modification. MIT upstream retained in
external/aims-team-drift. This substitutes the offline prior, not the update.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import importlib.util
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts import logistic_adaptation_v60 as old
from scripts.support_cv_residual_v48b import choose_from_support
from scripts import demand_residual_adaptation_v48 as cache

ROOT = Path('D:/AIMS/drift_feature_adaptation_v64')
UPSTREAM = REPO/'external/aims-team-drift/submission/model.py'
spec = importlib.util.spec_from_file_location('drift_v2_reference', UPSTREAM)
drift = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drift)
BUDGETS = old.BUDGETS
METHODS = ['rf_uniform', 'rf_bayes', 'probability_cv', 'drift_solo', 'drift_pooled',
           'named_prior', 'drift_named_solo', 'drift_named_pooled']


def fit(rows):
    instance = drift.Predictor.__new__(drift.Predictor)
    instance.sd = dict(drift.PRIOR_SD)
    return instance._fit(rows)


def forecast(z, ability, fitted, subjects, items):
    beta, g, delta, d, _ = fitted
    eta = z[items]+ability[subjects]+beta+g*ability[subjects]
    eta += np.array([delta.get(int(s), 0.) for s in subjects])
    eta += np.array([d.get(int(i), 0.) for i in items])
    return np.clip(expit(np.clip(eta, -30, 30)), *drift.CLIP)


def rows_for(z, ability, supports, observed):
    # Subject-major order matches upstream harness.labeled_at. Only supplied
    # observed labels are accepted; query outcomes cannot enter this function.
    return [(s, int(i), float(ability[s]), float(z[i]), int(y))
            for s, (items, labels) in enumerate(zip(supports, observed))
            for i, y in zip(items, labels)]


def smoke():
    from scipy.optimize import minimize
    rng = np.random.default_rng(640)
    z = rng.normal(size=12)
    ability = np.array([-.7, .4, .9])
    supports = [np.array([0, 1, 2]), np.array([1, 3, 4]), np.array([2, 4, 5])]
    ys = [np.array([0, 1, 0]), np.array([1, 0, 1]), np.array([1, 1, 0])]
    rows = rows_for(z, ability, supports, ys)
    got = fit(rows)
    pairs = list(dict.fromkeys(r[0] for r in rows))
    items = list(dict.fromkeys(r[1] for r in rows))
    x = np.zeros((len(rows), 2+len(pairs)+len(items)))
    for k, r in enumerate(rows):
        x[k, 0:2] = [1., r[2]]
        x[k, 2+pairs.index(r[0])] = 1.
        x[k, 2+len(pairs)+items.index(r[1])] = 1.
    prec = 1/np.square([drift.PRIOR_SD['beta'], drift.PRIOR_SD['g']]+
                      [drift.PRIOR_SD['delta']]*len(pairs)+[drift.PRIOR_SD['d']]*len(items))
    offset = np.array([r[2]+r[3] for r in rows])
    y = np.array([r[4] for r in rows])
    # Independent coordinate-Newton matrix implementation, same 12 sweeps and
    # same step cap, independently reconstructs upstream fit exactly.
    b = np.zeros(x.shape[1])
    eta = offset.copy()
    for _ in range(drift.SWEEPS):
        for j in range(len(b)):
            p = expit(eta)
            step = np.clip((x[:, j]@(y-p)-prec[j]*b[j]) /
                           (prec[j]+(x[:, j]**2)@(p*(1-p))), -1.5, 1.5)
            b[j] += step
            eta += step*x[:, j]
    upstream = np.array([got[0], got[1]]+[got[2][j] for j in pairs]+[got[3][j] for j in items])
    assert np.max(abs(b-upstream)) < 1e-12
    def objective(v):
        a = offset+x@v
        return np.sum(np.logaddexp(0., a)-y*a)+.5*np.sum(prec*v*v)
    opt = minimize(objective, b, jac=lambda v:x.T@(expit(offset+x@v)-y)+prec*v,
                   method='BFGS', options={'gtol':1e-10})
    gap = float(objective(b)-objective(opt.x))
    assert gap < 1e-5
    empty = fit([])
    assert empty[:2] == (0., 0.) and not empty[2] and not empty[3]
    p = forecast(z, ability, got, np.array([0, 1]), np.array([9, 10]))
    assert np.isfinite(p).all() and ((p >= .01)&(p <= .99)).all()
    return dict(independent_12_sweep_max_error=float(np.max(abs(b-upstream))),
                synthetic_full_optimizer_objective_gap=gap, zero_labels_pass=True)


def run():
    ROOT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT/'result.json').exists(), 'Completed already; do not duplicate.'
    idx = np.load(cache.BASE/'indices_and_weights.npz')
    cards = np.load(cache.COLD/'training_scorecards.npy')
    assert cards.shape == (216, 5)
    files = [Path(__file__), UPSTREAM, UPSTREAM.with_name('featurize.py'),
             cache.BASE/'indices_and_weights.npz', cache.COLD/'training_scorecards.npy']
    for protocol in old.PROTOCOLS:
        _, _, _, _, directory = old.orders_for(protocol, idx)
        files.append(directory/'predictions.npy')
    plan = dict(upstream_commit='5a67dfcbaf8742f639edbfb96f15ee6efe3eed04',
        upstream_url='https://github.com/cloudlabworks/aims-team-drift',
        update='Calls upstream Predictor._fit unchanged: 12 sweeps, released SDs, step cap, .01/.99 output clipping; variance option disabled as upstream default.',
        prior='Our fixed 18+52 feature RF historical mean. No upstream offline weights distributed.',
        neutral_ability='Set zs=0 because RF mean has no subject identity; g is inactive in this variant.',
        named_ability='Separate diagnostic: source-only existing named solver scorecards, logit mean across non-target benchmarks, centered on192historical solvers. No held benchmark source responses. Uses known target configuration history, NOT unseen-identity evaluation.',
        layouts=['legacy_shared_splits', 'independent_pair_splits'],
        evidence='Per benchmark, all24subjects each supply current budget only; no own query labels. Cross-subject same-item labels allowed ONLY from their own acquisition pools.',
        independent='50/50 question-group split independently per subject, 31 nested random support items; matched new controls, not comparable directly to legacy metrics.',
        budgets=BUDGETS, alc_weights=old.ALC.tolist(), methods=METHODS,
        selection='No target selection/tuning. Upstream fixed SD beta1.5/g.35/delta.8/d1.5.',
        limitations=['Exploratory inspected related24configurations, not official or independent confirmation.',
                    '24solvers pooled vs competitor differing cohort sizes; not equal evidence to solo controls.',
                    'Legacy identical splits prohibit evaluation item effects; separate splits test this mechanism.',
                    'Still requires18demand scores for unseen questions; this is not an API-free submission.',
                    'RF prior substitution means this is not reproduction of their full predictor.'],
        paid_calls=0, protected_test_used=False, production_modified=False,
        frozen_files={str(p):cache.sha(p) for p in files}, smoke=smoke())
    cache.save(ROOT/'plan.json', plan)
    # Seal all choices and prior inputs before reading outer outcomes.
    cache.save(ROOT/'prediction_freeze.json', dict(plan_sha256=cache.sha(ROOT/'plan.json'), timestamp=time.time()))
    target_all = np.load(cache.SWEEP/'development_outcomes.npy', mmap_mode='r')
    records = []
    overlap = []
    for protocol in old.PROTOCOLS:
        dp, family, groups, orders, directory = old.orders_for(protocol, idx)
        expert = np.clip(np.load(directory/'predictions.npy'), 1e-4, 1-1e-4)
        target = target_all[dp]
        p0 = expert.mean(1)
        z = np.log(p0)-np.log1p(-p0)
        columns = [j for j, name in enumerate(old.CORE) if name != protocol]
        raw = cards[:, columns].mean(1)
        strength = np.log(np.clip(raw, .01, .99)/np.clip(1-raw, .01, .99))
        ability = strength[idx['held']]-strength[idx['historical']].mean()
        selected = cache.read(cache.ROOT/protocol/'selection_frozen.json')['fixed']['intercept']['config']
        fallback = None if selected['kind']=='none' else (selected['global_penalty'], 1.)
        for layout in ['legacy_shared_splits', 'independent_pair_splits']:
            if protocol == 'mixed' and layout != 'legacy_shared_splits':
                continue
            for seed, (legacy_s, legacy_q) in enumerate(orders):
                if layout == 'legacy_shared_splits':
                    ss = [np.asarray(legacy_s)]*24
                    qq = [np.asarray(legacy_q)]*24
                else:
                    assert len(np.unique(groups)) == len(groups)
                    ss, qq = [], []
                    for s in range(24):
                        rng = np.random.default_rng(640000+10000*old.PROTOCOLS.index(protocol)+100*seed+s)
                        order = rng.permutation(len(groups))
                        half = len(order)//2
                        ss.append(order[:31]); qq.append(order[half:])
                        assert half >=31
                assert all(set(groups[s]).isdisjoint(groups[q]) for s,q in zip(ss,qq))
                traces = {}
                for n in BUDGETS:
                    support = [s[:n] for s in ss]
                    observed = [np.array(target[s,j],float) for j,s in enumerate(support)]
                    rows = rows_for(z, np.zeros(24), support, observed)
                    named_rows = rows_for(z, ability, support, observed)
                    pooled = fit(rows)
                    named_pooled = fit(named_rows)
                    for j, query in enumerate(qq):
                        s = support[j]; y = observed[j]
                        assert set(s).isdisjoint(query)
                        lw = (y[:,None]*np.log(expert[s])+(1-y[:,None])*np.log1p(-expert[s])).sum(0)
                        w = np.exp(lw-logsumexp(lw))
                        cv = choose_from_support(np.empty((n,0)), np.empty((len(query),0)),
                             p0[s],p0[query], y[:,None], 'intercept', fallback)[0][:,0]
                        solos = [r for r in rows if r[0] == j]
                        named_solos = [r for r in named_rows if r[0] == j]
                        subjects = np.full(len(query), j, dtype=int)
                        pred = dict(rf_uniform=p0[query], rf_bayes=expert[query]@w, probability_cv=cv,
                            drift_solo=forecast(z,np.zeros(24),fit(solos),subjects,query),
                            drift_pooled=forecast(z,np.zeros(24),pooled,subjects,query),
                            named_prior=np.clip(expit(z[query]+ability[j]),.01,.99),
                            drift_named_solo=forecast(z,ability,fit(named_solos),subjects,query),
                            drift_named_pooled=forecast(z,ability,named_pooled,subjects,query))
                        known_other = set(i for k,items in enumerate(support) if k!=j for i in items)
                        overlap.append(dict(protocol=protocol, layout=layout, seed=seed, budget=n, model=int(idx['held'][j]),
                                            query_items=len(query), other_subject_labeled_items=len(set(query)&known_other)))
                        for method,p in pred.items():
                            assert np.isfinite(p).all() and np.min(p)>=0 and np.max(p)<=1
                            # Query truths enter scoring ONLY, never fit rows.
                            errors=(p-target[query,j])**2
                            for f in np.unique(family[query]):
                                records.append(dict(protocol=protocol,layout=layout,seed=seed,budget=n,
                                    model=int(idx['held'][j]),family=str(f),method=method,
                                    brier=float(errors[family[query]==f].mean())))
                            if seed==0:
                                traces[f'{method}_{n}_{j}']=p
                        if seed==0:
                            traces[f'query_{j}']=query
                            traces[f'support_{j}']=ss[j]
                if seed==0:
                    np.savez(ROOT/f'{protocol}_{layout}_seed0.npz', **traces)
                cache.save(ROOT/'status.json',dict(protocol=protocol,layout=layout,seed=seed+1,of=len(orders),records=len(records)))
                print('DONE',protocol,layout,seed+1,flush=True)
    cache.save(ROOT/'records.json',records)
    cache.save(ROOT/'overlap.json',overlap)
    summaries={}
    for layout in ['legacy_shared_splits','independent_pair_splits']:
        for group in ['cold_primary','cold_all_secondary','mixed']:
            subset=[r for r in records if r['layout']==layout and
                    ((r['protocol']=='mixed') if group=='mixed' else r['protocol']!='mixed') and
                    (group!='cold_primary' or r['protocol']!='MuSR')]
            if not subset:continue
            scores={}
            for method in METHODS:
                curve=[float(np.mean([r['brier'] for r in subset if r['method']==method and r['budget']==b])) for b in BUDGETS]
                scores[method]=dict(brier31=curve[-1],alc=float(old.ALC@curve),curve=curve)
            summaries[layout+':'+group]=scores
    control=cache.read(old.ROOT/'result.json')['summaries']
    for group in ['cold_primary','cold_all_secondary','mixed']:
        for current, ref in [('rf_bayes','bayes'),('rf_uniform','uniform'),('probability_cv','probability_cv_control')]:
            for metric in ['brier31','alc']:
                assert abs(summaries['legacy_shared_splits:'+group][current][metric]-control[group][ref][metric])<1e-12
    cache.save(ROOT/'result.json',dict(completed=True,summaries=summaries,plan=plan,
                                     legacy_controls_reproduced=True,records=len(records)))
    cache.save(ROOT/'status.json',dict(stage='complete'))
    for name,scores in summaries.items():
        for method,r in scores.items():print('RESULT',name,method,r['brier31'],r['alc'],flush=True)


if __name__=='__main__':
    if '--smoke' in sys.argv: print(json.dumps(smoke(),indent=2))
    else:run()

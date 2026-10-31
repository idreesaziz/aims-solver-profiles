"""Cached-prediction-only calibration comparison, with source-frozen selection."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import csv
import gc
import json
import sys
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from models import logistic_adaptation_v60 as alg
from scripts import demand_residual_adaptation_v48 as previous

ROOT = Path('D:/AIMS/logistic_adaptation_v60')
CORE = ['BBH', 'GPQA', 'MATH', 'MMLU-Pro', 'MuSR']
PROTOCOLS = CORE+['mixed']
BUDGETS = [0, 1, 3, 7, 15, 31]
ALC = np.array([.1, .2, .2, .2, .2, .1])
IDENTITY = dict(kind='identity', intercept_penalty=1., slope_penalty=1.)
GRID = [IDENTITY]+[
    dict(kind='intercept', intercept_penalty=a, slope_penalty=1.) for a in [1., 4., 16., 64.]
]+[
    dict(kind='affine', intercept_penalty=a, slope_penalty=b)
    for a in [1., 4., 16., 64.] for b in [4., 16., 64., 256.]
]
CV_GRID = GRID[:5]+[
    dict(kind='affine', intercept_penalty=a, slope_penalty=4*a) for a in [1., 4., 16., 64.]
]
METHODS = ['uniform', 'bayes', 'source_uniform_intercept', 'source_uniform_affine',
           'source_bayes_intercept', 'source_bayes_affine', 'source_budget_policy',
           'support_cv_logistic', 'probability_cv_control']


def save(p, value):
    previous.save(p, value)


def source_batch(protocol, idx):
    folder = previous.ROOT/protocol
    split = np.load(folder/'source_split.npz')
    episodes = np.load(folder/'validation_episodes.npz')
    s, q, h = episodes['s'], episodes['q'], episodes['h']
    positions = split['source_positions']
    assert len(h) > 0 and set(h).issubset(set(split['validation_models']))
    assert np.all(split['validation_mask'][s]) and np.all(split['validation_mask'][q])
    if protocol != 'mixed':
        assert np.all(idx['families'][positions] != protocol)
    groups = idx['groups'][positions]
    assert all(set(groups[a]).isdisjoint(groups[b]) for a, b in zip(s, q))
    p = np.load(folder/'oob_predictions.npy', mmap_mode='r')
    count = np.load(folder/'oob_counts.npy')
    assert np.min(count[s]) >= 8 and np.min(count[q]) >= 8
    ps, pq = np.clip(p[s], alg.EPS, 1-alg.EPS), np.clip(p[q], alg.EPS, 1-alg.EPS)
    mask = np.arange(192)[None, :] != h[:, None]
    ps = ps[np.broadcast_to(mask[:, None, :], ps.shape)].reshape(len(h), s.shape[1], 191)
    pq = pq[np.broadcast_to(mask[:, None, :], pq.shape)].reshape(len(h), q.shape[1], 191)
    labels = np.load(previous.BASE/'historical_training_labels.npy', mmap_mode='r')
    source_labels = np.asarray(labels[positions[s], h[:, None]], dtype=float)
    truth = np.asarray(labels[positions[q], h[:, None]], dtype=float)
    assert np.isfinite(ps).all() and np.isfinite(pq).all()
    return dict(ps=ps, pq=pq, labels=source_labels, truth=truth,
                source_questions=s, query_questions=q, historical_columns=h)


def source_select(protocol, idx):
    folder = ROOT/protocol
    folder.mkdir(exist_ok=True)
    if (folder/'selection.json').exists():
        return previous.read(folder/'selection.json')
    batch = source_batch(protocol, idx)
    candidates = []
    for base in ['uniform', 'bayes_loo']:
        bases = {n: alg.bases(batch['ps'][:, :n], batch['pq'], batch['labels'][:, :n], base) for n in BUDGETS}
        for cfg in GRID:
            curve = []
            for n in BUDGETS:
                ps, pq = bases[n]
                prediction = alg.predict(ps, pq, batch['labels'][:, :n], cfg)
                curve.append(float(np.square(prediction-batch['truth']).mean()))
            candidates.append(dict(base=base, config=cfg, curve=curve, alc=float(ALC @ curve)))
        print('SOURCE', protocol, base, flush=True)
        save(ROOT/'status.json', dict(stage='source_selection', protocol=protocol, base=base))
    selected = {}
    for method, base in [('source_uniform', 'uniform'), ('source_bayes', 'bayes_loo')]:
        for kind in ['intercept', 'affine']:
            allowed = [r for r in candidates if r['base'] == base and r['config']['kind'] in ['identity', kind]]
            selected[method+'_'+kind] = min(allowed, key=lambda r: r['alc'])
    schedule = [min(candidates, key=lambda r: r['curve'][j]) for j in range(len(BUDGETS))]
    fallback = [min([r for r in candidates if r['base'] == 'uniform'], key=lambda r: r['curve'][j])['config'] for j in range(len(BUDGETS))]
    selection = dict(selected=selected, schedule=schedule, cv_fallback=fallback, all_candidates=candidates,
                     source_episodes=len(batch['labels']), historical_pseudo_target_excluded=True)
    save(folder/'selection.json', selection)
    del batch
    gc.collect()
    return selection


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT/'result.json').exists():
        raise RuntimeError('Already complete; do not duplicate')
    files = [Path(__file__), REPO/'models/logistic_adaptation_v60.py',
             previous.BASE/'indices_and_weights.npz', previous.BASE/'historical_training_labels.npy',
             previous.BASE/'demand_manual/predictions.npy', previous.ROOT/'result.json',
             previous.ROOT/'support_cv_secondary/result.json']
    for name in PROTOCOLS:
        files.extend([previous.ROOT/name/'source_split.npz', previous.ROOT/name/'validation_episodes.npz',
                      previous.ROOT/name/'oob_predictions.npy', previous.ROOT/name/'oob_counts.npy'])
        if name != 'mixed':
            files.extend([previous.COLD/name/'predictions.npy', previous.COLD/name/'orders.npz', previous.COLD/name/'fit_receipt.json'])
    plan = dict(methods=METHODS, protocols=PROTOCOLS, grid=GRID, support_cv_grid=CV_GRID,
        budgets=BUDGETS, alc_weights=ALC.tolist(), primary='Four >=80-item benchmark-cold folds; MuSR and mixed secondary.',
        fit='Sum Bernoulli NLL plus ridge toward identity; monotone nonnegative total logit slope.',
        selection='Existing SOURCE group-OOB historical validation episodes, excluding pseudo-target expert. Fixed and per-budget choices selected by query Brier/ALC only; all choices sealed before current outer labels loaded.',
        bayesian_base='Calibrator support covariates are leave-one-label-out mixture predictions; query uses all currently acquired answers.',
        support_cv='Uniform RF-mean base only, exact leave-one-out Brier among 9 prespecified calibrators, current support only; below7answers frozen source fallback.',
        inspiration='Alvord public blog; this is NOT a reproduction of unpublished four-level hierarchical code.',
        caveats=['Inspected correlated development data, not formal blind confirmation.',
                 'Inner source forests use shared splits across192historical columns; inner validation not wholly model-disjoint.',
                 'Source episodic regularization selection uses known source benchmarks with group-OOB predictions, not nested leave-benchmark-out refits.',
                 'No pooling of other subjects acquired labels; item effects from competitor not reproduced.',
                 'No guarantee of reaching .15; all predeclared methods reported and no automatic deployment.'],
        gpu_used=False, paid_calls=0, protected_test_used=False, production_modified=False,
        frozen_files={str(p): previous.sha(p) for p in files})
    if (ROOT/'plan.json').exists():
        assert previous.read(ROOT/'plan.json') == plan
    else:
        save(ROOT/'plan.json', plan)
    return plan


def orders_for(protocol, idx):
    nf = len(idx['fit'])
    if protocol == 'mixed':
        from scripts.coordinate_brier_ablation_v42 import support_orders
        devpos = np.arange(len(idx['development']))
        family, groups = idx['families'][nf:], idx['groups'][nf:]
        orders = support_orders(family, groups)
        directory = previous.BASE/'demand_manual'
    else:
        z = np.load(previous.COLD/protocol/'orders.npz')
        devpos = z['development_positions']
        family, groups = idx['families'][nf:][devpos], idx['groups'][nf:][devpos]
        orders = list(zip(z['support'], z['evaluation']))
        directory = previous.COLD/protocol
    return devpos, family, groups, orders, directory


def summarize(records):
    summaries = {}
    for group in ['cold_primary', 'cold_all_secondary', 'mixed']:
        rows = [r for r in records if (r[0] == 'mixed' if group == 'mixed' else r[0] != 'mixed')
                and (group != 'cold_primary' or r[2] != 'MuSR')]
        methods = {}
        for method in METHODS:
            selected = [r for r in rows if r[1] == method]
            curve = {str(b): float(np.mean([r[6] for r in selected if r[4] == b])) for b in BUDGETS}
            per_family = {f: {str(b): float(np.mean([r[6] for r in selected if r[2] == f and r[4] == b])) for b in BUDGETS}
                          for f in sorted({r[2] for r in selected})}
            methods[method] = dict(brier31=curve['31'], alc=float(ALC @ list(curve.values())),
                                   curve=curve, per_family=per_family)
        summaries[group] = methods
    return summaries


def run():
    plan = prepare()
    idx = np.load(previous.BASE/'indices_and_weights.npz')
    assert len(idx['historical']) == 192 and len(idx['held']) == 24
    assert set(idx['historical']).isdisjoint(idx['held'])
    for name in PROTOCOLS:
        source_select(name, idx)
    save(ROOT/'prediction_freeze.json', dict(plan_sha256=previous.sha(ROOT/'plan.json'),
         selections={name: previous.sha(ROOT/name/'selection.json') for name in PROTOCOLS},
         timestamp=time.time(), outer_labels_loaded=False))
    # Only now open target response array; fitting functions never receive query labels.
    outer = np.load(previous.SWEEP/'development_outcomes.npy', mmap_mode='r')
    records, cv_counts = [], {}
    for name in PROTOCOLS:
        selected = previous.read(ROOT/name/'selection.json')
        devpos, family, groups, orders, directory = orders_for(name, idx)
        expert = np.clip(np.load(directory/'predictions.npy'), alg.EPS, 1-alg.EPS)
        target = outer[devpos]
        assert expert.shape == (len(devpos), 192) and target.shape == (len(devpos), 24)
        for seed, (support, evaluation) in enumerate(orders):
            assert len(support) == 31 and len(set(support)) == 31
            assert set(groups[support]).isdisjoint(groups[evaluation])
            traces = {}
            for j, n in enumerate(BUDGETS):
                observed = support[:n]
                labels = target[observed].T
                ps = np.broadcast_to(expert[observed], (24, n, 192))
                pq = expert[evaluation]
                uniform_s, uniform_q = alg.bases(ps, pq, labels, 'uniform')
                bayes_s, bayes_q = alg.bases(ps, pq, labels, 'bayes_loo')
                bases = {'uniform': (uniform_s, uniform_q), 'bayes_loo': (bayes_s, bayes_q)}
                predictions = dict(uniform=uniform_q, bayes=bayes_q)
                for method, setting in selected['selected'].items():
                    bs, bq = bases[setting['base']]
                    predictions[method] = alg.predict(bs, bq, labels, setting['config'])
                setting = selected['schedule'][j]
                bs, bq = bases[setting['base']]
                predictions['source_budget_policy'] = alg.predict(bs, bq, labels, setting['config'])
                cv_prediction, choices = alg.support_cv(uniform_s, uniform_q, labels, CV_GRID, selected['cv_fallback'][j])
                predictions['support_cv_logistic'] = cv_prediction
                cv_counts[name+':'+str(seed)+':'+str(n)] = np.bincount(choices+1, minlength=len(CV_GRID)+1).tolist()
                # Exact earlier probability-space support-CV baseline, frozen fallback.
                from scripts.support_cv_residual_v48b import choose_from_support
                old_selected = previous.read(previous.ROOT/name/'selection_frozen.json')['fixed']['intercept']['config']
                fallback = None if old_selected['kind'] == 'none' else (old_selected['global_penalty'], 1.)
                predictions['probability_cv_control'] = choose_from_support(
                    np.empty((n, 0)), np.empty((len(evaluation), 0)),
                    uniform_s[0], uniform_q[0], labels.T, 'intercept', fallback)[0].T
                assert set(predictions) == set(METHODS)
                for method, prediction in predictions.items():
                    assert prediction.shape == (24, len(evaluation)) and np.isfinite(prediction).all()
                    assert prediction.min() >= 0 and prediction.max() <= 1
                    if n == 0:
                        assert np.max(abs(prediction-uniform_q)) < 1e-12
                    traces[method+':'+str(n)] = prediction.astype(np.float64)
                    error = np.square(prediction-target[evaluation].T)
                    for f in np.unique(family[evaluation]):
                        mask = family[evaluation] == f
                        for h, score in zip(idx['held'], error[:, mask].mean(1)):
                            records.append((name, method, str(f), seed, n, int(h), float(score)))
            # Small monitor trajectories retained for independent optimizer replay.
            if seed == 0:
                np.savez(ROOT/name/'seed0_predictions.npz', support=support, evaluation=evaluation, **traces)
            print('EVAL', name, seed+1, '/', len(orders), flush=True)
            save(ROOT/'status.json', dict(stage='evaluation', protocol=name, seed=seed+1, total=len(orders)))
        del expert, target
        gc.collect()
    with (ROOT/'per_model.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['protocol', 'method', 'family', 'seed', 'budget', 'model', 'brier'])
        writer.writerows(records)
    summaries = summarize(records)
    old = previous.read(previous.ROOT/'result.json')['summaries']
    old_cv = previous.read(previous.ROOT/'support_cv_secondary/result.json')['summaries']
    for group in summaries:
        ref = next(r for r in old[group] if r['method'] == 'bayes')
        refcv = next(r for r in old_cv[group] if r['method'] == 'support_cv_intercept')
        for metric in ['brier31', 'alc']:
            assert abs(summaries[group]['bayes'][metric]-ref[metric]) < 1e-12
            assert abs(summaries[group]['probability_cv_control'][metric]-refcv[metric]) < 1e-12
    save(ROOT/'result.json', dict(completed=True, summaries=summaries, plan=plan,
                                 records=len(records), old_bayes_and_cv_controls_reproduced=True))
    save(ROOT/'support_cv_selections.json', cv_counts)
    save(ROOT/'status.json', dict(stage='complete', records=len(records)))
    for group, results in summaries.items():
        for method, scores in results.items():
            print('RESULT', group, method, scores['brier31'], scores['alc'], flush=True)


def smoke():
    from scipy.optimize import minimize
    from scipy.special import expit
    rng = np.random.default_rng(6060)
    p = rng.uniform(.01, .99, (5, 31))
    y = rng.integers(0, 2, p.shape).astype(float)
    y[0] = 0.; y[1] = 1.
    q = rng.uniform(.01, .99, (5, 11))
    max_beta_error = 0.
    for cfg in GRID[1:]:
        beta = alg.fit(alg.logits(p), y, cfg['kind'], cfg['intercept_penalty'], cfg['slope_penalty'])
        for i in range(5):
            z = alg.logits(p[i])
            a, b = cfg['intercept_penalty'], cfg['slope_penalty']
            def loss(v):
                eta = z+v[0]+v[1]*z
                return float(np.sum(np.logaddexp(0., eta)-y[i]*eta)+.5*(a*v[0]**2+b*v[1]**2))
            def gradient(v):
                r = expit(z+v[0]+v[1]*z)-y[i]
                return np.array([r.sum()+a*v[0], r@z+b*v[1]])
            bounds = [(None, None), (0., 0.) if cfg['kind'] == 'intercept' else (-1., None)]
            ref = minimize(loss, np.zeros(2), jac=gradient, bounds=bounds, method='L-BFGS-B',
                           options={'ftol': 1e-14, 'gtol': 1e-10, 'maxiter': 300})
            max_beta_error = max(max_beta_error, float(np.max(abs(beta[i]-ref.x))))
            assert abs(loss(beta[i])-loss(ref.x)) < 1e-9
        assert max_beta_error < 2e-5
        actual = alg.predict(p, q, y, cfg)
        order = rng.permutation(31)
        permuted = alg.predict(p[:, order], q, y[:, order], cfg)
        assert np.max(abs(actual-permuted)) < 1e-9
        assert np.array_equal(alg.predict(p[:, :0], q, y[:, :0], cfg), q)
    # Exact current-support LOO exclusion, not a diagonal/hat approximation.
    pred, choice = alg.support_cv(p[:2, :7], q[:2], y[:2, :7], CV_GRID, IDENTITY)
    risks = []
    for cfg in CV_GRID:
        leave = []
        for j in range(7):
            keep = np.arange(7) != j
            leave.append(alg.predict(p[:2, :7][:, keep], p[:2, j:j+1], y[:2, :7][:, keep], cfg)[:, 0])
        risks.append(np.square(np.array(leave).T-y[:2, :7]).mean(1))
    assert np.array_equal(choice, np.argmin(risks, axis=0))
    experts = rng.uniform(.02, .98, (3, 7, 13))
    queries = rng.uniform(.02, .98, (3, 8, 13))
    labels = rng.integers(0, 2, (3, 7))
    bs, bq = alg.bases(experts, queries, labels, 'bayes_loo')
    for j in range(7):
        altered = labels.copy(); altered[:, j] = 1-altered[:, j]
        changed = alg.bases(experts, queries, altered, 'bayes_loo')[0]
        assert np.max(abs(changed[:, j]-bs[:, j])) < 1e-12
    for i in range(3):
        w = np.ones(13)/13
        for j in range(7):
            w *= experts[i, j] if labels[i, j] else 1-experts[i, j]
            w /= w.sum()
        assert np.max(abs(queries[i]@w-bq[i])) < 1e-12
    print('SMOKE PASS: 100 scipy optimizer fits; support LOO; zero budget; permutation; Bayes leave-self-out. max beta discrepancy', max_beta_error)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['smoke', 'run'])
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.action == 'smoke':
            smoke()
        else:
            run()

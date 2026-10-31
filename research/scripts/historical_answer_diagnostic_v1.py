"""Outcome-only exploratory diagnostic. No question-text features or encoder fit."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logsumexp

OUT = Path('E:/AIMS/historical_answer_diagnostic_v1')
REPORT = Path('reports/historical_answer_diagnostic_v1')
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = np.array([.1, .2, .2, .2, .2, .1])

def digest(x):
    return int.from_bytes(hashlib.sha256(str(x).encode()).digest()[:8], 'little')

def solver_groups(subjects):
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def key(x):
        return re.sub('[^a-z0-9]', '', str(x).rsplit(' / ', 1)[-1].lower())
    for s in subjects:
        a = 'raw:' + key(s.get('model') or s.get('display_name'))
        find(a)
        if s.get('normalized_name'):
            b = 'norm:' + key(s['normalized_name'])
            x, y = find(a), find(b)
            parent[max(x, y)] = min(x, y)
    return [find('raw:' + key(s.get('model') or s.get('display_name'))) for s in subjects]

def question_groups(path, name):
    schema = pq.read_schema(path/'questions.parquet').names
    if name == 'babilong_panel':
        t = pq.read_table(path/'input_groups.parquet').to_pandas()
        q = pq.read_table(path/'questions.parquet', columns=['question_id']).to_pandas()
        return q.merge(t, on='question_id', validate='one_to_one')['underlying_context_question_group'].astype(str).to_numpy()
    for field in ['underlying_group', 'underlying_source_group']:
        if field in schema:
            return np.array(pq.read_table(path/'questions.parquet', columns=[field])[field].to_pylist(), dtype=str)
    rows = pq.read_table(path/'questions.parquet', columns=['source_raw_ids']).to_pylist()
    raw = [r['source_raw_ids'][0] for r in rows]
    if name == 'cruxeval_panel':
        return np.array([r.rsplit('/', 1)[-1] for r in raw])
    if name == 'erbench_panel':
        return np.array([':'.join(r.split(':')[:2]) for r in raw])
    if name == 'worldcentralbanks_panel':
        # Conservatively keep all released rows of the same source CSV together.
        return np.array([r.rsplit('/row/', 1)[0] for r in raw])
    return np.array(raw)

def expected_brier(p, r):
    return float(np.mean(p*p - 2*p*r + r))

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path('E:/AIMS/fingerprint_sources_v2/manifest.json').read_text())
    original_hold = json.loads(Path('artifacts/demand_capability_v1/prepared.json').read_text())['heldout_model_columns']
    plan = dict(stage='historical answers only', epsilon=.1, seeds=5,
                budgets=BUDGETS, max_evaluation_questions=2048,
                auxiliary_protocol='leave one metadata alias group out; each active solver scored',
                aligned_protocol='original 24 target columns collectively excluded, plus their metadata aliases',
                support_protocol='hash-frozen 40% question groups; one random input per group, 31 groups',
                evaluation_protocol='remaining 60% groups; up to 2048 distinct inputs per target',
                missing_policy='absent expert predictions fall back to historical item mean; absent target cells never scored',
                qualification='Exploratory source-wise information diagnostic, not encoder performance, untouched confirmation, or official score. Panels are not aligned by this procedure. Fixed smoothing is not tuned on targets.')
    (OUT/'plan.json').write_text(json.dumps(plan, indent=2))
    records, audits = [], []
    for panel in manifest['panels']:
        name, path = panel['panel_id'], Path(panel['path'])
        y = np.load(path/'profiles.npy').astype(np.float64)
        mask = np.isfinite(y)
        assert np.all((y[mask] >= 0) & (y[mask] <= 1))
        subjects = json.loads((path/'subjects.json').read_text())
        assert len(subjects) == y.shape[1]
        sg = solver_groups(subjects)
        groups = question_groups(path, name)
        assert len(groups) == len(y)
        support_role = np.array([digest('role42:'+g) % 10 < 4 for g in groups])
        active = np.flatnonzero(mask.sum(axis=0) >= 100)
        targets = original_hold if name == 'aligned_216' else active
        excluded = {sg[j] for j in original_hold} if name == 'aligned_216' else set()
        done, skipped = 0, []
        for target in targets:
            historical = np.array([j for j in active if sg[j] not in excluded | {sg[target]}], dtype=int)
            if len(historical) < 2:
                skipped.append(dict(target=int(target), reason='fewer than two independent historical IDs'))
                continue
            hm = mask[:, historical]
            counts = hm.sum(axis=1)
            eligible = mask[:, target] & (counts > 0)
            support_pool = np.flatnonzero(eligible & support_role)
            eval_pool = np.flatnonzero(eligible & ~support_role)
            if len(np.unique(groups[support_pool])) < 31 or len(eval_pool) < 64:
                skipped.append(dict(target=int(target), reason='insufficient disjoint observed groups'))
                continue
            hy = y[:, historical]
            item = np.divide(np.where(hm, hy, 0).sum(axis=1), counts,
                             out=np.full(len(y), .5), where=counts > 0)
            # A fallback prediction is not an observed training label.
            expert = np.where(hm, .1 + .8*hy, (.1 + .8*item)[:, None])
            assert np.all(np.isfinite(expert))
            assert target not in historical
            assert not ({sg[j] for j in historical} & (excluded | {sg[target]}))
            for seed in range(5):
                rng = np.random.default_rng(digest(f'{name}:{target}:{seed}') % 2**32)
                candidate = rng.permutation(support_pool)
                _, first = np.unique(groups[candidate], return_index=True)
                support = candidate[np.sort(first)[:31]]
                evaluation = rng.permutation(eval_pool)[:2048]
                assert not set(groups[support]) & set(groups[evaluation])
                rate = y[evaluation, target]
                observed = rng.binomial(1, y[support, target])
                logw = np.zeros(len(historical))
                correct = 0
                for budget in range(32):
                    if budget in BUDGETS:
                        w = np.exp(logw - logsumexp(logw))
                        pred = np.sum(expert[evaluation]*w[None, :], axis=1)
                        difficulty = .1 + .8*item[evaluation]
                        # Global scalar ability baseline with a uniform Beta(1,1) prior.
                        scalar = np.full(len(evaluation), (1+correct)/(2+budget))
                        for method, p in [('historical_mixture', pred), ('historical_item_mean', difficulty), ('scalar_beta', scalar)]:
                            score = expected_brier(p, rate)
                            mse = float(np.mean((p-rate)**2))
                            noise = float(np.mean(rate*(1-rate)))
                            assert abs(score-mse-noise) < 1e-12
                            records.append(dict(panel=name, target=int(target), solver_id=subjects[target].get('canonical_column_id', subjects[target].get('subject_id')), seed=seed, budget=budget, method=method, brier=score, rate_mse=mse, rate_variance=noise, n_eval=len(rate), mean_historical_coverage=float(counts[evaluation].mean())))
                    if budget == 31:
                        break
                    pi = expert[support[budget]]
                    z = observed[budget]
                    logw += z*np.log(pi)+(1-z)*np.log1p(-pi)
                    correct += z
            done += 1
        audit = dict(panel=name, questions=len(y), configurations=y.shape[1], active_configurations=len(active), targets_scored=done, skipped=skipped, observed_cells=int(mask.sum()), support_group_count=len(set(groups[support_role])), evaluation_group_count=len(set(groups[~support_role])), solver_groups=sg)
        audits.append(audit)
        pd.DataFrame(records).to_csv(OUT/'per_solver.csv', index=False)
        print(name, 'targets', done, 'skipped', len(skipped), flush=True)
    df = pd.DataFrame(records)
    summary = []
    for (panel, method), sub in df.groupby(['panel', 'method']):
        curve = sub.groupby('budget').brier.mean().reindex(BUDGETS)
        summary.append(dict(panel=panel, method=method, brier0=float(curve.loc[0]), brier31=float(curve.loc[31]), alc=float(curve.to_numpy()@WEIGHTS), targets=int(sub.target.nunique())))
    result = dict(plan=plan, audits=audits, summary=summary)
    (OUT/'result.json').write_text(json.dumps(result, indent=2))
    table = pd.DataFrame(summary)
    table.to_csv(REPORT/'summary.csv', index=False)
    text = '# Historical-answer-only diagnostic\n\nNo text embeddings, question encoder, or neural-network training. Existing Bayesian historical-response mixture with fixed 0.1 smoothing. Missing historical entries use an explicit item-mean prediction fallback; missing target entries are excluded.\n\n'
    text += 'This exploratory test uses released answers for evaluation questions from other solvers. It is an information diagnostic, not a deployed question predictor. Source-specific IDs remain distinct; no cross-panel latent alignment is claimed. Absolute Briers across sources do not establish a better training corpus.\n\n'
    text += '| Source | Readout | Targets | Brier 0 | Brier 31 | ALC |\n|---|---|---:|---:|---:|---:|\n'
    for r in summary:
        text += f"| {r['panel']} | {r['method']} | {r['targets']} | {r['brier0']:.5f} | {r['brier31']:.5f} | {r['alc']:.5f} |\n"
    text += '\nOriginal 24 SOS targets collectively excluded. Auxiliary targets use leave-one-metadata-alias-group-out. Five support seeds; up to 2,048 evaluation inputs per target. Group-disjoint support/evaluation, but related examples can remain and source dependence is not a confidence guarantee. Released fractional success rates are scored with expected binary Brier; support responses are Bernoulli resamples, not newly collected answers. Fixed smoothing and fallback may be suboptimal: poor performance would not prove absent information. No official competition score.\n'
    (REPORT/'report.md').write_text(text)
    print(table.to_string(index=False), flush=True)

if __name__ == '__main__':
    main()

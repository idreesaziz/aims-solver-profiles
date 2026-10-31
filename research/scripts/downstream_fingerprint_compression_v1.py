"""Compress the saved oracle question representation; score NEW-MODEL Brier.

This is a matched compression ablation, not a text-to-question encoder.
Historical-only prior and source question features determine every rotation.
The original 24 held-out model columns supply only support/evaluation labels.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import qr
from scipy.special import expit
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_delean_shape_predictors import mapped_splits
from scripts.evaluate_demand_capability import representation, offset_update

OUT = ROOT / 'artifacts/downstream_fingerprint_compression_v1'
REPORT = ROOT / 'reports/downstream_fingerprint_compression_v1'
RANKS = list(range(26))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def update_batch(mean, covariance, features, offset, outcomes):
    probability = expit(offset + mean @ features)
    v = covariance @ features
    variance = probability * (1 - probability)
    scale = variance / (1 + variance * (v @ features))
    new = covariance - scale[:, None, None] * v[:, :, None] * v[:, None, :]
    mean = mean + (new @ features) * (outcomes - probability)[:, None]
    return mean, (new + new.transpose(0, 2, 1)) / 2


def unit_check_batch():
    rng = np.random.default_rng(72)
    for dimension in (1, 4, 9):
        mean = rng.normal(size=(5, dimension))
        a = rng.normal(size=(5, dimension, dimension))
        covariance = a @ a.transpose(0, 2, 1) + .1 * np.eye(dimension)
        x = rng.normal(size=dimension)
        z = rng.integers(0, 2, 5)
        actual = update_batch(mean, covariance, x, -.4, z)
        for j in range(5):
            expected = offset_update(mean[j], covariance[j], x, -.4, z[j])
            assert np.allclose(actual[0][j], expected[0], atol=1e-12)
            assert np.allclose(actual[1][j], expected[1], atol=1e-12)


def main():
    start = time.time()
    if OUT.exists() or REPORT.exists():
        raise RuntimeError('Frozen experiment exists; do not overwrite.')
    unit_check_batch()
    OUT.mkdir(parents=True)
    REPORT.mkdir(parents=True)
    old = ROOT / 'artifacts/demand_capability_v1'
    config = json.loads((ROOT / 'config_demand_capability_v1.json').read_text())
    prepared = json.loads((old / 'prepared.json').read_text())
    historical = np.array(prepared['historical_columns'])
    held = np.array(prepared['heldout_model_columns'])
    assert not set(historical) & set(held) and len(held) == 24
    source = ROOT / config['source']
    families = np.load(source / 'families.npy')
    keep = np.load(source / 'snapshot_indices.npy')
    splits = mapped_splits(ROOT / config['split_source'], keep)
    budgets = config['budgets']
    alc_weights = np.array(config['alc_weights'])
    input_paths = [old / s['name'] / 'ORACLE_question_representation.npz' for s in splits]
    input_paths += [ROOT / 'reports/demand_capability_v1/ORACLE_per_model.csv']
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in input_paths}
    plan = dict(objective='Minimize downstream Brier and identify compact near-best representations.',
        ranks=RANKS, question_numbers_per_vector='1 offset plus min(rank, original parameter dimension) adaptive coordinates.',
        compression='Whiten frozen historical Gaussian prior; SVD of source-refit weighted logit sensitivity selects nested directions. No held-out model labels select directions.',
        adaptation='Exactly original Bayesian online Laplace updates, 24 held-out models, 5 seeds, original support/evaluation sets and budgets.',
        comparison_thresholds=[.001, .002, .005], absolute_brier_target=.05,
        status='Exploratory compression on an already examined evaluation protocol; not fresh confirmation.',
        oracle='Target question representations still use historical outcomes on the target question. Not a deployed text-only encoder.',
        anchors='A mathematical signed linear basis of actual source questions; not clusters or a nearest-reference classifier.',
        input_hashes=input_hashes, code_sha256=sha(Path(__file__)))
    save(OUT / 'plan.json', plan)

    encodings = {}
    anchor_records = []
    for split in splits:
        name = split['name']
        rep = representation(old / name / 'ORACLE_question_representation.npz')
        # theta = historical mean + L @ standardized theta; standardized prior is N(0,I).
        factor = np.linalg.cholesky(rep['cov'])
        features = rep['features'] @ factor
        offset = rep['offset'] + rep['features'] @ rep['mean']
        rf = split['refit']
        weights = np.zeros(len(rf))
        for family in np.unique(families[rf]):
            mask = families[rf] == family
            weights[mask] = 1 / (len(np.unique(families[rf])) * mask.sum())
        _, singular, vt = np.linalg.svd(np.sqrt(weights)[:, None] * features[rf], full_matrices=False)
        full = features @ vt.T
        assert np.allclose(full @ full.T[:,:3], features @ features.T[:,:3], atol=1e-10)
        dimension = full.shape[1]
        fold = OUT / name
        fold.mkdir()
        bases = {}
        for rank in range(dimension + 1):
            encoded = np.c_[offset, full[:, :rank]]
            _, _, pivots = qr(encoded[rf].T, mode='economic', pivoting=True)
            selected = rf[pivots[:rank+1]]
            basis = encoded[selected]
            assert np.linalg.matrix_rank(basis) == rank + 1
            coordinates = np.linalg.solve(basis.T, encoded.T).T
            error = float(np.max(np.abs(coordinates @ basis - encoded)))
            assert error < 1e-9
            bases[str(rank)] = dict(anchor_shape_indices=selected.tolist(),
                anchor_original_local_indices=keep[selected].tolist(), actual_question_count=rank+1,
                condition_number=float(np.linalg.cond(basis)), maximum_reconstruction_error=error)
            anchor_records.append(dict(fold=name, rank=rank, **bases[str(rank)]))
        np.savez_compressed(fold / 'encoding.npz', offset=offset, adaptive_features=full,
            historical_mean=rep['mean'], cholesky=factor, rotation=vt.T,
            source_singular_values=singular, source_refit_rows=rf)
        save(fold / 'bases.json', bases)
        encodings[name] = (offset, full)
    save(OUT / 'anchors.json', anchor_records)
    # Freeze all candidate encoders before reading held-out support/evaluation outcomes.
    save(OUT / 'encodings_frozen.json', {s['name']: sha(OUT/s['name']/'encoding.npz') for s in splits})
    y = np.load(source / 'y.npy', mmap_mode='r')
    old_rows = {}
    with input_paths[-1].open(newline='', encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            if row['method'] == 'oracle_capability':
                key = (row['protocol'], row['family'], int(row['seed']), int(row['model_column']), int(row['budget']))
                old_rows[key] = float(row['brier'])
    rows = []
    maximum_original_difference = 0.
    for split in splits:
        name = split['name']
        offset, full = encodings[name]
        dimension = full.shape[1]
        print('EVALUATING', name, 'full adaptive dimension', dimension, flush=True)
        for family in np.unique(families[split['test']]):
            ids = split['test'][families[split['test']] == family]
            for seed in range(config['adaptation_seeds']):
                rng = np.random.default_rng(config['seed'] + seed + sum(map(ord, name + str(family))))
                order = rng.permutation(ids)
                support = order[:len(ids)//2][:31]
                evaluation = order[len(ids)//2:]
                assert len(support) == 31 and not set(support) & set(evaluation)
                targets = y[evaluation][:, held]
                observed = y[support][:, held]
                for rank in range(dimension + 1):
                    features = full[:, :rank]
                    mean = np.zeros((len(held), rank))
                    covariance = np.broadcast_to(np.eye(rank), (len(held), rank, rank)).copy()
                    for budget in range(32):
                        if budget in budgets:
                            probabilities = expit(offset[evaluation, None] + features[evaluation] @ mean.T)
                            errors = np.mean((probabilities - targets)**2, axis=0)
                            for j, model in enumerate(held):
                                if rank == dimension:
                                    key = (split['protocol'], str(family), seed, int(model), budget)
                                    maximum_original_difference = max(maximum_original_difference,
                                        abs(float(errors[j]) - old_rows[key]))
                                rows.append(dict(fold=name, protocol=split['protocol'], family=str(family),
                                    seed=seed, model_column=int(model), budget=budget, actual_rank=rank,
                                    full_rank=dimension, question_vector_numbers=rank+1,
                                    evaluation_questions=len(evaluation), brier=float(errors[j])))
                        if budget == 31:
                            break
                        if rank:
                            i = support[budget]
                            mean, covariance = update_batch(mean, covariance, features[i], offset[i], observed[budget])
        save(OUT / 'status.json', dict(last_completed_fold=name, rows=len(rows), elapsed_seconds=time.time()-start))
    assert maximum_original_difference < 1e-9, maximum_original_difference
    with (OUT / 'per_model.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_cap = defaultdict(list)
    for row in rows:
        for cap in RANKS:
            if row['actual_rank'] == min(cap, row['full_rank']):
                by_cap[row['protocol'], cap, row['budget']].append(row['brier'])
    summary = []
    for protocol in ('within_benchmark', 'heldout_benchmark'):
        for cap in RANKS:
            scores = [float(np.mean(by_cap[protocol, cap, b])) for b in budgets]
            summary.append(dict(protocol=protocol, adaptive_dimension_cap=cap,
                question_vector_numbers_cap=cap+1, brier_by_budget=dict(zip(map(str,budgets), scores)),
                brier_alc=float(alc_weights @ scores)))
    decisions = {}
    for protocol in ('within_benchmark', 'heldout_benchmark'):
        relevant = [r for r in summary if r['protocol'] == protocol]
        decisions[protocol] = {}
        for metric in ('brier31', 'brier_alc'):
            def value(row):
                return row['brier_by_budget']['31'] if metric == 'brier31' else row['brier_alc']
            best = min(relevant, key=value)
            decisions[protocol][metric] = dict(best=value(best), best_rank_cap=best['adaptive_dimension_cap'],
                smallest_cap_under_005=next((r['adaptive_dimension_cap'] for r in relevant if value(r)<=.05),None),
                smallest_cap_within_best={str(e):next(r['adaptive_dimension_cap'] for r in relevant if value(r)<=value(best)+e)
                                          for e in (.001,.002,.005)})
    for path in input_paths:
        assert sha(path) == input_hashes[str(path.relative_to(ROOT))]
    result = dict(results=summary, decisions=decisions, verification=dict(
        maximum_full_dimension_difference_from_original=maximum_original_difference,
        batched_update_checked_against_original=True, original_artifacts_unchanged=True,
        heldout_model_columns_excluded_from_compression=True),
        seconds=time.time()-start, qualifications=plan)
    save(OUT / 'result.json', result)
    report = ['# Compact question representations: downstream Brier', '',
        'Matched ablation of the saved oracle experiment: 24 held-out models, 5 seeds, original benchmark folds and 31 support observations.', '',
        'Each question has one fixed offset plus k adaptive coordinates. Source-only SVD orders whitened prior prediction directions. A cap above a fold\'s original rank has no additional effect (original maximum adaptive rank is 9 or 25).', '',
        '| Adaptive dimensions (cap) | Question numbers (cap) | Held-benchmark Brier @31 | Held-benchmark Brier ALC | Within-benchmark Brier @31 |',
        '|---:|---:|---:|---:|---:|']
    for cap in RANKS:
        transfer = next(r for r in summary if r['protocol']=='heldout_benchmark' and r['adaptive_dimension_cap']==cap)
        within = next(r for r in summary if r['protocol']=='within_benchmark' and r['adaptive_dimension_cap']==cap)
        report.append(f'| {cap} | {cap+1} | {transfer["brier_by_budget"]["31"]:.6f} | {transfer["brier_alc"]:.6f} | {within["brier_by_budget"]["31"]:.6f} |')
    report += ['', '## Limits', '',
        'The original low Brier is an oracle result: historical model answers on the target question determine its representation. This experiment measures how compact that information can be while preserving downstream adaptation; it does not solve question-text encoding.', '',
        'All dimensions are exploratory on the same previously inspected evaluation panel. Smallest tested is not a proof of the minimum over every nonlinear model. ALC and Brier after 31 labels are different objectives.', '',
        'The actual-question bases span these numeric representations through signed coefficients. They do not define semantic clusters, nearest-question assignments, or a method to infer coefficients for new text.', '',
        f'Full-dimensional reproduction maximum per-pair Brier difference: {maximum_original_difference:.3g}. No original files changed or API calls made.', '',
        'Threshold decisions:','```json',json.dumps(decisions,indent=2),'```']
    (REPORT / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for protocol in ('heldout_benchmark','within_benchmark'):
        values = [r for r in summary if r['protocol']==protocol]
        for ax, metric in zip(axes,('brier31','brier_alc')):
            ax.plot([r['question_vector_numbers_cap'] for r in values],
                    [r['brier_by_budget']['31'] if metric=='brier31' else r['brier_alc'] for r in values],
                    marker='.',label=protocol.replace('_',' '))
            ax.set(xlabel='Question representation numbers (maximum)',ylabel='Brier (lower is better)',
                   title='After 31 observations' if metric=='brier31' else 'Across observation budgets (ALC)')
            ax.axhline(.05,color='gray',linestyle=':')
    axes[0].legend()
    fig.savefig(REPORT / 'brier_vs_dimensions.png',dpi=170)
    plt.close(fig)
    print(json.dumps(dict(decisions=decisions, verification=result['verification'], seconds=result['seconds']),indent=2),flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()

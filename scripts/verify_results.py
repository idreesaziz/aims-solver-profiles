"""Recompute released table values and paired source-group intervals."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = dict(zip(BUDGETS, [.1, .2, .2, .2, .2, .1]))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def near(a, b):
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)


def main():
    core = pd.DataFrame(read(R / 'drift_exact_prior_v80/comparison_records.json'))
    summary = read(R / 'drift_exact_prior_v80/comparison.json')['results']
    for cohort, methods in summary.items():
        for method, expected in methods.items():
            rows = core[(core.cohort == int(cohort)) & (core.method == method)]
            curve = rows.groupby('budget').brier.mean().reindex(BUDGETS).to_numpy()
            near(curve, expected['curve'])
            near(curve @ list(WEIGHTS.values()), expected['alc'])
            near(curve[-1], expected['brier31'])
    core['weighted'] = core.brier * core.budget.map(WEIGHTS)
    paired = core.groupby(['benchmark', 'cohort', 'seed', 'model', 'method']).weighted.sum().unstack('method')
    effects = (paired.drift_prior_our52_update - paired.drift_prior_drift_update).groupby(['benchmark', 'cohort']).mean()
    provenance = read(ROOT / 'report/report_provenance.json')['statistics']['core_paired_effects']
    for benchmark, cohorts in provenance.items():
        for cohort, value in cohorts.items():
            near(effects.loc[benchmark, int(cohort)], value)

    frame = pd.DataFrame(read(R / 'broad_prior_audit_v81/v81_validation_records.json'))
    frame['weighted'] = frame.brier * frame.budget.map(WEIGHTS)
    paired = frame.groupby(['benchmark', 'cohort', 'seed', 'pair', 'method']).weighted.sum().unstack('method')
    plan = read(R / 'broad_prior_audit_v81/plan.json')
    expected = read(R / 'recent_run_diagnosis_v86/result.json')['neutral_vs_drift_cluster_bootstrap']
    rng = np.random.default_rng(860)
    for cohort in [1, 3]:
        rows = paired.xs(cohort, level='cohort')
        delta = rows.neutral - rows.drift
        groups = np.array([plan['groups'][b] for b in delta.index.get_level_values('benchmark')])
        unique = np.unique(groups)
        sums = np.array([delta.to_numpy()[groups == group].sum() for group in unique])
        counts = np.array([(groups == group).sum() for group in unique])
        picks = rng.integers(len(unique), size=(10000, len(unique)))
        boots = sums[picks].sum(1) / counts[picks].sum(1)
        near(delta.mean(), expected[str(cohort)]['neutral_minus_drift'])
        near(np.quantile(boots, [.025, .975]), expected[str(cohort)]['interval95'])
        assert len(unique) == expected[str(cohort)]['groups'] == 41

    oracle = pd.read_csv(R / 'oracle/per_model.csv')
    summaries = read(R / 'oracle/result.json')['summaries']
    for expected in summaries:
        rows = oracle[(oracle.cohort == expected['cohort']) & (oracle.method == expected['method'])]
        curve = rows.groupby('budget').brier.mean().reindex(BUDGETS)
        near(curve.loc[31], expected['brier31'])
        near(curve.to_numpy() @ list(WEIGHTS.values()), expected['alc'])

    hosted = read(R / 'hosted_budget_means.json')
    near(sum(hosted['curve'][str(b)] * w for b, w in WEIGHTS.items()), hosted['brier_alc'])
    archive = ROOT / 'submission/submission_drift_prior_ours_v82.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == 'b0aef886620c7ba3545ab7b132336b969079ee5160d758d67125703b51a9ffd1'
    print(json.dumps(dict(passed=True, core_table=True, paired_core_effects=True,
                          source_group_bootstrap=True, oracle_table=True, hosted_alc=True,
                          exact_submitted_archive=True), indent=2))


if __name__ == '__main__':
    main()

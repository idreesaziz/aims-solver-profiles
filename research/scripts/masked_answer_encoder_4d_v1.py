"""Matched 4-number HISTORICAL-ANSWER encoder: aligned panel vs full masked pool.

No text features, text encoder, API calls, or held-out solver training labels.
"""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[k] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import copy
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch import nn
from scipy.special import expit
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_delean_shape_predictors import mapped_splits
from scripts.downstream_fingerprint_compression_v1 import update_batch, unit_check_batch
from scripts.historical_answer_diagnostic_v1 import solver_groups, digest

OUT = Path('E:/AIMS/masked_answer_encoder_4d_v1')
REPORT = ROOT/'reports/masked_answer_encoder_4d_v1'
BUDGETS = [0, 1, 3, 7, 15, 31]
ALC = np.array([.1, .2, .2, .2, .2, .1])

def save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(path)

class AnswerEncoder(nn.Module):
    def __init__(self, columns):
        super().__init__()
        # Signed outcomes and a separate observed mask preserve a missing vs zero distinction.
        self.net = nn.Sequential(nn.Linear(2*columns, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 4))
        self.ability = nn.Parameter(torch.zeros(columns))
        self.model_vectors = nn.Parameter(torch.randn(columns, 3)*.1)

    def encode(self, values, mask, columns):
        x = torch.zeros((len(values), self.ability.numel()*2), device=values.device)
        safe = torch.where(mask, 2*values-1, torch.zeros_like(values))
        x[:, columns] = safe
        x[:, self.ability.numel()+columns] = mask.float()
        return self.net(x)

    def logits(self, coordinates, columns):
        return coordinates[:, 0, None]+self.ability[columns]+coordinates[:, 1:]@self.model_vectors[columns].T

def observed_loss(logits, values, mask):
    safe = torch.where(mask, values, torch.zeros_like(values))
    p = torch.sigmoid(logits)
    # Proper expected binary Brier for empirical success rates, masked per question.
    loss = torch.where(mask, p*p-2*p*safe+safe, torch.zeros_like(p))
    return (loss.sum(1)/mask.sum(1).clamp(min=1)).mean()

class Cycle:
    def __init__(self, ids, seed):
        self.ids = np.asarray(ids)
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.ids)
        self.pos = 0
        self.seen = set()

    def get(self, count):
        chunks = []
        while count:
            n = min(count, len(self.order)-self.pos)
            v = self.order[self.pos:self.pos+n]
            chunks.append(v)
            self.seen.update(map(int, v))
            self.pos += n
            count -= n
            if self.pos == len(self.order):
                self.order = self.rng.permutation(self.ids)
                self.pos = 0
        return np.concatenate(chunks)

def prepare():
    manifest = json.loads(Path('E:/AIMS/fingerprint_sources_v2/manifest.json').read_text())
    prep = json.loads((ROOT/'artifacts/demand_capability_v1/prepared.json').read_text())
    held = np.array(prep['heldout_model_columns'])
    metadata = []
    panels = []
    for p in manifest['panels']:
        path = Path(p['path'])
        subjects = json.loads((path/'subjects.json').read_text())
        start = len(metadata)
        metadata.extend(subjects)
        panels.append(dict(name=p['panel_id'], path=path, columns=np.arange(start, len(metadata)),
                           y=np.load(path/'profiles.npy').astype(np.float32)))
    groups = solver_groups(metadata)
    held_groups = {groups[j] for j in held}
    kept = [j for j, g in enumerate(groups) if g not in held_groups]
    remap = {old: i for i, old in enumerate(kept)}
    for p in panels:
        ix = [j for j, c in enumerate(p['columns']) if c in remap and np.isfinite(p['y'][:, j]).any()]
        original = p['columns'][ix]
        p['columns'] = np.array([remap[c] for c in original])
        p['original_columns'] = original
        p['y'] = p['y'][:, ix]
        p['mask'] = np.isfinite(p['y'])
        p['usable'] = np.flatnonzero(p['mask'].any(1))
        p['yt'] = torch.tensor(np.nan_to_num(p['y']), device='cuda')
        p['mt'] = torch.tensor(p['mask'], device='cuda')
        p['ct'] = torch.tensor(p['columns'], device='cuda')
    aligned = panels[0]
    assert aligned['name'] == 'aligned_216'
    q = pq.read_table(aligned['path']/'questions.parquet', columns=['source_panel', 'original_core_rows', 'underlying_group']).to_pandas()
    oldrows = np.load(ROOT/'artifacts/delean_individual_v1/source_indices.npy')[
        np.load(ROOT/'artifacts/delean_shape_v1/snapshot_indices.npy')]
    old_to_clean = {int(old): i for i, rows in enumerate(q.original_core_rows) for old in rows}
    valid_old = np.array([i for i, old in enumerate(oldrows) if int(old) in old_to_clean])
    evaluation_map = np.array([old_to_clean[int(oldrows[i])] for i in valid_old])
    oldy = np.load(ROOT/'artifacts/delean_shape_v1/y.npy')[valid_old]
    fully = np.load(aligned['path']/'profiles.npy', mmap_mode='r')
    assert np.array_equal(oldy, fully[evaluation_map])
    families = np.load(ROOT/'artifacts/delean_shape_v1/families.npy')[valid_old]
    splits = mapped_splits(ROOT/'artifacts/delean_individual_v1', np.load(ROOT/'artifacts/delean_shape_v1/snapshot_indices.npy'))
    valid_lookup = {int(old): new for new, old in enumerate(valid_old)}
    for split in splits:
        for role in ['fit', 'tune', 'calibrate', 'refit', 'test']:
            split[role] = np.array([valid_lookup[int(i)] for i in split[role] if int(i) in valid_lookup])
    # Auxiliary rows retain separate IDs; shared network is tested, not assumed aligned.
    aux_offsets = np.cumsum([0]+[len(p['usable']) for p in panels[1:]])
    return panels, len(kept), held, q, evaluation_map, oldy, families, splits, aux_offsets, metadata, kept

def encode_rows(net, panel, rows):
    net.eval()
    result = []
    with torch.no_grad():
        for chunk in np.array_split(np.asarray(rows), max(1, int(np.ceil(len(rows)/2048)))):
            ix = torch.tensor(chunk, device='cuda')
            result.append(net.encode(panel['yt'][ix], panel['mt'][ix], panel['ct']).cpu().numpy())
    return np.concatenate(result).astype(float)

def fit(panels, ncols, train, calibration, aux_offsets, condition, seed, folder):
    torch.manual_seed(4251+10000*seed)
    net = AnswerEncoder(ncols).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.001)
    core = panels[0]
    corecycle = Cycle(train, 42+seed)
    auxcycle = Cycle(np.arange(aux_offsets[-1]), 142+seed)
    ca = torch.tensor(calibration, device='cuda')
    # Same core minibatches, initialization, budget, validation and decoder in both conditions.
    mixweight = len(train)/(len(train)+int(aux_offsets[-1]))
    best = float('inf')
    best_state = None
    best_step = 0
    curve = []
    # >= one complete auxiliary pass; matched core-only control has same update count.
    steps = max(900, int(np.ceil(aux_offsets[-1]/1024))+100)
    for step in range(1, steps+1):
        net.train()
        opt.zero_grad(set_to_none=True)
        ix = torch.tensor(corecycle.get(512), device='cuda')
        v, m = core['yt'][ix], core['mt'][ix]
        z = net.encode(v, m, core['ct'])
        loss = observed_loss(net.logits(z, core['ct']), v, m)
        if condition == 'full_masked_pool':
            loss = mixweight*loss
            aux = auxcycle.get(1024)
            for k, panel in enumerate(panels[1:]):
                subset = aux[(aux >= aux_offsets[k]) & (aux < aux_offsets[k+1])]-aux_offsets[k]
                if not len(subset):
                    continue
                ids = torch.tensor(panel['usable'][subset], device='cuda')
                v, m = panel['yt'][ids], panel['mt'][ids]
                zz = net.encode(v, m, panel['ct'])
                loss = loss+(1-mixweight)*(len(subset)/1024)*observed_loss(net.logits(zz, panel['ct']), v, m)
        penalty = .0001*(net.ability[core['ct']].square().mean()+net.model_vectors[core['ct']].square().mean()+z.square().mean())
        loss = loss+penalty
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        opt.step()
        if step % 100 == 0 or step == steps:
            net.eval()
            with torch.no_grad():
                values = []
                for c in ca.split(1024):
                    v, m = core['yt'][c], core['mt'][c]
                    zz = net.encode(v, m, core['ct'])
                    values.append((len(c), float(observed_loss(net.logits(zz, core['ct']), v, m))))
                value = sum(n*v for n, v in values)/sum(n for n, v in values)
            curve.append(dict(step=step, calibration_brier=value))
            if step >= int(np.ceil(aux_offsets[-1]/1024)) and value < best:
                best, best_step = value, step
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
            save(folder/'progress.json', dict(step=step, steps=steps, calibration_brier=value))
    assert len(corecycle.seen) == len(train)
    if condition == 'full_masked_pool':
        assert len(auxcycle.seen) == aux_offsets[-1]
    net.load_state_dict(best_state)
    torch.save(best_state, folder/'model.pt')
    save(folder/'training.json', dict(core_questions=len(train), auxiliary_questions=int(aux_offsets[-1]) if condition=='full_masked_pool' else 0,
         distinct_core_seen=len(corecycle.seen), distinct_aux_seen=len(auxcycle.seen), steps=steps,
         checkpoint_step=best_step, calibration_brier=best, curve=curve))
    return net

def score_coordinates(q, theta, split, families, targets, seed, condition, records):
    # Historical-only Gaussian prior, same shrinkage/update as previous experiments.
    mu = theta.mean(0)
    empirical = np.cov(theta, rowvar=False)
    cov = .75*empirical+.25*np.diag(np.diag(empirical))+.01*np.eye(4)
    cov[0, 0] = max(cov[0, 0], .25)
    features = np.c_[np.ones(len(q)), q[:, 1:]]
    for family in np.unique(families[split['test']]):
        ids = split['test'][families[split['test']] == family]
        for acquisition_seed in range(5):
            order = np.random.default_rng(42+acquisition_seed+sum(map(ord, split['name']+str(family)))).permutation(ids)
            support, evaluation = order[:len(ids)//2][:31], order[len(ids)//2:]
            assert len(support) == 31 and not set(support)&set(evaluation)
            mean = np.tile(mu, (targets.shape[1], 1))
            covariance = np.broadcast_to(cov, (targets.shape[1], 4, 4)).copy()
            for budget in range(32):
                if budget in BUDGETS:
                    pred = expit(q[evaluation, 0, None]+features[evaluation]@mean.T)
                    errors = ((pred-targets[evaluation])**2).mean(0)
                    for target, error in enumerate(errors):
                        records.append(dict(fold=split['name'], protocol=split['protocol'], family=str(family),
                            training_seed=seed, acquisition_seed=acquisition_seed, target=target,
                            condition=condition, budget=budget, brier=float(error)))
                if budget < 31:
                    i = support[budget]
                    mean, covariance = update_batch(mean, covariance, features[i], q[i, 0], targets[i])

def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError('Frozen run exists; do not overwrite')
    OUT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this full-pool experiment')
    unit_check_batch()
    # Missing cells cannot influence the loss or its gradient.
    logits = torch.tensor([[.2, -.4]], requires_grad=True)
    mask = torch.tensor([[True, False]])
    a = observed_loss(logits, torch.tensor([[1., float('nan')]]), mask)
    b = observed_loss(logits, torch.tensor([[1., 999.]]), mask)
    assert torch.equal(a, b)
    a.backward()
    assert logits.grad[0, 1] == 0
    panels, ncols, held, table, mapping, oldy, families, splits, aux_offsets, metadata, kept = prepare()
    save(OUT/'plan.json', dict(question_vector_numbers=4, architecture='masked signed outcomes + observed indicators ->128 GELU ->64 GELU ->4; sigmoid linear solver readout',
         loss='observed-cell expected binary Brier, equal per-question weights; full pool weighted by question counts',
         conditions=['aligned_panel_only', 'full_masked_pool'], training_seeds=[0, 1], acquisition_seeds=5,
         heldout_original_models=held.tolist(), question_inputs='historical solver outcomes only, never question text',
         evaluation='same old six folds, test questions,24 targets and support/evaluation random seeds; source-calibration checkpoints only',
         removed_old_question_rows='Two previously quarantined contradictory-reference rows (original IDs261,409) absent from the cleaned panel; removed identically in both conditions. Remaining19,347 old rows retain order.',
         four_numbers='one question offset plus three interaction coordinates; four adaptive solver parameters including ability',
         code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         caveat='Shared masked encoder is tested; disconnected source IDs do not guarantee aligned latent coordinates. Exploratory previously examined evaluation folds. Frozen records can contain repeated empirical rates.'))
    save(OUT/'identity_audit.json', dict(allocated_historical_ids=ncols, excluded_models=held.tolist(),
         panels=[dict(panel=p['name'], input_rows=len(p['y']), usable_rows=len(p['usable']), columns=p['original_columns'].tolist(), observed_cells=int(p['mask'].sum())) for p in panels]))
    freeze = []
    for split in splits:
        test_groups = set(table.underlying_group.iloc[mapping[split['test']]])
        calibration_groups = set(table.underlying_group.iloc[mapping[split['calibrate']]])
        original_source = set(mapping[split['refit']])
        # New BBQ examples are allowed, but protect whole groups from the old folds.
        train = np.flatnonzero((table.index.isin(original_source) | table.source_panel.eq('sos_bbq_216')) &
                               ~table.underlying_group.isin(test_groups | calibration_groups))
        calibration = np.array([i for i in np.unique(mapping[split['calibrate']]) if table.underlying_group.iloc[i] not in test_groups])
        assert len(train) > 1000 and not set(train)&set(mapping[split['test']])
        assert not set(table.underlying_group.iloc[train]) & (test_groups | calibration_groups)
        save(OUT/f"{split['name']}_roles.json", dict(core_train=train.tolist(), core_calibration=calibration.tolist(), old_test=split['test'].tolist()))
        print('FOLD', split['name'], 'core', len(train), 'full', len(train)+int(aux_offsets[-1]), flush=True)
        for seed in [0, 1]:
            for condition in ['aligned_panel_only', 'full_masked_pool']:
                folder = OUT/f"{split['name']}__{condition}__seed{seed}"
                folder.mkdir()
                save(OUT/'status.json', dict(stage='training', fold=split['name'], seed=seed, condition=condition))
                net = fit(panels, ncols, train, calibration, aux_offsets, condition, seed, folder)
                q = encode_rows(net, panels[0], mapping)
                with torch.no_grad():
                    cols = panels[0]['ct']
                    theta = np.c_[net.ability[cols].cpu().numpy(), net.model_vectors[cols].cpu().numpy()]
                np.savez_compressed(folder/'frozen.npz', coordinates=q, theta=theta)
                freeze.append(dict(fold=split['name'], seed=seed, condition=condition, path=str(folder/'frozen.npz'),
                                   sha256=hashlib.sha256((folder/'frozen.npz').read_bytes()).hexdigest()))
                print('FROZEN', folder.name, flush=True)
                del net
                torch.cuda.empty_cache()
    save(OUT/'all_encodings_frozen.json', freeze)
    records = []
    for item in freeze:
        split = next(s for s in splits if s['name'] == item['fold'])
        with np.load(item['path']) as z:
            score_coordinates(z['coordinates'], z['theta'], split, families, oldy[:, held], item['seed'], item['condition'], records)
    df = pd.DataFrame(records)
    df.to_csv(OUT/'per_model.csv', index=False)
    summary = []
    for (protocol, condition), d in df.groupby(['protocol', 'condition']):
        curve = d.groupby('budget').brier.mean().reindex(BUDGETS)
        summary.append(dict(protocol=protocol, condition=condition, brier31=float(curve.iloc[-1]), alc=float(curve.to_numpy()@ALC), curve=curve.to_dict()))
    save(OUT/'result.json', dict(summary=summary))
    text = '# Four-dimensional historical-answer encoder: matched dataset expansion\n\nSame network, original evaluation targets/questions and 31-answer updates. Four question coordinates total. No text inputs or text encoder. All target solver IDs/metadata aliases excluded from training. Missing labels have zero loss; observed outcomes and masks form the input.\n\n| Protocol | Training pool | Brier 31 | Brier ALC |\n|---|---|---:|---:|\n'
    for r in summary:
        text += f"| {r['protocol']} | {r['condition']} | {r['brier31']:.6f} | {r['alc']:.6f} |\n"
    text += '\nOriginal evaluation folds were previously inspected: exploratory, not untouched confirmation. The added panels use source-specific solver identities. A common bottleneck does not establish cross-panel alignment. Training loss averages observed cells within each question; full-pool sampling weights questions uniformly. Every eligible training row was visited, though the source-selected checkpoint may precede the last training step. No target Brier selected checkpoints.\n'
    (REPORT/'report.md').write_text(text)
    save(OUT/'status.json', dict(stage='complete'))
    print(json.dumps(summary, indent=2), flush=True)

if __name__ == '__main__':
    main()

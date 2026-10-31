"""Explicit credential-free allowlist; exactly reproduced prior and fixed update."""
import sys,shutil,zipfile,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.demand_residual_adaptation_v48 import read,save,sha

def main():
    out=ROOT/'output/final_v82';folder=out/'submission';folder.mkdir(parents=True,exist_ok=True)
    artifact=Path('E:/AIMS/drift_exact_prior_v80/full_public');receipt=read(artifact/'fit_receipt.json')
    assert receipt['warnflag']==0 and len(receipt['benchmarks'])==167
    (folder/'model.py').write_text('''from functools import lru_cache
from pathlib import Path
from models.drift_pooling_runtime_v82 import Predictor

@lru_cache(maxsize=1)
def _get():return Predictor(Path(__file__).resolve().parent/'payload')

def predict(input,labeled=None):return float(_get().predict(input,labeled))
''',encoding='utf-8')
    files=['models/__init__.py','models/drift_pooling_runtime_v82.py','models/subject_adaptation_v79.py',
        'models/final_runtime_v74.py','models/manual_text_features_v1.py','models/calibration_priors_v73.py',
        'models/statistical_adaptation_v70.py','models/correlated_profile_adaptation_v67.py']
    for name in files:
        dst=folder/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dst)
    for source,name in [('model.py','drift_prior_model.py'),('featurize.py','featurize.py')]:
        shutil.copyfile(ROOT/'external/aims-team-drift/submission'/source,folder/name)
    (folder/'payload').mkdir(exist_ok=True)
    for name in ['meta.json','ws.f32','wi.f32','idf.f32']:
        assert sha(artifact/name)==receipt['hashes'][name]
        shutil.copyfile(artifact/name,folder/'payload'/name)
    save(folder/'payload/fit_receipt.json',receipt)
    (folder/'licenses').mkdir(exist_ok=True);shutil.copyfile(ROOT/'external/aims-team-drift/LICENSE',folder/'licenses/Team-Drift-MIT.txt')
    (folder/'requirements.txt').write_text('numpy\nscipy\n')
    (folder/'README.md').write_text('''# Cross-fitted probability pooling with a reproduced cross-benchmark prior

Idrees Aziz, idrees.mustafa.aziz@gmail.com.

The offline subject/text logistic prior is reproduced from Team Drift, Alex Alvord /
Cloud Lab Works LLC, MIT: https://github.com/cloudlabworks/aims-team-drift,
commit5a67dfcbaf8742f639edbfb96f15ee6efe3eed04. Its original runtime and
featurizer are included unchanged as drift_prior_model.py and featurize.py.
All167 listed public binary sources at measurement-db revision
54074453da7b9911be0f48e4c560be547bc84d21 enter the full deployment fit.
The10k item cap, .01 subject/text ridge and .5 benchmark weighting exponent
are unchanged. Disk-backed fitting changes floating-point reduction order;
these are retrained weights, not the original historical artifact. IDF hash
matches the original release. No raw training questions/outcomes are bundled.

Our contribution is the cross-fitted item-probability pooling and Brier stacking
layer over that prior: kappa.25, pseudo4 global calibration, solver ridge16,
stack ridge1. Gaussian adaptation extends the credited Drift baseline. Standard
Beta-binomial conjugacy and regularization are not claimed as inventions.
Only currently supplied same-benchmark labels are used; full subject/item/tool
identity, unequal counts and repeated trials are supported. No API credentials,
network calls, pretrained downloads, annotation service or GPU are required.
OpenAI Codex assisted development. NumPy/SciPy must be present in the scorer.

Validation uses benchmark-excluded prior fits, never these full-fit weights.
Broader public validation found neutral priors slightly better; the user chose
this metadata-aware prior for the final candidate. Local numbers are not hosted
scores. Dataset source licenses and terms remain applicable.
''',encoding='utf-8')
    target=out/'submission_drift_prior_ours_v82.zip'
    members=[p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in members:z.write(p,p.relative_to(folder).as_posix())
    dest=ROOT/'reports/finalization_v82';dest.mkdir(parents=True,exist_ok=True)
    save(dest/'package_manifest.json',dict(archive=str(target),sha256=sha(target),bytes=target.stat().st_size,credentials=False,network=False,
        files={p.relative_to(folder).as_posix():sha(p) for p in members}))
    print(target,sha(target),target.stat().st_size)
if __name__=='__main__':main()

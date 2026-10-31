# Predicting unfamiliar solvers with cross-fitted probability pooling

**Idrees Aziz** · idrees.mustafa.aziz@gmail.com

Code, fitted model and report for the AIMS / NeurIPS 2026 PAIEC competition.
The predictor starts from Team Drift's reproduced metadata/text prior, pools
currently revealed answers across solvers, and combines two adapted probability
estimates through support-only Brier stacking.

## Results

| Evaluation | Brier ALC | Brier after 31 answers |
|---|---:|---:|
| Codabench submission **1022433** | **0.190208** | **0.155706** |
| Local benchmark-excluded, ten-solver development | 0.151344 | 0.071451 |
| Privileged four-coordinate question positions | — | 0.046928 |

The rows use different evaluation protocols. The last row uses historical
answers to locate questions; predicting those positions from new question text
remains an open problem. Full protocols, paired comparisons and confidence
intervals are in the [report](report/aims_report.pdf).

## Run the submitted predictor

Python 3.11+ and NumPy/SciPy are sufficient. No API key, GPU or network is needed.

```bash
python -m pip install -r requirements.txt
python scripts/smoke_test.py
python -m pip install -r requirements-research.txt
python scripts/verify_results.py
```

`model.predict([subject, item], labeled)` returns a Python float. Each labeled
entry is `[[subject, item], outcome]`. See the smoke test for complete synthetic
inputs. The exact uploaded archive and its SHA-256 are in [submission/](submission/).
The root runtime files and payload match that archive; the repository README and
requirements provide additional release instructions.

## Reproduction

* [REPRODUCING.md](REPRODUCING.md): inference, prior training, result aggregation,
  confidence intervals and requirements for the original experiments.
* [results/](results/): public development scores, frozen splits, oracle records
  and hosted budget means. No hidden item-level outcomes are distributed.
* [research/](research/): original analysis and representation-training source.
* [artifacts/oracle/](artifacts/oracle/): fitted numerical representation arrays,
  source roles and solver-identity audit.

## Attribution

The initial logistic prior, feature hashing and Gaussian baseline come from
[Team Drift](https://github.com/cloudlabworks/aims-team-drift), Alex Alvord /
Cloud Lab Works LLC, commit `5a67dfcbaf8742f639edbfb96f15ee6efe3eed04`.
Their MIT notice is retained in `licenses/Team-Drift-MIT.txt`. Our code adds
cross-fitted item-probability pooling, solver calibration and Brier stacking.
Beta conjugacy, Gaussian shrinkage and stacking are established methods.
OpenAI Codex assisted development.

Code is MIT licensed. Dataset source terms and the report template's notices
remain applicable. Raw benchmark texts and response tables are obtained from
their original sources rather than redistributed here.

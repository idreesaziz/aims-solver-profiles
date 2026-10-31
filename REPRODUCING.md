# Reproduction guide

## Exact inference artifact

Install `requirements.txt`, then run `python scripts/smoke_test.py`.
Run `python scripts/check_submission.py` for the full 168-case isolated reference
comparison on the exact archive. The smoke test blocks network access and checks a finite probability, deterministic
dispatch, and benchmark isolation. `release_manifest.json` records file hashes.
`submission/` contains the exact credential-free Codabench archive. The fitted
prior payload is retrained under the released Drift objective; floating-point
reduction order differs from the historical upstream training run.

## Prior training from the public data

Install `requirements-research.txt`, then run `python scripts/train_prior.py`.
This downloads the exact 167-source manifest from
`aims-foundations/measurement-db` at revision
`54074453da7b9911be0f48e4c560be547bc84d21` using Hugging Face's ordinary access
mechanism. Each dataset retains its source license and access terms. No raw
tables are bundled. The source manifest is
`external/aims-team-drift/report/TRAINING_SOURCES.json`.

Defaults are `data/measurement-db` and `work/prior`. Override them with
`AIMS_DATA_DIR` and `AIMS_PRIOR_DIR`. Training is CPU-based and disk-backed;
downloading and fitting all sources takes substantial disk space and time.
The script fits both `cold_core` (core relatives excluded) and `full_public`
(deployment) priors. It preserves the original 10,000-item cap, selection RNG,
0.01 penalties and 0.5 source-weight exponent. Never evaluate `full_public`
as a benchmark-excluded core prior.

## Tables, paired effects and source-group confidence intervals

`python scripts/verify_results.py` recomputes core ALCs, paired benchmark effects,
the 10,000-draw source-group bootstrap (NumPy seed 860), and the oracle's 31-answer
scores from released episode-level records. It checks these against the frozen
summaries. This command reproduces numerical aggregation, without retraining.
The hosted curve contains only the organizer's published aggregate budget means.

## Original experiment sources and input requirements

`research/scripts/evaluate_drift_prior_v80.py` produced the matched local core
comparison. `research/scripts/broad_prior_audit_v81.py` and its independent verifier
define the broad source-group audit. `research/scripts/masked_answer_encoder_4d_v1.py`
trains the historical-answer encoder; `coordinate_brier_ablation_v42.py` evaluates
the coordinate subsets. Related imported local modules are included.

These archived experiment scripts preserve the Windows paths used during the
study. They require remapping their `D:/AIMS` and `E:/AIMS` directories and rebuilding
the aligned panels and caches before running. They are not a one-command research
pipeline. In addition to the pinned public tables, their prerequisites are:

* the 216-configuration aligned response panel with stable question/solver IDs;
* grouped train/calibration/development split indices and source identity maps;
* frozen 52/70-feature forest prediction caches for the corresponding folds;
* the masked auxiliary panels, per-source observed-label masks and solver maps;
* the four-coordinate targets and source/validation row mapping used by the
  coordinate ablation.

The original scripts document each input path and expected schema. Released
source roles, numerical oracle arrays and frozen scoring plans identify the
evaluated configuration. Some prerequisite data-preparation caches are not
bundled, so the current release reproduces inference, prior fitting, and reported
statistical aggregation directly; full reconstruction of every representation
training run requires those additional preparation steps.

The oracle encoder consumes observed historical outcomes and a separate mask;
it is not a question-text encoder. `artifacts/oracle/model_weights.npz` contains
the learned state as numerical arrays, avoiding pickle deserialization. The
Torch package is required only to rerun encoder training, not for inference.

## Report

The PDF and LaTeX source are in `report/`, including all seven vector figures.
Run `pdflatex -interaction=nonstopmode -halt-on-error aims_report.tex` twice from
that directory. This reproduces document typesetting using the frozen figures.

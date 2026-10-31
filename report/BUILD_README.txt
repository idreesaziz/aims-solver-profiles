Report source bundle (v89)

To typeset this report, extract all files into one directory and run:
pdflatex -interaction=nonstopmode -halt-on-error aims_report.tex
pdflatex -interaction=nonstopmode -halt-on-error aims_report.tex

Requires a LaTeX installation with the standard packages used in the .tex file.
The NeurIPS style and seven vector figures are bundled. Figure PDFs are final
assets; no access to the original data or Python workspace is needed to typeset.
Measurements and checks are recorded in the JSON manifests. Schematics are
explicitly labelled in captions. No credentials or raw test inputs are included.

This bundle reproduces document typesetting. The public repository provides the
fitted runtime, prior trainer, metric records and statistical aggregation code:
https://github.com/idreesaziz/aims-solver-profiles
Its reproduction guide documents additional input requirements for archived
representation-training experiments. OpenReview submission remains separate.

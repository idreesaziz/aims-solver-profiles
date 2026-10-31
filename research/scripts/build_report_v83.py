"""Two-result report: submitted pooling method and privileged4D diagnostic."""
import os
os.environ['MPLBACKEND']='Agg'
import json,sys,shutil
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/aims_report'

def diagram():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42})
    fig,ax=plt.subplots(figsize=(6.4,2.85));ax.set(xlim=(0,10),ylim=(0,4));ax.axis('off')
    def box(x,y,w,h,text):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.06,rounding_size=0.06',facecolor='white',edgecolor='black',lw=1,zorder=2))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=8,zorder=3)
    def arrow(a,b):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color='black',lw=1,shrinkA=3,shrinkB=4,zorder=1))
    box(.1,2.55,1.9,.85,'Solver metadata\n+ question text')
    box(2.65,2.55,2.3,.85,"Drift's reproduced prior\nInitial success probability")
    box(.1,.55,1.9,.85,'Revealed answers\nSame benchmark only')
    box(2.65,.55,2.3,.85,'Cross-fitted pooling\nItem counts + solver shift')
    box(5.6,2.55,2.05,.85,'Gaussian hierarchical\nupdate')
    box(5.6,.55,2.05,.85,'Beta probability\nupdate')
    box(8.25,1.55,1.6,.85,'Brier stacking\nFinal probability')
    arrow((2,2.98),(2.65,2.98));arrow((2, .98),(2.65,.98))
    arrow((3.8,2.55),(3.8,1.4));arrow((4.95,2.98),(5.6,2.98))
    arrow((4.95,1.15),(5.6,2.65))
    arrow((4.95,.98),(5.6,.98));arrow((7.65,2.98),(8.25,2.15));arrow((7.65,.98),(8.25,1.8))
    ax.text(5.1,.1,'The prior stays fixed. Only the revealed evidence changes the update.',ha='center',fontsize=8)
    fig.tight_layout(pad=.2);fig.savefig(OUT/'architecture.pdf',bbox_inches='tight');plt.close(fig)

def figures(core,oracle):
    diagram()
    fig,axes=plt.subplots(1,2,figsize=(6.4,2.25),layout='constrained')
    for method,label,style,marker in [('drift_prior_drift_update','Drift update','--','s'),('drift_prior_our52_update','Our pooling','-','o')]:
        v=core['results']['10'][method]
        axes[0].plot(range(6),v['curve'],style,color='black',marker=marker,ms=3,label=label,lw=1.2)
    axes[0].set(xticks=range(6),xticklabels=[0,1,3,7,15,31],xlabel='Revealed answers per solver',ylabel='Brier score',ylim=(0,.27),title='Same prior; different adaptation')
    axes[0].legend(frameon=False,fontsize=7);axes[0].grid(axis='y',color='.88',lw=.5)
    names=['1D','2D','3D','4D'];axes[1].bar(range(4),oracle,color=['white','.8','.5','black'],edgecolor='black',width=.55)
    axes[1].set(xticks=range(4),xticklabels=names,ylim=(0,.08),ylabel='Brier after 31 answers',title='Known question fingerprints (oracle)')
    for j,y in enumerate(oracle):axes[1].text(j,y+.002,f'{y:.3f}',ha='center',fontsize=8)
    axes[1].grid(axis='y',color='.88',lw=.5);axes[1].set_axisbelow(True)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.savefig(OUT/'results.pdf',bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.4,1.65));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,2))
    boxes=[(.1,2.5,'Historical outcomes\nfor this question'),(3.7,2.2,'Four coordinates\nKnown oracle position'),(7,2.7,'Solver-specific readout\nSuccess probability')]
    for x,w,text in boxes:
        ax.add_patch(FancyBboxPatch((x,.6),w,.8,boxstyle='round,pad=.05',fc='white',ec='black'))
        ax.text(x+w/2,1,text,ha='center',va='center',fontsize=8)
    for a,b in [(2.65,3.65),(5.95,6.95)]:ax.add_patch(FancyArrowPatch((a,1),(b,1),arrowstyle='-|>',mutation_scale=11,color='black'))
    ax.text(5,.1,'Open problem: infer the middle representation from a new question without its historical answers.',ha='center',fontsize=8)
    fig.tight_layout(pad=.2);fig.savefig(OUT/'oracle.pdf',bbox_inches='tight');plt.close(fig)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(ROOT/'output/pdf/aims_first_draft/neurips_2026.sty',OUT/'neurips_2026.sty')
    core=json.loads((ROOT/'reports/drift_exact_prior_v80/comparison.json').read_text())
    raw=json.loads(Path('D:/AIMS/original_single_window_v41/analysis/coordinate_brier_ablation_v42/result.json').read_text())
    rows=[r for r in raw['summaries'] if r['cohort']=='larger_true']
    if not rows:raise ValueError('Missing larger oracle cohort')
    names=['rotated_difficulty_plus_0','rotated_difficulty_plus_1','rotated_difficulty_plus_2','raw_1234']
    lookup={r['method']:r for r in rows};oracle=[lookup[m]['brier31'] for m in names]
    figures(core,oracle)
    state=json.loads((ROOT/'reports/finalization_v82/submission_state.json').read_text())
    if state.get('score') is None:
        hosted=r'Submission \texttt{1022433} has been received by Codabench. Its hosted score is pending; no local score below is presented as its official result.'
    else:
        hosted='Submission \\texttt{1022433} achieves hosted Brier ALC \\textbf{'+f"{state['score']:.6f}"+'}. Across its eight solver--benchmark pairs, mean Brier is '+f"{state['official_budget_means']['0']:.5f}"+' before observations and '+f"{state['brier31']:.5f}"+' after 31 observations. This formative run uses a separately sampled hidden subset; it is not a paired comparison with local validation.'
    table='\n'.join(f"{c} & {core['results'][str(c)]['drift_prior_drift_update']['alc']:.4f} & {core['results'][str(c)]['drift_prior_our52_update']['alc']:.4f} & {core['results'][str(c)]['drift_prior_our52_update']['brier31']:.4f}\\\\" for c in [1,3,10])
    oracle_table='\n'.join(f"{j+1} & {v:.5f}\\\\" for j,v in enumerate(oracle))
    tex=r'''\documentclass{article}
\usepackage[preprint,nonatbib]{neurips_2026}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx,booktabs,amsmath,amssymb}
\usepackage[hidelinks]{hyperref}
\renewcommand{\ttdefault}{pcr}
\hypersetup{pdfauthor={Idrees Aziz},pdftitle={Predicting Unfamiliar Solvers with Cross-Fitted Probability Pooling}}
\title{Predicting Unfamiliar Solvers\\with Cross-Fitted Probability Pooling}
\author{Idrees Aziz\\\href{mailto:idrees.mustafa.aziz@gmail.com}{\texttt{idrees.mustafa.aziz@gmail.com}}}
\begin{document}\maketitle
\begin{abstract}
An unfamiliar AI system need not be evaluated exhaustively to make useful predictions about its behaviour. We start with a model of past evaluation results and revise its predictions using a small set of revealed answers. Our method combines a reproduced cross-benchmark prior with an adaptation rule that pools evidence about questions across solvers, corrects for the new solver's performance, and controls how strongly that evidence overrides the prior. A cross-fitted combination is chosen directly for squared probability error. In a matched local benchmark-cold evaluation with ten solvers supplying evidence, the method reaches a Brier area under the learning curve of 0.151. A separate oracle experiment finds that four coordinates derived from historical answers preserve substantial question-level predictive information. Inferring these coordinates for a genuinely new question remains an open problem.
\end{abstract}

\section{The prediction problem}
An average benchmark score records how often a system succeeds. It says less about where it succeeds. Two systems may have the same accuracy while solving different questions; two equally difficult questions may separate different groups of systems. Predicting an individual response requires some account of both distinctions.

The Predictive AI Evaluation Competition~\cite{aims} asks a submitted program to estimate the probability that a specified solver answers a specified question correctly. A solver is a model together with its evaluation configuration. Inputs include model metadata, question text, additional question attributes and tool metadata. At test time, the program may also receive a small budget of revealed responses. The benchmark itself is unfamiliar: its identifier links evidence within the current evaluation, rather than identifying a known training benchmark.

We report two results. The first is an executable prediction method: a cross-benchmark prior followed by cross-fitted probability pooling. The second concerns the representation of questions. A question's historical solver outcomes can be compressed into four coordinates without losing much of their predictive value in a local oracle diagnostic. The executable method does not require those coordinates.

\section{A prior updated by shared evidence}
\begin{figure}[ht]\centering
\includegraphics[width=\linewidth]{architecture.pdf}
\caption{Submitted architecture. A fixed prior supplies a probability for each solver--question pair. Revealed answers feed both an item-probability pooling rule and a Gaussian hierarchical update. Cross-fitted Brier stacking combines their predictions. All evidence comes from the current benchmark; the diagram is schematic.}
\label{fig:architecture}\end{figure}

\subsection{The cross-benchmark starting point}
We reproduce Team Drift's released logistic prior~\cite{drift}, retaining its feature functions and training settings. For solver $s$ and item $i$,
\begin{equation}
p^0_{si}=\sigma\!\left(b+\mathbf w_s^\top\phi(s)+\mathbf w_i^\top\psi(i)\right),\qquad \sigma(z)=\frac{1}{1+e^{-z}}.
\end{equation}
The subject features hash the model name, provider, harness, reasoning effort, dates and configuration fields at several granularities. Item features hash words, word pairs, length bins, supplied attributes and tool metadata. Text features receive source-fitted inverse-document-frequency weights and row normalization. This is an additive probability model, not an invocation of the solver being evaluated.

The deployment prior is fitted to 167 public binary benchmark sources at the pinned AI Measurement Data Bank revision~\cite{db}. It retains at most 10,000 observed items per source, using the released deterministic selection. Repeated trials become success and attempt counts for each solver--item pair. If source $b$ has $N_b$ attempts, each likelihood term has weight $N_b^{-1/2}$. Subject and item ridge coefficients are both 0.01. Our disk-backed fitter evaluates the same objective without loading the complete feature matrix into RAM. The resulting IDF array matches the historical release hash; fitted weights are retrained artifacts, not claimed to be byte-identical historical weights.

\subsection{Let the evidence override a mistaken prior}
The prior can be wrong about an unfamiliar benchmark. The adaptation rule should use new evidence strongly enough to correct that error, without treating one solver's observed answer as independent evidence for its own calibration.

For each item $i$ currently observed, let $n_i$ be the number of revealed trials and $k_i$ their successes, pooled over permitted solvers. Define the shared item prior $\bar p_i$ as the mean of $p^0_{si}$ over the current solver cohort. A regularized log-odds shift $g$ calibrates these shared priors using one fractional outcome $k_i/n_i$ per distinct observed item. This prevents a frequently observed item from dominating the benchmark shift solely through its multiplicity.

The shift minimizes logistic cross-entropy plus four soft prior observations centred on zero shift. For offsets $z_i=\operatorname{logit}\bar p_i$, $\mu=\operatorname{mean}_i\sigma(z_i)$ and $a=\operatorname{logit}\mu$, its objective is
\begin{equation}
L(g)=\sum_i\left[\operatorname{softplus}(z_i+g)-\frac{k_i}{n_i}(z_i+g)\right]
+4\left[\operatorname{softplus}(a+g)-\mu(a+g)\right].
\end{equation}
For an observed item, its calibration shift $g_{-i}$ is fitted using the other items only; a previously unobserved query uses the all-support shift. We then use the target solver's own prior,
\begin{equation}
m_{si}=\sigma\!\left(\operatorname{logit}p^0_{si}+g_{-i}\right),\qquad
\widetilde p_{si}=\frac{\kappa m_{si}+k_i}{\kappa+n_i},\qquad\kappa=0.25.
\label{eq:pool}
\end{equation}
Equation~\ref{eq:pool} has the form of a Beta-binomial predictive mean~\cite{beta}. Here it is a regularized pooling rule: heterogeneous solver outcomes are not assumed to be exchangeable draws from one exact joint Bayesian model. A small $\kappa$ lets observed outcomes override the starting probability.

Each solver also receives a regularized log-odds correction, fitted with ridge precision 16. For a support answer $y_{si}$, its calibration base removes that answer from the counts:
\begin{equation}
\widetilde p^{\,\mathrm{support}}_{si}
=\frac{\kappa m_{si}+k_i-y_{si}}{\kappa+n_i-1}.
\end{equation}
The benchmark shift for that base excludes the whole item. Removing both paths matters: otherwise the label used to judge the solver has already moved its supposed prior in the same direction. Query predictions use all evidence permitted at the current budget. Repeated trials count as trials, not as extra question identities.

\subsection{Choose the combination for Brier error}
The pooling prediction is combined with a correlated Gaussian hierarchical correction of the prior. The latter extends Drift's benchmark, solver and item log-odds update: benchmark and item standard deviations are 1.5, solver standard deviation is 0.8, and the fixed solver correlation matrix is $0.1I+0.9\mathbf1\mathbf1^\top$. This covariance is shared shrinkage, not a learned identification of model families.

Three folds of the currently revealed answers generate held-out predictions from both updates. Let $a_t$ be the Gaussian prediction, $b_t$ the pooling prediction and $d_t=b_t-a_t$. The weight on pooling is
\begin{equation}
w=\operatorname{clip}\!\left(\frac{\sum_t d_t(y_t-a_t)+1}{\sum_t d_t^2+1},0,1\right),\qquad
\widehat p=(1-w)a+wb.
\end{equation}
This minimizes held-out Brier error with penalty $(w-1)^2$; the penalty favours pooling when little support evidence distinguishes the updates. Each fold refits both methods without its held labels. No scored query outcome selects the weight. The procedure is related to predictive stacking~\cite{stacking}, but uses squared probability error directly.

The complete submission is CPU-only and API-free. It uses the supplied evidence afresh at every budget, filters it by benchmark, and distinguishes configurations by their full visible attributes. Memory caches accelerate repeated calls but carry no assumed evidence across checkpoints.

\section{Result I: predictive performance}
\subsection{Matched benchmark-cold comparison}
The core local evaluation holds out BBH, GPQA, MATH and MMLU-Pro in turn. A source-trained prior excludes the target benchmark and known multi-benchmark relatives. An aligned response panel supplies 192 historical configurations and 24 target configurations; the four core development cohorts contain 934, 88, 211 and 1,940 questions respectively. The target configurations are excluded from the forest controls' training outputs. The broad logistic prior can encounter their metadata on other permitted sources; this is not a strict unseen-name claim.

For each benchmark and one, three or ten sampled target solvers, we use a pool capped at approximately 1,000 solver--item pairs. Acquisition and evaluation groups are disjoint for each solver. Three fixed randomization seeds generate matched episodes, with nested budgets $n\in\{0,1,3,7,15,31\}$. Solvers may reveal answers to a question that is scored for another solver. This permitted sharing is more common in larger cohorts.

We measure squared probability error, $B_n=\operatorname{mean}(\widehat p-y)^2$. Brier ALC combines the budgets as
\begin{equation}
\mathrm{ALC}=0.1B_0+0.2B_1+0.2B_3+0.2B_7+0.2B_{15}+0.1B_{31}.
\end{equation}
Lower is better. Both adaptation methods in Table~\ref{tab:core} use the same reproduced benchmark-excluded prior and identical evidence. Only their update differs.

\begin{table}[ht]\centering
\caption{Matched local core-benchmark results. The submitted pooling settings are used throughout. Brier31 is an endpoint; ALC averages across budgets. These are development results, not official scores.}
\label{tab:core}
\begin{tabular}{rrrr}\toprule
Solvers & Drift update ALC & Our pooling ALC & Our pooling Brier31\\\midrule
CORETABLE
\bottomrule\end{tabular}\end{table}

With ten solvers, pooling lowers ALC from 0.1642 to 0.1513 and the 31-answer error from 0.0903 to 0.0715. The gain is smaller with one solver, where there are no other solvers' answers to pool. The ten-solver result therefore describes a favourable evidence-sharing regime, not a universal expected score.

\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{results.pdf}
\caption{Two measured results under different protocols. Left: matched ten-solver core evaluation, using the same prior with two adaptation rules. Right: a separate privileged-position diagnostic on mixed-benchmark development questions. The oracle bars use historical outcomes of the evaluated questions and are not deployable text-only predictions.}
\end{figure}

\subsection{Broader benchmark transfer and hosted evaluation}
A separate source-group-held-out evaluation tests the same submitted prior-and-update recipe across 43 additional benchmark sources, spanning 41 related-source groups. Subject metadata and tool attributes come directly from the released tables. Groups with common source repositories or benchmark families remain together. Text-identical questions cannot appear in both a solver's acquisition and evaluation pools; configurations with identical visible attributes cannot supply separate solver budgets.

This evaluation contains 311 solver--benchmark episodes across the one- and up-to-three-solver settings, with two fixed episode seeds. All recorded response counts contribute to each pair's Brier score, and pairs receive equal weight within each setting. Inputs above 100,000 characters and episodes with fewer than 80 distinct question texts are omitted. The validation groups are separate from those used to select among starting-prior candidates. The submitted method's regularization is fixed.

On this broader validation, the submitted recipe obtains ALC 0.2170 with one solver and 0.2125 with up to three. The corresponding mean errors over budgets 0, 1 and 3 are 0.2384 and 0.2300. A neutral 50\% starting prior with the same update is slightly better, at ALC 0.2153 and 0.2118. The metadata-aware prior is retained in the submitted candidate; the broader evidence does not establish that it is better than a neutral start. These results constrain the interpretation of the stronger core-benchmark numbers.

HOSTED

\clearpage
\section{Result II: a four-coordinate question oracle}
Read a historical response matrix across a row and it describes a solver's behaviour. Read down a column and it describes which solvers answer a question correctly. We call this column the question's \emph{fingerprint}. Equal mean difficulty does not imply an equal fingerprint: two questions can be solved by different halves of the same solver population.

\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{oracle.pdf}
\caption{The representation diagnostic. A question's observed successes and failures across historical solvers determine its coordinates. A new solver's revealed answers adapt the readout. For unseen questions, the left-hand information is unavailable: recovering the coordinates without it is the open problem.}
\end{figure}

The oracle diagnostic encodes observed fingerprints and their missingness masks through a feedforward network with hidden widths 128 and 64, GELU activations and a four-number output. Training minimizes observed-cell expected binary Brier, averaging within questions so that a densely observed row does not automatically receive more weight. The representation fit sees 70,952 core and 361,141 auxiliary question rows, preserving solver identities and masking unavailable responses. Its readout is
\begin{equation}
P(Y_{si}=1)=\sigma\!\left(q_{i0}+a_s+\sum_{d=1}^{3}q_{id}v_{sd}\right).
\end{equation}
The first question coordinate acts as an offset, while the other three interact with solver traits. For an excluded target solver, a source-fitted Gaussian prior over its intercept and traits is adapted through sequential Laplace updates using 31 mixed-benchmark revealed responses. The coordinates of the evaluated questions themselves come from historical answers, so this experiment measures the value of knowing a compact behavioural representation; it does not measure the ability to infer that representation from question text.

The evaluation uses 3,241 development questions from BBH, GPQA, MATH, MMLU-Pro and MuSR, with 192 historical and 24 excluded target configurations. Source calibration of the readout prior and coordinate rotations uses 12,144 questions. Ten fixed support trajectories reveal six questions per benchmark plus one rotating extra question; support groups are excluded from evaluation. Source-fitted whitening and rotations permit lower-dimensional ablations. Each readout is adapted and scored under this mixed-benchmark protocol, which differs from the benchmark-cold comparison in Table~\ref{tab:core}.

\begin{table}[ht]\centering
\caption{Known question-position diagnostic. Lower-dimensional rows retain source-constructed directions. Four coordinates reach Brier31 0.04693. The score uses privileged historical outcomes and is not a competition submission result.}
\begin{tabular}{rr}\toprule
Question coordinates & Brier31\\\midrule
ORACLETABLE
\bottomrule\end{tabular}\end{table}

The compact representation suggests a concrete research target. A useful question encoder would place a new question near others with similar solver-outcome patterns, rather than merely similar wording or topic. If those positions could be recovered reliably without historical answers, a small number of revealed responses could identify a solver's location relative to the same behavioural structure. The present result establishes an informative oracle representation in this development cohort. It does not establish an accurate text-to-coordinate encoder or a transferable semantic interpretation of each axis.

\section{Scope, attribution and reproducibility}
The submitted contribution is the cross-fitted pooling and Brier-stacking layer over a reproduced public prior. Team Drift's original feature functions, prior model and hierarchical adaptation are credited and their MIT license is retained~\cite{drift}. Beta-binomial conjugacy, Gaussian shrinkage and stacking are established statistical tools; we claim neither their invention nor an exact Bayesian coverage guarantee for the combined procedure. The four-coordinate result is a separate representation diagnostic.

All training outcomes come from permitted public Measurement Data Bank sources at revision \texttt{54074453da7b9911be0f48e4c560be547bc84d21}. The deployment prior uses 624,900 selected source-item records and 12,839,906 aggregated solver--item rows. No hidden evaluation outcomes enter training. Local benchmark-cold comparisons use separately fitted priors, never the full deployment prior on a benchmark it has seen. Source-group exclusion reduces related-source leakage; global duplicate detection across unrelated repositories is not certified. The development data and related solver configurations have been inspected during research, so the reported local improvements require independent confirmation.

The exact submitted ZIP contains the trained prior, NumPy/SciPy adaptation and the official \texttt{predict(input,labeled)} entry point. It needs no API key, network inference, model download or GPU. The organizer's ZIP checker passes. An isolated exact-archive test verifies 168 dispatch predictions against the reference adaptation to within $6\times10^{-17}$, including unequal support sizes, repeated trials, tool-distinct items, unseen target subjects and benchmark isolation. The disk-backed trainer reproduces the unchanged upstream trainer on a fixture; real-data preprocessing agrees exactly on three checked sources. OpenAI Codex assisted implementation, analysis and writing.

The executable method and the oracle answer different questions. The former predicts with the evidence available at deployment. The latter shows what a compact description of a question's observed behaviour can retain. Recovering that description for a new question remains the step needed to connect them.

\begin{thebibliography}{9}
\bibitem{aims} AIMS. The Predictive AI Evaluation Competition at NeurIPS 2026. Rules, interface and evaluation protocol. \url{https://aims.stanford.edu/competition/}.
\bibitem{drift} Alex Alvord / Cloud Lab Works LLC. Team Drift: AIMS / NeurIPS 2026 PAIEC. Released implementation and training recipe, commit \texttt{5a67dfcbaf8742f639edbfb96f15ee6efe3eed04}. \url{https://github.com/cloudlabworks/aims-team-drift}.
\bibitem{db} AIMS Foundations. The AI Measurement Data Bank. Public tables and benchmark-level provenance. \url{https://huggingface.co/datasets/aims-foundations/measurement-db}.
\bibitem{beta} Bob Carpenter. Getting Started with Bayesian Statistics. Bernoulli inference and Beta conjugacy. \url{https://bob-carpenter.github.io/stan-getting-started/stan-getting-started.html}.
\bibitem{stacking} Yuling Yao, Aki Vehtari, Daniel Simpson, and Andrew Gelman. Using stacking to average Bayesian predictive distributions (with discussion). \emph{Bayesian Analysis}, 13(3):917--1007, 2018. \url{https://doi.org/10.1214/17-BA1091}.
\end{thebibliography}

\appendix
\section{Reproduction map}
The source manifest records all 167 benchmark names, the pinned dataset revision and fitted-artifact hashes. \path{scripts/reproduce_drift_prior_v80.py} reconstructs the prior from the public tables with disk-backed sparse features. \path{scripts/evaluate_drift_prior_v80.py} produces Table~\ref{tab:core}; \path{scripts/broad_prior_audit_v81.py} defines the source-group splits and broader comparison, and \path{scripts/verify_broad_prior_audit_v81.py} independently checks selection, aggregation, split boundaries and paired source-group bootstrap calculations. The oracle table is read directly from the coordinate-ablation result generated by \path{scripts/coordinate_brier_ablation_v42.py}.

\path{models/drift_pooling_runtime_v82.py} implements the submitted dispatch and \path{models/subject_adaptation_v79.py} implements the subject-specific pooling layer. \path{scripts/package_drift_pooling_v82.py} builds the explicit-allowlist scoring ZIP, and \path{scripts/check_drift_pooling_zip_v82.py} tests that exact archive in a fresh process with network access blocked. The archive SHA-256 is
\begin{quote}\small\ttfamily
b0aef886620c7ba3545ab7b132336b969079\\
ee5160d758d67125703b51a9ffd1.
\end{quote}
Local probability pooling clips its numerical bases away from zero and one. The zero-answer prediction is the starting prior, clipped to $[0.001,0.999]$. Acquired labels are filtered to the same anonymous benchmark identifier; if that identifier is missing, only unscoped evidence from the identical subject is retained. Item identity includes content, supplied features and tool metadata. Current support evidence is refitted when its signature changes; caches do not add labels.

Only the credential-free scoring ZIP and explicit reproduction files are intended for release. Raw Measurement Data Bank tables remain subject to their access terms and individual source licenses. The report and code must be accompanied by an accessible release location when completing the competition entry.
\end{document}
'''
    tex=tex.replace('CORETABLE',table).replace('ORACLETABLE',oracle_table).replace('HOSTED',hosted)
    (OUT/'aims_report.tex').write_text(tex,encoding='utf-8')
    (OUT/'report_provenance.json').write_text(json.dumps(dict(submission=1022433,hosted_score=state.get('score'),main_method='Reproduced Drift prior + kappa.25pseudo4 pooling + Brierstack1',local=core['results'],oracle=dict(zip(names,oracle)),oracle_privileged=True,author='Idrees Aziz',affiliation=None),indent=2))
    print(OUT/'aims_report.tex')
if __name__=='__main__':main()

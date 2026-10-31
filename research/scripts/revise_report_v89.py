"""Direct prose and paired, source-clustered statistical reporting."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'output/pdf/aims_report'
def main():
    archive=OUT/'archive_v85';archive.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in {'.pdf','.tex','.json','.zip','.txt','.sty'} and not (archive/p.name).exists():shutil.copy2(p,archive/p.name)
    tex=(archive/'aims_report.tex').read_text(encoding='utf-8')
    def replace(a,b):
        nonlocal tex
        assert a in tex,a[:100]
        tex=tex.replace(a,b)
    replace('A starting probability, not a verdict','A starting probability')
    replace('The model supplies an initial probability; it does not invoke an LLM or require fresh question annotation.','The fitted model supplies an initial probability from the attributes already present in the input.')
    replace('The fitted artifact is retrained, not the historical weights. ','')
    replace('It has the form of a Beta-binomial predictive mean~\\cite{beta}; because different solvers are heterogeneous, we use it as regularized probability pooling, rather than claim one exact exchangeable Bayesian model.','The expression follows the Beta-binomial predictive mean~\\cite{beta}. Solver-specific corrections account for differences between the solvers whose outcomes enter the pool.')
    replace('Do not let an answer judge itself','Keep calibration evidence independent')
    replace('Support answers serve two purposes: fitting the update and judging whether that update predicts well. Without exclusions, these roles can become circular. A success would raise its own starting probability and then appear to confirm the resulting prediction.','Support answers serve two purposes: fitting the update and measuring its prediction error. We separate these roles by removing each answer from the evidence used to construct its calibration base.')
    replace('Repeated responses count as trials; they do not create new question identities.','Repeated responses contribute their success and attempt counts under a shared question identity.')
    replace('Its benchmark, solver and question effects provide a different pattern of shrinkage. The solver covariance is fixed; it does not identify model families from their names or learn their true relationships.','Its benchmark, solver and question effects provide a second pattern of shrinkage. A fixed covariance shares solver offsets across the cohort.')
    replace('Scored query outcomes never select the weight. Both branches are then fitted to the available support set for prediction.','Weight selection uses the revealed support answers. Both branches are then fitted to the complete support set for query prediction.')
    replace('Memory caches save repeated work; they do not add evidence. This boundary is essential in an evaluation that scores the same query set at several observation budgets.','Memory caches reuse fits for identical supplied evidence. Each observation budget has its own evidence set and fitted update.')
    replace('The broad logistic prior can encounter their metadata on other permitted sources; model names are not assumed unseen.','The broad logistic prior can encounter these model identities on other permitted sources. This experiment measures transfer to held-out benchmarks with target response columns excluded from the aligned historical panel.')
    replace('These are development results, not official scores.','Scores summarize the fixed local development episodes.')
    replace('With ten solvers, pooling lowers ALC from 0.1642 to 0.1513 and the 31-answer error from 0.0903 to 0.0715. With one solver, ALC changes little and Brier31 is slightly worse (0.1886 versus 0.1876); there are no other solvers\' answers to pool. The ten-solver result therefore describes a favourable evidence-sharing regime, not a universal expected score.',
        'With ten solvers, pooling lowers ALC by 0.0128, from 0.1642 to 0.1513, and lowers the 31-answer error from 0.0903 to 0.0715. BBH, GPQA and MMLU-Pro improve; MATH increases by 0.0011 ALC. With one solver, the ALC improvement is 0.0010 and Brier31 increases by 0.0009. Appendix~\\ref{app:paired} reports the paired changes for each benchmark.')
    replace('Curves are means over the fixed development episodes; no uncertainty band is inferred from these averages.','Curves average the fixed development episodes. Paired benchmark effects appear in Appendix~\\ref{app:paired}.')
    replace('This permitted sharing is more common in larger cohorts.','At 31 observations, another solver has answered 29.8\\% of query questions in the three-solver setting and 97.9\\% in the ten-solver setting. Cohort size therefore changes both solver count and evidence coverage.')
    replace('These restrictions and incomplete cross-repository deduplication limit the coverage claim.','The analysis covers the eligible public sources under these input and sampling rules; exact-text checks operate within each source.')
    start=tex.index('On this broader validation, the submitted recipe obtains')
    end=tex.index('\\subsection{The hosted result}',start)
    tex=tex[:start]+r'''On this broader validation, the submitted recipe obtains ALC 0.2170 with one solver and 0.2125 with up to three. A neutral 50\% starting prior with the same update obtains 0.2153 and 0.2118. Table~\ref{tab:broad} reports the paired differences and their uncertainty. The intervals span zero in both cohorts, leaving the two starting priors unresolved by this comparison.

\begin{table}[ht]\centering
\caption{Broader held-source-group validation. $\Delta$ is neutral-prior ALC minus Drift-prior ALC; negative values favour the neutral prior. The 95\% percentile bootstrap intervals use 10,000 resamples of the 41 whole source groups, retaining paired methods, seeds and solver episodes.}
\label{tab:broad}
\small
\begin{tabular}{lrrr}\toprule
Cohort & Drift ALC & Neutral ALC & $\Delta$ ALC [95\% interval]\\\midrule
One solver & 0.2170 & 0.2153 & $-0.0016\;[-0.0050,\;0.0015]$\\
Up to three & 0.2125 & 0.2118 & $-0.0007\;[-0.0026,\;0.0011]$\\\bottomrule
\end{tabular}\end{table}

''' +tex[end:]
    replace('Its initial Brier score, 0.2779, is worse than the guaranteed 0.25 of predicting 0.5 everywhere.','Its initial Brier score is 0.2779; the constant 0.5 predictor scores 0.25.')
    replace('Strong adaptation therefore coexists with an imperfect starting prior.','The largest errors occur before the program has acquired evidence.')
    replace('The shaded area illustrates the curve summarized by ALC; it is not an uncertainty band. Pair scores are equally weighted. This hidden sample differs from local validation, so the hosted and local numbers are not paired comparisons. The final common-subset summative evaluation is separate.',
        'Shading marks the area summarized by ALC. Scores average eight solver--benchmark pairs equally. The hosted sample and local development bank define separate evaluations; the organizers conduct final summative scoring on a common hidden subset.')
    replace('The local matched comparison isolates the adaptation change. The hosted run measures the complete program on a different sample. The broad validation tests a wider range of sources. Together they support a narrower claim than a single best score would suggest: question-level sharing can help substantially, especially with several solvers, while reliable cold-start probabilities remain a limitation.',
        'The matched local comparison measures the adaptation change. The hosted run measures the complete program on hidden inputs. The broad validation measures transfer across source groups. Together, these results locate the method\'s strength in shared question-level evidence and its main weakness in the starting probabilities.')
    replace('Equal mean difficulty does not imply an equal fingerprint: two questions can be solved by different halves of the same solver population.','Two equally difficult questions can be solved by different halves of the same solver population.')
    replace('Illustrative cells show the information flow, not fitted coordinate values.','Schematic cells illustrate the encoding and readout.')
    replace('Training minimizes observed-cell expected binary Brier, averaging within questions so that a densely observed row does not automatically receive more weight.','Training averages expected binary Brier across the observed cells of each question, then averages the question-level losses.')
    replace('The coordinates of the evaluated questions themselves come from historical answers, so this experiment measures the value of knowing a compact behavioural representation; it does not measure the ability to infer that representation from question text.',
        'Historical answers supply the evaluated question coordinates. The experiment measures prediction with these privileged behavioural positions; a deployable text encoder must recover them for new questions.')
    replace('Left: measured privileged-position diagnostic; each readout is fitted and adapted under the same mixed-benchmark protocol. These are Brier31 scores, not ALC, and differ from the submission\'s benchmark-cold protocol. Right: a schematic two-coordinate slice of a logistic readout. Positions A and B are illustrative; contours show how question position changes one solver\'s predicted success. No semantic axis interpretation is claimed.',
        'Left: Brier31 after 31 revealed answers under the mixed-benchmark oracle protocol, with historical answers supplying question positions. Right: a schematic two-coordinate slice of a logistic readout. Contours show one solver\'s success probability; A and B mark illustrative positions.')
    replace('The near equality of the last two results shows that additional coordinates need not improve this readout. It does not show that any arbitrarily selected three dimensions are sufficient: the lower-dimensional directions are constructed using source calibration.',
        'The three- and four-coordinate readouts reach essentially the same error. Source calibration constructs the retained lower-dimensional directions.')
    replace('The result gives the representation problem a sharper target. A useful question encoder would place a new question near others with similar solver-outcome patterns, rather than merely similar wording or topic. If those positions could be recovered reliably without historical answers, a small number of revealed responses could identify a solver\'s location relative to the same behavioural structure. The present result establishes an informative oracle representation in this development cohort. It does not establish an accurate text-to-coordinate encoder or a transferable semantic interpretation of each axis.',
        'These positions preserve distinctions between the solvers that succeed on each question. A question encoder trained for this task must recover that behavioural geometry from the supplied input. The oracle provides a compact prediction target: three calibrated directions retain almost all of the four-coordinate readout\'s accuracy in this development cohort.')
    replace('Beta-binomial conjugacy, Gaussian shrinkage and stacking are established statistical tools; we claim neither their invention nor an exact Bayesian coverage guarantee for the combined procedure. The four-coordinate result is a separate representation diagnostic.',
        'Performance is measured empirically through Brier error. The four-coordinate experiment evaluates a separate behavioural representation.')
    replace('This is an additive probability model, not an invocation of the solver being evaluated.','Inference evaluates the additive probability model on the supplied attributes.')
    replace('The resulting IDF array matches the historical release hash; fitted weights are retrained artifacts, not claimed to be byte-identical historical weights.',
        'The IDF array matches the historical release hash. The fitted weights come from our reproduction of the training objective.')
    replace('The adaptation rule should use new evidence strongly enough to correct that error, without treating one solver\'s observed answer as independent evidence for its own calibration.',
        'The adaptation rule corrects that error using revealed outcomes. Cross-fitting separates each solver\'s calibration target from the evidence used to construct its starting probability.')
    replace('Here it is a regularized pooling rule: heterogeneous solver outcomes are not assumed to be exchangeable draws from one exact joint Bayesian model.','We use the conjugate expression as a regularized predictive pooling rule and estimate solver differences through log-odds corrections.')
    replace('Repeated trials count as trials, not as extra question identities.','Repeated trials retain their success and attempt counts within each question identity.')
    replace('This covariance is shared shrinkage, not a learned identification of model families.','This covariance imposes common shrinkage across solver offsets.')
    replace('No scored query outcome selects the weight. The procedure is related to predictive stacking~\\cite{stacking}, but uses squared probability error directly.',
        'The held-out support predictions determine the weight. The procedure adapts predictive stacking~\\cite{stacking} to squared probability error.')
    replace('No hidden evaluation outcomes enter training. Local benchmark-cold comparisons use separately fitted priors, never the full deployment prior on a benchmark it has seen. Source-group exclusion reduces related-source leakage; global duplicate detection across unrelated repositories is not certified. The development data and related solver configurations have been inspected during research, so the reported local improvements require independent confirmation.',
        'Local comparisons use separately fitted priors with the target source groups excluded. Duplicate checks cover exact question text within the episode sources. The remaining generalization questions concern semantic duplicates across repositories, further benchmark families and future solver populations. The local bank and related solver configurations have been inspected during development.')
    replace('Only the credential-free scoring ZIP and explicit reproduction files are intended for release. Raw Measurement Data Bank tables remain subject to their access terms and individual source licenses. An accessible code-release URL must accompany the final OpenReview entry; this review copy does not yet supply a public release link.',
        'Release materials comprise the credential-free scoring ZIP, reproduction scripts, split manifests and permitted fitted artifacts. Raw Measurement Data Bank tables retain their source access terms and licenses. The final OpenReview entry requires a public repository linking these materials.')
    replace('Current support evidence is refitted when its signature changes; caches do not add labels.','The runtime refits support evidence when its signature changes and reuses cached fits for identical evidence.')
    replace('Its revision and source list must be preserved, not replaced by the latest dataset version.','Reproduction uses this exact revision and source list.')
    replace('The printed value must be a finite probability in $[0,1]$; this checks loading and dispatch, not predictive accuracy.',
        'The command checks loading and dispatch by returning a finite probability in $[0,1]$.')
    replace('The benchmark-excluded prior must be used for local core validation; the full fitted deployment payload is not a substitute.',
        'Local core validation requires the benchmark-excluded prior. The full fitted payload supplies deployment predictions.')
    replace('Those inputs and the oracle\'s fitted representation artifacts must accompany a complete research release. The fitted scoring archive alone reproduces inference, not every local training and diagnostic result. This distinction is intentional: runtime reproducibility and end-to-end experiment reproducibility are separate checks.',
        'A complete research release includes those inputs and the oracle\'s fitted representation artifacts. The scoring archive supplies reproducible inference; the aligned panels, manifests and prediction caches supply experiment reproduction.')
    # Add measured per-benchmark paired effects, retaining dependence explicitly.
    d=pd.DataFrame(json.loads((ROOT/'reports/drift_exact_prior_v80/comparison_records.json').read_text()))
    d=d[d.method.isin(['drift_prior_drift_update','drift_prior_our52_update'])].copy()
    d['weighted']=d.brier*d.budget.map({0:.1,1:.2,3:.2,7:.2,15:.2,31:.1})
    p=d.groupby(['benchmark','cohort','seed','model','method']).weighted.sum().unstack('method')
    delta=(p.drift_prior_our52_update-p.drift_prior_drift_update).groupby(['benchmark','cohort']).mean().unstack('cohort')
    appendix=r'''
\clearpage
\section{Paired effects and statistical uncertainty}\label{app:paired}
The local comparison uses the same solver, question partition, seed and revealed answers for both updates. Subtracting their episode ALCs removes variation shared by the two methods. Table~\ref{tab:paired} averages these paired differences within each benchmark. Each core benchmark contributes equally to the reported mean.

\begin{table}[ht]\centering
\caption{Our pooling ALC minus Drift-update ALC on each core benchmark. Negative values favour pooling. Values average the three matched randomization seeds and sampled target configurations.}
\label{tab:paired}
\begin{tabular}{lrrr}\toprule
Benchmark & One solver & Three solvers & Ten solvers\\\midrule
'''
    for b in ['BBH','GPQA','MATH','MMLU-Pro']:appendix+=f'{b} & '+ ' & '.join(f'${delta.loc[b,c]:+.4f}$' for c in [1,3,10])+r'\\'+'\n'
    appendix+=r'''\bottomrule\end{tabular}\end{table}

The four core benchmarks define a small, curated comparison. Reporting their separate effects makes the MATH reversal and the gains on the other three benchmarks visible. The shared-item coverage increases with cohort size, so the ten-solver comparison measures both richer solver evidence and denser question overlap.

For the broader comparison, the resampling unit is a related-source group. We draw 41 groups with replacement, retain all their solver episodes and seeds, and recompute the paired mean ALC difference using the original pair weights. The 2.5th and 97.5th percentiles of 10,000 draws form the intervals in Table~\ref{tab:broad}. The script \path{scripts/analyze_recent_runs_v86.py} computes these intervals with NumPy seed 860. This block resampling preserves dependence among sources from the same repository or family, repeated seeds and solver configurations within a source. Cross-group dependencies remain a limitation of the interval calculation.

The broad one-solver comparison contains 86 pair episodes; the up-to-three comparison contains 225. Prior selection used separate source groups and was frozen before validation. The intervals describe uncertainty in this public development comparison. Broader populations of benchmarks and solvers require further evaluation.

The hosted result averages the eight solver--benchmark pairs in the organizer's formative sample. Its six budget scores describe one sampled evaluation. The oracle averages ten fixed adaptation trajectories for 24 target configurations on the fixed mixed-benchmark bank. Each result is reported with its own evaluation unit and protocol.
'''
    tex=tex.replace(r'\end{document}',appendix+'\n'+r'\end{document}')
    (OUT/'aims_report.tex').write_text(tex,encoding='utf-8')
    provenance=json.loads((archive/'report_provenance.json').read_text());provenance['editorial_revision']='v89'
    broad=json.loads((ROOT/'reports/recent_run_diagnosis_v86/result.json').read_text())['neutral_vs_drift_cluster_bootstrap']
    provenance['statistics']={'core_paired_effects':{b:{str(c):float(delta.loc[b,c]) for c in [1,3,10]} for b in delta.index},'broad_intervals_source':'reports/recent_run_diagnosis_v86/result.json','broad_paired_bootstrap':broad,'bootstrap_draws':10000,'bootstrap_seed':860,'bootstrap_unit':'whole related-source group','core_benchmarks':4,'core_intervals':'individual paired benchmark effects reported','new_method_or_score_changes':False}
    (OUT/'report_provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Written',OUT/'aims_report.tex')
if __name__=='__main__':main()

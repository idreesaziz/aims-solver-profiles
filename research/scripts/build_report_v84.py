"""Visual/editorial revision. All measured values are read from frozen receipts."""
import os
os.environ['MPLBACKEND']='Agg'
import json, shutil, hashlib, zipfile
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
import build_report_v83 as previous

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/pdf/aims_report'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.labelsize':9,'axes.titlesize':10,'axes.titleweight':'bold'})

def canvas(height, xmax=12):
    fig,ax=plt.subplots(figsize=(6.5,height))
    ax.set(xlim=(0,xmax),ylim=(0,height)); ax.axis('off')
    fig.subplots_adjust(left=.015,right=.985,top=.98,bottom=.02)
    return fig,ax

def label(ax,x,y,t,**kw):
    ax.text(x,y,t,ha=kw.pop('ha','center'),va=kw.pop('va','center'),**kw)

def arrow(ax,a,b,**kw):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,
                               lw=1,color='black',shrinkA=3,shrinkB=3,zorder=4,**kw))

def box(ax,x,y,w,h,t,fc='white',size=9):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.03,rounding_size=0.04',
                              fc=fc,ec='black',lw=.85,zorder=2))
    label(ax,x+w/2,y+h/2,t,fontsize=size,zorder=3)

def finish(fig,name):
    fig.savefig(OUT/name,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

def fingerprint():
    fig,ax=canvas(1.95)
    # Deliberately constructed: equal column rates and different solver identities.
    m=np.array([[1,1,0,0,1,1,0,0],[1,1,0,0,1,1,0,0],
                [1,0,1,0,1,0,1,0],[1,0,1,0,1,0,1,0],
                [0,1,0,1,0,1,0,1],[0,1,0,1,0,1,0,1],
                [0,0,1,1,0,0,1,1],[0,0,1,1,0,0,1,1]])
    left=.65; bottom=.28; cw=.31;ch=.15
    for i in range(8):
        for j in range(8):
            ax.add_patch(Rectangle((left+j*cw,bottom+(7-i)*ch),cw,ch,
                                  fc='black' if m[i,j] else 'white',ec='.65',lw=.35))
    ax.add_patch(Rectangle((left,bottom),cw,8*ch,fill=False,ec='black',lw=1.8))
    ax.add_patch(Rectangle((left+cw,bottom),cw,8*ch,fill=False,ec='black',lw=1.8))
    label(ax,1.9,1.78,'A RESPONSE MATRIX',fontsize=8,weight='bold')
    label(ax,.25,.87,'Solvers',rotation=90,fontsize=8)
    label(ax,1.9,.10,'Questions',fontsize=8)
    arrow(ax,(3.35,.98),(4.1,.98))
    label(ax,7.4,1.77,'SAME RATE, DIFFERENT SOLVERS',fontsize=8,weight='bold')
    for k,j in enumerate([0,1]):
        x=4.45+k*1.14
        for i in range(8):
            ax.add_patch(Rectangle((x,bottom+(7-i)*ch),.45,ch,
                                  fc='black' if m[i,j] else 'white',ec='.6',lw=.4))
        label(ax,x+.225,.10,['A','B'][k],fontsize=8)
        label(ax,x+.225,1.57,'50%',fontsize=9,weight='bold')
    label(ax,9.1,1.03,'A mean counts successes.\nA fingerprint keeps\nwho succeeded.',fontsize=10,linespacing=1.5)
    label(ax,10.4,.23,'Illustrative outcomes',fontsize=7,color='.35')
    finish(fig,'fingerprints.pdf')

def architecture():
    fig,ax=canvas(2.45)
    label(ax,2.2,2.25,'BEFORE ANY ANSWERS',fontsize=8,weight='bold')
    label(ax,7.1,2.25,'AFTER ANSWERS ARE REVEALED',fontsize=8,weight='bold')
    box(ax,.12,1.12,1.8,.70,'Solver + item\nattributes',size=8)
    box(ax,2.38,1.12,2.0,.70,'Reproduced\nDrift prior',size=8)
    arrow(ax,(1.92,1.47),(2.38,1.47))
    box(ax,4.92,1.36,2.6,.68,'Item probability pooling\n+ solver correction',fc='.94',size=8)
    box(ax,4.92,.26,2.6,.68,'Gaussian hierarchical\ncorrection',size=8)
    arrow(ax,(4.38,1.48),(4.92,1.7))
    arrow(ax,(4.38,1.35),(4.92,.62))
    box(ax,8.2,.82,2.0,.72,'Brier stacking\nCross-fitted weight',fc='.94',size=8)
    arrow(ax,(7.52,1.7),(8.2,1.35));arrow(ax,(7.52,.60),(8.2,1.0))
    arrow(ax,(10.2,1.18),(11.15,1.18))
    label(ax,11.48,1.18,r'$\widehat p$',fontsize=17)
    label(ax,11.12,.66,'P(correct)',fontsize=8)
    label(ax,2.45,.40,'Fixed fitted parameters',fontsize=8,color='.3')
    label(ax,6.2,2.15,'Same-benchmark evidence enters both updates',fontsize=7)
    # Dashed brace-like evidence routing remains outside boxes.
    ax.plot([4.65,4.65],[.61,1.7],color='.3',lw=.8,ls=':')
    label(ax,8.5,.12,'Only revealed labels adapt the prediction.',fontsize=8,color='.3')
    finish(fig,'architecture.pdf')

def exclusion():
    fig,ax=canvas(1.72)
    label(ax,2.8,1.52,'CALIBRATING ONE SUPPORT ANSWER',weight='bold',fontsize=9)
    label(ax,2.5,.9,'Outcome being judged',fontsize=9)
    box(ax,1.1,.22,2.7,.45,r'Solver $s$, item $i$: $y_{si}$',fc='.9',size=9)
    arrow(ax,(4.1,.65),(5.2,.65))
    box(ax,5.4,.94,5.9,.46,'Benchmark shift: exclude every answer to item i',size=8)
    box(ax,5.4,.21,5.9,.46,'Item pooling: subtract this answer from its counts',size=8)
    label(ax,8.4,1.56,'TWO ROUTES FOR SELF-INFLUENCE',weight='bold',fontsize=9)
    label(ax,4.6,1.10,'hold out',fontsize=7)
    finish(fig,'exclusion.pdf')

def learning_curves(core):
    fig,axes=plt.subplots(1,3,figsize=(6.5,2.28),sharey=True,layout='constrained')
    for ax,c in zip(axes,[1,3,10]):
        for key,name,ls,m in [('drift_prior_drift_update','Drift update','--','s'),
                              ('drift_prior_our52_update','Our pooling','-','o')]:
            a=core['results'][str(c)][key]
            ax.plot(range(6),a['curve'],color='black',ls=ls,marker=m,ms=3,lw=1.15,label=name)
        ax.set(xticks=range(6),xticklabels=['0','1','3','7','15','31'],ylim=(0,.26),
               title=f'{c} solver'+('s' if c>1 else ''),xlabel='Revealed answers')
        ax.tick_params(labelsize=8);ax.grid(axis='y',color='.88',lw=.5)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Brier score (lower is better)');axes[0].legend(frameon=False,fontsize=7,loc='lower left')
    finish(fig,'local_curves.pdf')

def hosted():
    r=json.loads((ROOT/'reports/finalization_v82/official_budget_means.json').read_text())
    y=np.array([r['curve'][str(n)] for n in [0,1,3,7,15,31]])
    fig,(ax,b)=plt.subplots(1,2,figsize=(6.5,2.18),gridspec_kw={'width_ratios':[1.8,1]},layout='constrained')
    ax.axhline(.25,color='.4',ls=':',lw=1)
    ax.text(3.1,.254,'Constant 0.5 prediction',fontsize=7,color='.3')
    ax.fill_between(range(6),0,y,color='.93');ax.plot(range(6),y,'o-',color='black',lw=1.4,ms=4)
    ax.set(xticks=range(6),xticklabels=[0,1,3,7,15,31],ylim=(0,.32),xlabel='Revealed answers per solver',ylabel='Mean Brier score')
    ax.annotate(f'{y[0]:.3f}',(0,y[0]),xytext=(4,7),textcoords='offset points',fontsize=8)
    ax.annotate(f'{y[-1]:.3f}',(5,y[-1]),xytext=(-4,9),textcoords='offset points',ha='right',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    b.axis('off')
    b.text(.06,.92,'HOSTED FORMATIVE RUN',fontsize=9,weight='bold',transform=b.transAxes)
    b.text(.06,.62,f"{r['brier_alc']:.6f}",fontsize=24,weight='bold',transform=b.transAxes)
    b.text(.06,.43,'Brier ALC\n8 solver–benchmark pairs',fontsize=8,linespacing=1.4,transform=b.transAxes)
    b.text(.06,.12,'Area averages the entire curve.\nThe last point is Brier31.',fontsize=8,linespacing=1.5,transform=b.transAxes)
    finish(fig,'hosted_curve.pdf')

def oracle_diagram():
    fig,ax=canvas(2.5)
    label(ax,2,2.30,'OBSERVED BEHAVIOUR',fontsize=8,weight='bold')
    label(ax,6.1,2.30,'COMPACT QUESTION POSITION',fontsize=8,weight='bold')
    label(ax,10.4,2.30,'NEW SOLVER',fontsize=8,weight='bold')
    for i,v in enumerate([1,0,1,1,0,0,1,0]):
        ax.add_patch(Rectangle((.3+i*.39,1.35),.35,.38,fc='black' if v else 'white',ec='black',lw=.6))
    label(ax,1.85,1.05,'Successes, failures + missingness',fontsize=7)
    arrow(ax,(3.55,1.54),(4.4,1.54))
    box(ax,4.5,1.18,1.3,.70,'Encoder\n128 → 64',size=8)
    arrow(ax,(5.85,1.54),(6.3,1.54))
    for i in range(4):
        ax.add_patch(Rectangle((6.4+i*.42,1.35),.38,.38,fc=str(.2+.22*i),ec='black',lw=.6))
    label(ax,7.22,1.03,r'$q_0$    $q_1$    $q_2$    $q_3$',fontsize=9)
    arrow(ax,(8.2,1.54),(8.65,1.54))
    box(ax,8.75,1.18,2.9,.70,'Adapted solver readout\nP(correct)',size=8)
    box(ax,8.75,.21,2.9,.48,'31 revealed answers',fc='.94',size=8)
    arrow(ax,(10.2,.69),(10.2,1.18))
    box(ax,.35,.17,3.1,.48,'New question text',size=8)
    ax.plot([3.5,7.2,7.2],[.41,.41,.83],color='black',ls='--',lw=.85)
    label(ax,5.6,.65,'Missing link: infer this position',fontsize=8)
    finish(fig,'oracle.pdf')

def oracle_results(values):
    fig,(ax,b)=plt.subplots(1,2,figsize=(6.5,2.0),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    ax.bar(range(4),values,width=.48,color=['white','.8','.45','black'],edgecolor='black',lw=.9)
    ax.set(xticks=range(4),xticklabels=['1','2','3','4'],ylim=(0,.077),xlabel='Retained question coordinates',ylabel='Oracle Brier31')
    for j,v in enumerate(values):ax.text(j,v+.002,f'{v:.4f}',ha='center',fontsize=8)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.9',lw=.5);ax.set_axisbelow(True)
    b.set(xlim=(-1.4,1.4),ylim=(-1.1,1.1),xticks=[],yticks=[],
          xlabel='Question coordinate 1',ylabel='Question coordinate 2')
    b.set_title('Readout geometry (schematic)',fontsize=9)
    xx,yy=np.meshgrid(np.linspace(-1.4,1.4,200),np.linspace(-1.1,1.1,200))
    prob=1/(1+np.exp(-(xx+.65*yy)))
    b.contourf(xx,yy,prob,levels=np.linspace(0,1,11),cmap='Greys',vmin=0,vmax=1,alpha=.48)
    c=b.contour(xx,yy,prob,levels=[.25,.5,.75],colors='black',linewidths=.65)
    b.clabel(c,fmt={.25:'0.25',.5:'P = 0.5',.75:'0.75'},fontsize=7,inline=True)
    b.scatter([-.85,.82],[.38,.12],s=24,facecolors='white',edgecolors='black',zorder=5)
    b.annotate('A',(-.85,.38),xytext=(-10,8),textcoords='offset points',fontsize=8)
    b.annotate('B',(.82,.12),xytext=(5,-11),textcoords='offset points',fontsize=8)
    b.spines[['top','right']].set_visible(False)
    finish(fig,'oracle_results.pdf')

def main():
    backup=OUT/'archive_v83';backup.mkdir(exist_ok=True)
    for name in ['aims_report.pdf','aims_report.tex','report_audit.json']:
        if (OUT/name).exists() and not (backup/name).exists():shutil.copy2(OUT/name,backup/name)
    previous.main()
    core=json.loads((ROOT/'reports/drift_exact_prior_v80/comparison.json').read_text())
    p=json.loads((OUT/'report_provenance.json').read_text())
    vals=list(p['oracle'].values())
    fingerprint();architecture();exclusion();learning_curves(core);hosted();oracle_diagram();oracle_results(vals)
    t=(OUT/'aims_report.tex').read_text()
    t=t.replace(r'\usepackage{graphicx,booktabs,amsmath,amssymb}',r'\usepackage{graphicx,booktabs,amsmath,amssymb,array}')
    t=t.replace(r'\begin{document}\maketitle',r'\setlength{\textfloatsep}{13pt plus 2pt minus 2pt}'+'\n'+r'\begin{document}\maketitle')
    aa=t.index(r'\begin{abstract}');bb=t.index(r'\end{abstract}',aa)
    t=t[:aa]+r'''\begin{abstract}
How much can a few answers reveal about an unfamiliar AI solver? We start with a model of historical evaluation results, then update its predictions using answers from the solver and its peers. Our method pools question-level evidence, corrects for solver differences, and combines two updates according to held-out probability error. In a matched local benchmark-cold evaluation with ten solvers supplying evidence, it reaches Brier ALC 0.151. A separate oracle experiment compresses questions' historical solver outcomes into four informative coordinates. Recovering those coordinates for new questions remains an open problem.
'''+t[bb:]
    a=t.index(r'\section{The prediction problem}')
    b=t.index(r'\section{Result I: predictive performance}')
    technical=t[t.index(r'\subsection{The cross-benchmark starting point}'):b]
    method=r'''\section{A few answers, two kinds of evidence}
Two solvers can have the same accuracy while answering different questions correctly. Two questions can have the same difficulty while separating different solvers. An individual-response prediction needs more than either average.

\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{fingerprints.pdf}
\caption{A fingerprint records \emph{which} solvers succeed. The two highlighted columns have the same mean, but disagree about individual solvers. Rows are solver profiles; columns are question fingerprints. Outcomes are illustrative, with black indicating success.}
\label{fig:fingerprints}\end{figure}

The AIMS competition~\cite{aims} supplies a solver's configuration, question attributes and a limited set of revealed answers from an unfamiliar benchmark. The program predicts a success probability without running the target solver.

An answer says something about the solver, but also about the question. When other solvers have answered that question, their results supply evidence too. Our submitted method combines these signals. A separate oracle experiment asks how much of a question's behavioural fingerprint fits into four coordinates.

\clearpage
\section{The submitted method: probability pooling}
\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{architecture.pdf}
\caption{The deployed architecture. The reproduced prior stays fixed. Revealed answers feed two adaptation rules, and a weight selected on held-out support answers combines their probabilities. Shaded boxes identify our pooling and combination layer; arrows describe the prediction flow.}
\label{fig:architecture}\end{figure}

\subsection{A starting probability, not a verdict}
The starting point is Team Drift's released logistic prior~\cite{drift}. Solver metadata and question attributes enter an additive model,
\begin{equation}
p^0_{si}=\sigma\!\left(b+\mathbf w_s^\top\phi(s)+\mathbf w_i^\top\psi(i)\right).
\end{equation}
Here $s$ denotes a solver configuration and $i$ a question. Hashed features describe model names, harnesses, reasoning effort, text and tool metadata. The model supplies an initial probability; it does not invoke an LLM or require fresh question annotation.

We reproduce the released feature functions and fit the same regularized objective to 167 permitted public sources. The fitted artifact is retrained, not the historical weights. A disk-backed implementation avoids holding the complete feature matrix in memory. Exact feature settings, data provenance and reproduction checks appear in Appendix~\ref{app:details}.

\subsection{Learn about the question and the solver}
Suppose several solvers have attempted question $i$. Let $k_i$ be their revealed successes and $n_i$ their revealed trials. A benchmark-wide log-odds correction first adjusts the prior toward the evidence on this new benchmark. Call the resulting probability for solver $s$ on item $i$ $m_{si}$. We pool it with the observed question outcomes:
\begin{equation}
\widetilde p_{si}=\frac{\kappa m_{si}+k_i}{\kappa+n_i},\qquad\kappa=0.25.
\label{eq:pool-main}
\end{equation}
This is a weighted average of a starting probability and an observed success rate. The small prior weight allows even a few answers to change a mistaken prediction. It has the form of a Beta-binomial predictive mean~\cite{beta}; because different solvers are heterogeneous, we use it as regularized probability pooling, rather than claim one exact exchangeable Bayesian model.

The new solver also receives its own regularized log-odds correction. That correction can move it above or below the pooled probabilities. Thus the method learns about a question from available solvers, then learns how the target solver differs. Sharing is possible only when the supplied evidence actually includes other solvers; a one-solver episode offers much less of it.

\clearpage
\subsection{Do not let an answer judge itself}
Support answers serve two purposes: fitting the update and judging whether that update predicts well. Without exclusions, these roles can become circular. A success would raise its own starting probability and then appear to confirm the resulting prediction.

\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{exclusion.pdf}
\caption{Two exclusions protect solver calibration. When fitting against a support answer, the benchmark shift leaves out the entire item and the pooling counts leave out that answer. These exclusions are separate from the outer cross-fitting used to select the final combination.}
\label{fig:exclusion}\end{figure}

For each support answer, we therefore remove the answer from its item counts and exclude the whole item when computing its benchmark shift. Query predictions use all currently permitted evidence. Repeated responses count as trials; they do not create new question identities.

\subsection{Combine the updates on held-out answers}
Pooling is paired with a correlated Gaussian hierarchical update, extending the credited Drift baseline. Its benchmark, solver and question effects provide a different pattern of shrinkage. The solver covariance is fixed; it does not identify model families from their names or learn their true relationships.

Three folds of the revealed support answers provide predictions from both branches. Each fold refits both updates without its held labels. Write $a_t$ for the Gaussian prediction, $b_t$ for pooling, and $d_t=b_t-a_t$. The selected weight and final prediction are
\begin{equation}
w=\operatorname{clip}\!\left(\frac{\sum_t d_t(y_t-a_t)+1}{\sum_t d_t^2+1},0,1\right),
\qquad \widehat p=(1-w)a+wb.
\end{equation}
The weight minimizes held-out squared probability error, with penalty $(w-1)^2$ favouring pooling when the support set gives little reason to depart from it. This is a Brier-loss form of predictive stacking~\cite{stacking}. Scored query outcomes never select the weight. Both branches are then fitted to the available support set for prediction.

\subsection{What the runtime uses}
The submitted program is CPU-only and API-free. It requires the supplied subject and item attributes, the fitted prior, NumPy and SciPy. Evidence is restricted to the current benchmark. Subject identity uses the full visible configuration; item identity includes text, attributes and tool metadata. If the benchmark identifier is missing, only unscoped evidence from the identical subject is retained.

At zero observations, the output is simply the reproduced prior, clipped away from zero and one. At later budgets, the update is recomputed from the labels supplied at that budget. Memory caches save repeated work; they do not add evidence. This boundary is essential in an evaluation that scores the same query set at several observation budgets.

'''
    t=t[:a]+method+t[b:]
    t=t.replace(r'\section{Result I: predictive performance}',r'\clearpage'+'\n'+r'\section{Result I: what shared evidence buys}')
    t=t.replace("The target configurations are excluded from the forest controls' training outputs. The broad logistic prior can encounter their metadata on other permitted sources; this is not a strict unseen-name claim.",
                "The 24 target configurations supply adaptation and evaluation episodes. The broad logistic prior can encounter their metadata on other permitted sources; model names are not assumed unseen.")
    t=t.replace('The core local evaluation holds out BBH, GPQA, MATH and MMLU-Pro in turn.',
                'The core local evaluation tests BBH, GPQA, MATH and MMLU-Pro with their known source families excluded from the prior fit.')
    t=t.replace("The gain is smaller with one solver, where there are no other solvers' answers to pool.",
                "With one solver, ALC changes little and Brier31 is slightly worse (0.1886 versus 0.1876); there are no other solvers' answers to pool.")
    start=t.index(r'\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{results.pdf}')
    stop=t.index(r'\subsection{Broader benchmark transfer and hosted evaluation}',start)
    t=t[:start]+r'''\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{local_curves.pdf}
\caption{Matched local learning curves. Both updates receive the same prior, questions and revealed answers. Larger solver cohorts create more opportunities to share question-level evidence. Budget positions are equally spaced in $\log_2(n+1)$, as in the competition's ALC. Curves are means over the fixed development episodes; no uncertainty band is inferred from these averages.}
\label{fig:curves}\end{figure}

'''+t[stop:]
    t=t.replace(r'\subsection{Broader benchmark transfer and hosted evaluation}',r'\clearpage'+'\n'+r'\subsection{Beyond the favourable sharing regime}')
    aa=t.index('A separate source-group-held-out evaluation tests')
    bb=t.index('On this broader validation,',aa)
    t=t[:aa]+r'''A separate evaluation covers 43 additional public sources in 41 related-source groups. Entire groups, including related repositories and benchmark families, are excluded from the prior fit. Its 311 solver--benchmark episodes use two fixed seeds; validation groups are separate from prior-selection groups. The submitted update settings stay fixed.

Support and query texts are disjoint within each solver. Identical visible solver configurations share one budget, repeated response counts enter Brier correctly, and pairs receive equal weight. Inputs above 100,000 characters and episodes with fewer than 80 distinct texts are omitted. These restrictions and incomplete cross-repository deduplication limit the coverage claim.

'''+t[bb:]
    old='These results constrain the interpretation of the stronger core-benchmark numbers.'
    t=t.replace(old,old+r'''

\begin{table}[ht]\centering
\caption{Broader held-source-group validation: 43 sources, 41 groups. The same fixed pooling update is used with each starting prior. A neutral start is slightly better here.}
\begin{tabular}{lrr}\toprule
Starting prior & One solver ALC & Up to three solvers ALC\\\midrule
Reproduced Drift prior & 0.2170 & 0.2125\\
Constant 0.5 & 0.2153 & 0.2118\\\bottomrule
\end{tabular}\end{table}
''')
    host_start=t.index(r'Submission \texttt{1022433} achieves hosted')
    host_stop=t.index(r'\clearpage',host_start)
    t=t[:host_start]+r'''\subsection{The hosted result}
Submission \texttt{1022433} achieves hosted Brier ALC \textbf{0.190208}. Figure~\ref{fig:hosted} shows the entire formative learning curve, averaged equally over its eight solver--benchmark pairs. Its initial Brier score, 0.2779, is worse than the guaranteed 0.25 of predicting 0.5 everywhere. At 31 observations the mean is 0.1557. Strong adaptation therefore coexists with an imperfect starting prior.

\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{hosted_curve.pdf}
\caption{Actual hosted formative feedback. The shaded area illustrates the curve summarized by ALC; it is not an uncertainty band. Pair scores are equally weighted. This hidden sample differs from local validation, so the hosted and local numbers are not paired comparisons. The final common-subset summative evaluation is separate.}
\label{fig:hosted}\end{figure}

The local matched comparison isolates the adaptation change. The hosted run measures the complete program on a different sample. The broad validation tests a wider range of sources. Together they support a narrower claim than a single best score would suggest: question-level sharing can help substantially, especially with several solvers, while reliable cold-start probabilities remain a limitation.

'''+t[host_stop:]
    t=t.replace('A question\'s observed successes and failures across historical solvers determine its coordinates. A new solver\'s revealed answers adapt the readout. For unseen questions, the left-hand information is unavailable: recovering the coordinates without it is the open problem.',
    'A behavioural fingerprint and its missingness mask enter a small encoder. Four question coordinates interact with a solver-specific readout, adapted from revealed answers. The dashed path is the open problem: predicting the same coordinates from new question text. Illustrative cells show the information flow, not fitted coordinate values.')
    # Move the oracle's performance to a dedicated page with its own plot.
    a=t.index(r'\begin{table}[ht]\centering'+'\n'+r'\caption{Known question-position diagnostic.')
    b=t.index(r'The compact representation suggests',a)
    t=t[:a]+r'''\clearpage
\subsection{How small can a useful fingerprint be?}
\begin{figure}[ht]\centering\includegraphics[width=\linewidth]{oracle_results.pdf}
\caption{Left: measured privileged-position diagnostic; each readout is fitted and adapted under the same mixed-benchmark protocol. These are Brier31 scores, not ALC, and differ from the submission's benchmark-cold protocol. Right: a schematic two-coordinate slice of a logistic readout. Positions A and B are illustrative; contours show how question position changes one solver's predicted success. No semantic axis interpretation is claimed.}
\label{fig:oracle-results}\end{figure}

One retained direction reaches Brier31 0.06060. Two reach 0.05023, and three reach 0.04692. All four give 0.04693. The near equality of the last two results shows that additional coordinates need not improve this readout. It does not show that any arbitrarily selected three dimensions are sufficient: the lower-dimensional directions are constructed using source calibration.

'''+t[b:]
    # Sharpen the last conceptual paragraph; leave honest scope and credit intact.
    t=t.replace("The compact representation suggests a concrete research target.","The result gives the representation problem a sharper target.")
    t=t.replace('The executable method and the oracle answer different questions. The former predicts with the evidence available at deployment. The latter shows what a compact description of a question\'s observed behaviour can retain. Recovering that description for a new question remains the step needed to connect them.',
    'The two results meet at the same question: what can an answer tell us beyond its own cell in the response matrix? The submitted update transfers evidence through shared questions. The oracle transfers it through a compact behavioural representation. Recovering that representation for unseen questions remains the missing connection.')
    t=t.replace(r'\begin{thebibliography}{9}',r'\clearpage'+'\n'+r'\begin{thebibliography}{9}')
    # Provenance and implementation receipts belong with the reproduction specification.
    aa=t.index('All training outcomes come from permitted public Measurement')
    bb=t.index('The two results meet at the same question:',aa)
    provenance=t[aa:bb]
    t=t[:aa]+r'''Training uses permitted public Measurement Data Bank outcomes, with separate benchmark-excluded fits for local validation. The exact submitted archive passes the organizer's checker and isolated reference-parity tests. It runs without APIs, downloads or a GPU. Appendix~\ref{app:details} records data revisions, exclusions, implementation checks and reproduction requirements. OpenAI Codex assisted implementation, analysis and writing.

'''+t[bb:]
    # Technical detail is preserved, but no longer interrupts the main explanation.
    t=t.replace(r'\appendix'+'\n'+r'\section{Reproduction map}',r'\clearpage'+'\n'+r'\appendix'+'\n'+r'\section{Exact statistical specification}\label{app:details}'+'\n'+technical+'\n'+r'\section{Data provenance and implementation checks}'+'\n'+provenance+'\n'+r'\section{Reproduction map}')
    t=t.replace('The complete submission is CPU-only and API-free. It uses the supplied evidence afresh at every budget, filters it by benchmark, and distinguishes configurations by their full visible attributes. Memory caches accelerate repeated calls but carry no assumed evidence across checkpoints.','')
    t=t.replace('The oracle table is read directly from','The measured oracle figure is read directly from')
    t=t.replace('The report and code must be accompanied by an accessible release location when completing the competition entry.',
    'An accessible code-release URL must accompany the final OpenReview entry; this review copy does not yet supply a public release link.')
    t=t.replace(r'\end{document}',r'''
\subsection{Run the fitted predictor without training}
Python 3.12.10, NumPy 2.2.4 and SciPy 1.15.2 were used for the local archive checks. The fitted archive contains its payload and requires no raw dataset to run inference. After extracting it into a directory named \texttt{submission}, a minimal smoke test is:
\begin{verbatim}
python -m pip install numpy==2.2.4 scipy==1.15.2
cd submission
python -c "import model; print(model.predict([
{'normalized_name':'example','harness':'example'},
{'item_content':'What is 2+2?','benchmark_id':'demo'}], []))"
\end{verbatim}
The printed value must be a finite probability in $[0,1]$; this checks loading and dispatch, not predictive accuracy. In the development checkout, run \path{scripts/check_drift_pooling_zip_v82.py} against the exact archive for the stronger isolation and reference-parity checks described above. The organizer's \path{check_submission_zip.py} provides the interface check.

\subsection{Rebuild the experiments}
The deployment prior uses all 167 source names in the pinned upstream \path{report/TRAINING_SOURCES.json}. Its revision and source list must be preserved, not replaced by the latest dataset version. The disk-backed fitter downloads missing permitted public tables through Hugging Face, subject to dataset access and source terms. Its \texttt{DATA} and \texttt{OUT} paths identify the local data and feature-cache roots and must be set for another machine. The benchmark-excluded prior must be used for local core validation; the full fitted deployment payload is not a substitute.

The local evaluation scripts additionally depend on the aligned response panel, grouped split manifests and frozen prediction caches. Those inputs and the oracle's fitted representation artifacts must accompany a complete research release. The fitted scoring archive alone reproduces inference, not every local training and diagnostic result. This distinction is intentional: runtime reproducibility and end-to-end experiment reproducibility are separate checks.

\end{document}
''')
    (OUT/'aims_report.tex').write_text(t,encoding='utf-8')
    p['editorial_revision']='v84';p['figures']='monochrome vector; measured plots use frozen results; schematic outcomes explicitly marked'
    (OUT/'report_provenance.json').write_text(json.dumps(p,indent=2))
    print('Revised report source:',OUT/'aims_report.tex')

if __name__=='__main__':main()

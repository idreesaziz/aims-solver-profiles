"""Final visual refinement of v84. No changes to measured results or method."""
import json,shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import build_report_v84 as base

OUT=base.OUT

def architecture():
    fig,ax=base.canvas(2.45)
    ax.set_ylim(0,4)
    base.label(ax,2.25,3.7,'FIXED STARTING MODEL',fontsize=9,weight='bold')
    base.label(ax,7.8,3.7,'ADAPTATION FROM REVEALED ANSWERS',fontsize=9,weight='bold')
    base.box(ax,.08,1.95,1.9,.90,'Solver + item\nattributes',size=9)
    base.box(ax,2.48,1.95,1.95,.90,'Reproduced\nDrift prior',size=9)
    base.arrow(ax,(1.98,2.40),(2.48,2.40))
    base.box(ax,5.12,2.45,2.7,.95,'Item pooling\n+ solver correction',fc='.94',size=9)
    base.box(ax,5.12,.90,2.7,.95,'Gaussian\nhierarchical update',size=9)
    # One visible fork, with arrows ending outside the box borders.
    ax.plot([4.45,4.76],[2.4,2.4],color='black',lw=1)
    ax.plot([4.76,4.76],[1.375,2.925],color='black',lw=1)
    base.arrow(ax,(4.76,2.925),(5.12,2.925))
    base.arrow(ax,(4.76,1.375),(5.12,1.375))
    base.box(ax,8.55,1.70,2.05,.95,'Brier stacking\nCross-fitted\nweight',fc='.94',size=8.8)
    base.arrow(ax,(7.82,2.925),(8.55,2.40))
    base.arrow(ax,(7.82,1.375),(8.55,1.93))
    base.arrow(ax,(10.60,2.175),(11.28,2.175))
    base.label(ax,11.60,2.175,r'$\widehat p$',fontsize=18)
    base.label(ax,11.35,1.60,'P(correct)',fontsize=8.5)
    base.label(ax,6,.27,"Both updates use only the current benchmark's revealed answers.",fontsize=8.5,color='.25')
    base.finish(fig,'architecture.pdf')

def exclusion():
    fig,ax=base.canvas(1.72)
    base.label(ax,2.5,1.52,'ONE SUPPORT ANSWER',weight='bold',fontsize=9)
    base.label(ax,8.35,1.52,'EXCLUDE ITS OWN EVIDENCE',weight='bold',fontsize=9)
    base.label(ax,2.5,1.03,'Outcome being judged',fontsize=9)
    base.box(ax,1.08,.35,2.8,.48,r'Solver $s$, item $i$: $y_{si}$',fc='.94',size=9)
    ax.plot([3.92,4.65],[.59,.59],color='black',lw=1)
    ax.plot([4.65,4.65],[.44,1.155],color='black',lw=1)
    base.arrow(ax,(4.65,1.155),(5.20,1.155))
    base.arrow(ax,(4.65,.44),(5.20,.44))
    base.box(ax,5.20,.925,6.35,.46,'Benchmark shift: exclude every answer to item i',size=8.5)
    base.box(ax,5.20,.21,6.35,.46,'Item pooling: subtract this answer from its counts',size=8.5)
    base.finish(fig,'exclusion.pdf')

def oracle_diagram():
    fig,ax=base.canvas(2.5)
    base.label(ax,1.95,2.30,'OBSERVED BEHAVIOUR',fontsize=8.5,weight='bold')
    base.label(ax,6.1,2.30,'COMPACT QUESTION POSITION',fontsize=8.5,weight='bold')
    base.label(ax,10.4,2.30,'NEW SOLVER',fontsize=8.5,weight='bold')
    for i,v in enumerate([1,0,1,1,0,0,1,0]):
        ax.add_patch(Rectangle((.3+i*.39,1.35),.35,.38,fc='black' if v else 'white',ec='black',lw=.6))
    base.label(ax,1.85,1.05,'Successes, failures + missingness',fontsize=8)
    base.arrow(ax,(3.55,1.54),(4.4,1.54))
    base.box(ax,4.5,1.18,1.3,.70,'Encoder\n128 → 64',size=9)
    base.arrow(ax,(5.85,1.54),(6.3,1.54))
    for i in range(4):
        ax.add_patch(Rectangle((6.4+i*.42,1.35),.38,.38,fc=str(.2+.22*i),ec='black',lw=.6))
        base.label(ax,6.59+i*.42,1.04,f'$q_{i}$',fontsize=10)
    base.arrow(ax,(8.2,1.54),(8.65,1.54))
    base.box(ax,8.75,1.18,2.9,.70,'Adapted solver readout\nP(correct)',size=8.8)
    base.box(ax,8.75,.21,2.9,.48,'31 revealed answers',fc='.94',size=9)
    base.arrow(ax,(10.2,.69),(10.2,1.18))
    base.box(ax,.35,.17,3.1,.48,'New question text',size=9)
    ax.plot([3.5,7.2,7.2],[.41,.41,.68],color='black',ls='--',lw=.85)
    base.arrow(ax,(7.2,.68),(7.2,.90),linestyle='--')
    base.label(ax,5.32,.70,'Open: infer this position',fontsize=8.5)
    base.finish(fig,'oracle.pdf')

def oracle_results(values):
    fig,(ax,b)=plt.subplots(1,2,figsize=(6.5,2.05),gridspec_kw={'width_ratios':[1.4,1]},layout='constrained')
    ax.bar(range(4),values,width=.48,color=['white','.8','.45','black'],edgecolor='black',lw=.9)
    ax.set(xticks=range(4),xticklabels=['1','2','3','4'],ylim=(0,.077),xlabel='Retained question coordinates',ylabel='Oracle Brier31')
    for j,v in enumerate(values):ax.text(j,v+.002,f'{v:.4f}',ha='center',fontsize=8.5)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',color='.9',lw=.5);ax.set_axisbelow(True)
    b.set(xlim=(-1.4,1.4),ylim=(-1.1,1.1),xticks=[],yticks=[],xlabel='Question coordinate 1',ylabel='Question coordinate 2')
    b.set_title('Readout geometry (schematic)',fontsize=9)
    xx,yy=np.meshgrid(np.linspace(-1.4,1.4,200),np.linspace(-1.1,1.1,200))
    prob=1/(1+np.exp(-(xx+.65*yy)))
    b.contourf(xx,yy,prob,levels=np.linspace(0,1,11),cmap='Greys',vmin=0,vmax=1,alpha=.48)
    c=b.contour(xx,yy,prob,levels=[.25,.5,.75],colors='black',linewidths=.65)
    # Explicit in-panel label positions: the automatic 0.25 label was clipped.
    b.clabel(c,fmt={.25:'0.25',.5:'P = 0.5',.75:'0.75'},fontsize=7.5,inline=True,
             manual=[(-.90,-.305),(0.,0.),(.75,.54)])
    b.scatter([-.85,.82],[.38,.12],s=24,facecolors='white',edgecolors='black',zorder=5)
    b.annotate('A',(-.85,.38),xytext=(-10,8),textcoords='offset points',fontsize=8.5)
    b.annotate('B',(.82,.12),xytext=(6,-11),textcoords='offset points',fontsize=8.5)
    b.spines[['top','right']].set_visible(False)
    base.finish(fig,'oracle_results.pdf')

def main():
    backup=OUT/'archive_v84';backup.mkdir(exist_ok=True)
    for p in OUT.iterdir():
        if p.is_file() and p.suffix in {'.pdf','.tex','.json','.zip'} and not (backup/p.name).exists():
            shutil.copy2(p,backup/p.name)
    base.main()
    # Use the same numerical inputs; update only visual presentation.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                         'axes.labelsize':9,'axes.titlesize':10,'axes.titleweight':'bold'})
    architecture();exclusion();oracle_diagram()
    provenance=json.loads((OUT/'report_provenance.json').read_text())
    oracle_results(list(provenance['oracle'].values()))
    tex=(OUT/'aims_report.tex').read_text(encoding='utf-8')
    tex=tex.replace(r'\begin{document}\maketitle',r'\renewcommand{\arraystretch}{1.16}'+'\n'+r'\begin{document}\maketitle')
    tex=tex.replace(r'\begin{thebibliography}{9}',r'\begin{thebibliography}{9}\raggedright')
    # Keep URLs on their own lines; avoid breaking a protocol across two lines.
    aa=tex.index(r'\begin{thebibliography}');bb=tex.index(r'\end{thebibliography}',aa)
    bibliography=tex[aa:bb].replace(r'\raggedright',r'\raggedright\urlstyle{same}')
    bibliography=bibliography.replace(r'. \url{',r'.\newline\url{')
    tex=tex[:aa]+bibliography+tex[bb:]
    tex=tex.replace(r'\section{Reproduction map}',r'\clearpage'+'\n'+r'\section{Reproduction map}')
    (OUT/'aims_report.tex').write_text(tex,encoding='utf-8')
    provenance['editorial_revision']='v85';provenance['visual_pass']='150dpi full-page + 220dpi figure inspection'
    (OUT/'report_provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    print('Visual revision:',OUT/'aims_report.tex')

if __name__=='__main__':main()

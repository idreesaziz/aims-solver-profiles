"""Existing v74 update generalized to subject-specific starting probabilities.

Identical rows reproduce v74. Raw item counts and regularization are unchanged;
the only difference is which subject's prior supplies each likelihood offset.
This remains an approximate pooling procedure, not exact joint Bayes.
"""
import numpy as np
from scipy.special import logit,expit
from models.final_runtime_v74 import BetaFit
from models.statistical_adaptation_v70 import intercept
from models.correlated_profile_adaptation_v67 import fit,predict

class SubjectBeta(BetaFit):
    def __init__(self,p0,support,labels,kappa,setting):
        super().__init__(p0.mean(0),support,labels,kappa,setting)
        self.solver_shift=[]
        for j,(ids,y) in enumerate(zip(support,labels)):
            at=np.array([self.lookup[int(i)] for i in ids],int)
            m=expit(logit(p0[j,ids])+self.item_shift[at])
            base=(kappa*m+self.success[at]-y)/(kappa+self.counts[at]-1)
            self.solver_shift.append(float(intercept(logit(np.clip(base,1e-5,1-1e-5))[None,:],y[None,:],16.)[0]))

def current(p0,support,ys):
    count=len(support);ids=np.concatenate(support);sid=np.concatenate([np.repeat(j,len(s)) for j,s in enumerate(support)])
    covariance=(.1*np.eye(count)+.9*np.ones((count,count)))[None,:,:]
    return fit(logit(p0[sid,ids]),np.concatenate(ys),sid,ids,covariance)

class Adaptation:
    def __init__(self,p0,support,ys,kappa=.25,setting='pseudo4'):
        p0=np.clip(np.asarray(p0,float),1e-5,1-1e-5)
        assert p0.ndim==2 and len(p0)==len(support)
        self.old=current(p0,support,ys);self.beta=SubjectBeta(p0,support,ys,kappa,setting)
        aa=[];bb=[];truth=[]
        folds=min(3,max(map(len,support),default=0))
        for fold in range(folds):
            held=[np.arange(len(s))%folds==fold for s in support]
            ss=[s[~h] for s,h in zip(support,held)];yy=[y[~h] for y,h in zip(ys,held)]
            a=current(p0,ss,yy);b=SubjectBeta(p0,ss,yy,kappa,setting)
            for j,(s,h) in enumerate(zip(support,held)):
                q=s[h]
                if not len(q):continue
                aa.extend(predict(p0[j,q],q,j,a)[0]);bb.extend(b.predict(p0[j,q],q,j));truth.extend(ys[j][h])
        aa=np.array(aa);bb=np.array(bb);truth=np.array(truth);d=bb-aa
        self.weight=float(np.clip((d@(truth-aa)+1)/(d@d+1),0,1))
    def predict(self,base,ids,subject):
        a=predict(base,ids,subject,self.old)[0];b=self.beta.predict(base,ids,subject)
        return (1-self.weight)*a+self.weight*b

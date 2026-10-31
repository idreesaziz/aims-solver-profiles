"""Alternative small-sample priors for the global beta-pooling calibration.

Gaussian, Student-t, logistic pseudo-observations and Firth correction are
standard methods. Their integration with cross-fitted item pooling is tested
here; no inference guarantees or novelty of these priors are claimed.
"""
import numpy as np
from scipy.special import expit,logit
from models.statistical_adaptation_v70 import intercept

SETTINGS={
    'gaussian0.0625':('gaussian',.0625,0.),
    'gaussian0.25':('gaussian',.25,0.),
    'gaussian1':('gaussian',1.,0.),
    'gaussian4':('gaussian',4.,0.),
    'cauchy2.5':('student',1.,2.5),
    'student3_2.5':('student',3.,2.5),
    'firth':('firth',0.,0.),
    'firth_ridge0.25':('firth',.25,0.),
    'pseudo1':('pseudo',1.,0.),
    'pseudo4':('pseudo',4.,0.),
}


def objective(z,y,b,setting):
    kind,a,scale=SETTINGS[setting]
    eta=z+b[:,None];p=expit(eta);v=p*(1-p)
    value=np.sum(np.logaddexp(0,eta)-y*eta,axis=1)
    grad=np.sum(p-y,axis=1);hess=np.sum(v,axis=1)
    if kind=='gaussian':
        value+=.5*a*b*b;grad+=a*b;hess+=a
    elif kind=='student':
        den=a*scale*scale+b*b
        value+=.5*(a+1)*np.log1p(b*b/(a*scale*scale))
        grad+=(a+1)*b/den;hess+=(a+1)*(a*scale*scale-b*b)/(den*den)
    elif kind=='firth':
        info=np.maximum(np.sum(v,axis=1),1e-100)
        d=np.sum(v*(1-2*p),axis=1)
        dd=np.sum(v*((1-2*p)**2-2*v),axis=1)
        value+=-.5*np.log(info)+.5*a*b*b
        grad+=-.5*d/info+a*b
        hess+=-.5*(dd/info-(d/info)**2)+a
    elif kind=='pseudo':
        mean=np.clip(expit(z).mean(1),1e-8,1-1e-8)
        ref=logit(mean)+b;pp=expit(ref)
        value+=a*(np.logaddexp(0,ref)-mean*ref)
        grad+=a*(pp-mean);hess+=a*pp*(1-pp)
    return value,grad,hess


def fit_shift(z,y,setting):
    z=np.asarray(z,float);y=np.asarray(y,float)
    assert z.shape==y.shape and z.ndim==2
    if z.shape[1]==0:return np.zeros(len(z))
    if SETTINGS[setting][0]=='gaussian':return intercept(z,y,SETTINGS[setting][1])
    starts=[-6.,0.,6.] if SETTINGS[setting][0] in ['student','firth'] else [0.]
    best=np.zeros(len(z));bestvalue=np.full(len(z),np.inf)
    for start in starts:
        b=np.full(len(z),start)
        for _ in range(150):
            val,g,h=objective(z,y,b,setting)
            if np.max(abs(g),initial=0)<1e-8:break
            step=np.clip(-g/np.where(h>1e-10,h,.05),-3.,3.)
            rate=np.ones(len(z))
            for _ in range(30):
                trial=b+rate*step
                value=objective(z,y,trial,setting)[0]
                accept=value<=val+1e-4*rate*g*step+1e-12
                if accept.all():break
                rate=np.where(accept,rate,rate*.5)
            b+=rate*step
        val,g,_=objective(z,y,b,setting)
        assert np.max(abs(g),initial=0)<2e-5,(setting,np.max(abs(g)))
        better=val<bestvalue;best[better]=b[better];bestvalue[better]=val[better]
    return best


def predictions(p0,support,labels,query,kappa,settings=None):
    if settings is None:settings=SETTINGS
    p0=np.clip(np.asarray(p0,float),1e-5,1-1e-5)
    iid=np.concatenate(support);ys=np.concatenate(labels);keys=np.unique(iid)
    lookup={int(i):j for j,i in enumerate(keys)}
    counts=np.array([np.sum(iid==i) for i in keys],float)
    success=np.array([ys[iid==i].sum() for i in keys],float)
    u=len(keys);n=len(support[0]);assert all(len(s)==n for s in support)
    if u:
        frac=success/counts;z=logit(p0[keys])
        if u>1:
            keep=~np.eye(u,dtype=bool)
            loo_z=np.broadcast_to(z,(u,u))[keep].reshape(u,u-1)
            loo_y=np.broadcast_to(frac,(u,u))[keep].reshape(u,u-1)
    outputs={}
    for setting in settings:
        b=fit_shift(z[None,:],frac[None,:],setting)[0] if u else 0.
        bi=fit_shift(loo_z,loo_y,setting) if u>1 else np.zeros(u)
        prior=np.clip(expit(logit(p0[keys])+bi),1e-5,1-1e-5)
        bases=[]
        for ids,y in zip(support,labels):
            at=np.array([lookup[int(i)] for i in ids],int)
            bases.append((kappa*prior[at]+success[at]-y)/(kappa+counts[at]-1))
        shift=intercept(logit(np.clip(np.array(bases).reshape(len(support),n),1e-5,1-1e-5)),np.array(labels,float),16.) if n else np.zeros(len(support))
        pred=[]
        for s,q in enumerate(query):
            m=expit(logit(p0[q])+b)
            k=np.zeros(len(q));nn=np.zeros(len(q))
            for j,i in enumerate(q):
                if int(i) in lookup:
                    at=lookup[int(i)];k[j]=success[at];nn[j]=counts[at]
                    m[j]=expit(logit(p0[i])+bi[at])
            m=np.clip(m,1e-5,1-1e-5)
            pooled=(kappa*m+k)/(kappa+nn)
            pred.append(np.clip(expit(logit(np.clip(pooled,1e-5,1-1e-5))+shift[s]),.001,.999))
        outputs[setting]=pred
    return outputs

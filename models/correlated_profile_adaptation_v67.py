"""Correlated Gaussian priors on new-solver benchmark offsets.

Extends Team Drift's hierarchical logistic MAP structure (Alex Alvord / Cloud
Lab Works LLC, MIT,2026). Replaces independent solver offsets by a covariance
constructed from historical-response profiles or current revealed fingerprints.
General correlated/multitask priors are established methods, not claimed novel.
"""
import numpy as np
from scipy.special import expit


def correlation(gram):
    """PSD Gram -> unit diagonal, zero-information rows independent."""
    gram=np.asarray(gram,float)
    gram=.5*(gram+gram.T)
    norms=np.sqrt(np.maximum(np.diag(gram),0))
    active=norms>1e-8
    out=np.zeros_like(gram)
    out[np.ix_(active,active)]=gram[np.ix_(active,active)]/np.outer(norms[active],norms[active])
    out=.5*(out+out.T)
    np.fill_diagonal(out,1.)
    assert np.isfinite(out).all()
    assert np.linalg.eigvalsh(out).min()>-1e-8
    return out


def historical_gram(labels,weights):
    """Question x historical-model outcomes, remove difficulty and model mean."""
    y=np.asarray(labels,float);w=np.asarray(weights,float);w=w/w.sum()
    residual=y-y.mean(1,keepdims=True)
    residual-=w@residual
    return correlation((residual*w[:,None]).T@residual)


def observed_gram(subject_ids,item_ids,labels,base,subjects):
    """Current observed solver fingerprints; missing cells contribute zero.

    Center each solver's observed residuals to reduce mere strength effects.
    No pairwise deletion covariance: a single Gram guarantees PSD.
    """
    sid=np.asarray(subject_ids,int);iid=np.asarray(item_ids,int)
    y=np.asarray(labels,float);p=np.asarray(base,float)
    unique=list(dict.fromkeys(iid.tolist()));lookup={i:j for j,i in enumerate(unique)}
    matrix=np.zeros((subjects,len(unique)))
    for s in range(subjects):
        take=sid==s
        values=y[take]-p[take]
        if len(values):values=values-values.mean()
        for i,v in zip(iid[take],values):matrix[s,lookup[int(i)]]=v
    return correlation(matrix@matrix.T)


def fit(offset,labels,subject_ids,item_ids,covariances,sweeps=12):
    """Parallel MAP fits for fixed covariance candidates; no query labels.

    delta ~ N(0,.8^2*K), beta~N(0,1.5^2), d_item~N(0,1.5^2).
    K=identity reproduces upstream12-sweep fit with neutral ability exactly.
    Correlated delta block gets a Newton update, with step scaling/backtracking.
    """
    z=np.asarray(offset,float);y=np.asarray(labels,float)
    sid=np.asarray(subject_ids,int);iid=np.asarray(item_ids,int)
    kernels=np.asarray(covariances,float)
    count,subjects,_=kernels.shape
    assert z.shape==y.shape==sid.shape==iid.shape and np.isin(y,[0,1]).all()
    assert np.isfinite(z).all() and np.isfinite(kernels).all()
    assert np.linalg.eigvalsh(kernels).min()>0
    precision=np.linalg.inv(kernels)/(.8**2)
    keys=list(dict.fromkeys(iid.tolist()))
    by_s=[np.flatnonzero(sid==s) for s in range(subjects)]
    by_i=[np.flatnonzero(iid==i) for i in keys]
    beta=np.zeros(count);delta=np.zeros((count,subjects));item=np.zeros((count,len(keys)))
    eta=np.broadcast_to(z,(count,len(z))).copy()
    for _ in range(sweeps):
        p=expit(eta)
        step=np.clip(((y-p).sum(1)-beta/(1.5**2))/(1/(1.5**2)+(p*(1-p)).sum(1)),-1.5,1.5)
        beta+=step;eta+=step[:,None]
        p=expit(eta);v=p*(1-p)
        gradient=np.stack([(y[ix]-p[:,ix]).sum(1) for ix in by_s],axis=1)
        curvature=np.stack([v[:,ix].sum(1) for ix in by_s],axis=1)
        gradient-=np.einsum('mij,mj->mi',precision,delta)
        hessian=precision.copy()
        hessian[:,np.arange(subjects),np.arange(subjects)]+=curvature
        step=np.linalg.solve(hessian,gradient[...,None])[...,0]
        scale=np.minimum(1.,1.5/np.maximum(np.max(abs(step),axis=1),1e-20))
        # Identity control must retain upstream coordinatewise step clipping.
        # Since its Hessian is diagonal and groups disjoint, update is separable.
        identity=np.max(abs(kernels-np.eye(subjects)),axis=(1,2))<1e-14
        step[identity]=np.clip(step[identity],-1.5,1.5)
        scale[identity]=1.
        old=(np.logaddexp(0,eta)-y*eta).sum(1)+.5*np.einsum('mi,mij,mj->m',delta,precision,delta)
        for _ls in range(30):
            candidate=delta+scale[:,None]*step
            new_eta=eta+scale[:,None]*step[:,sid]
            value=(np.logaddexp(0,new_eta)-y*new_eta).sum(1)+.5*np.einsum('mi,mij,mj->m',candidate,precision,candidate)
            bad=(value>old+1e-10)&~identity
            if not bad.any():break
            scale[bad]*=.5
        else:raise AssertionError('Correlated-offset line search failed')
        delta=candidate;eta=new_eta
        for j,ix in enumerate(by_i):
            p=expit(eta[:,ix]);v=p*(1-p)
            step=np.clip(((y[ix]-p).sum(1)-item[:,j]/(1.5**2))/(1/(1.5**2)+v.sum(1)),-1.5,1.5)
            item[:,j]+=step;eta[:,ix]+=step[:,None]
    return dict(beta=beta,delta=delta,item=item,lookup={k:j for j,k in enumerate(keys)})


def predict(base,query_items,target,fitted):
    p=np.clip(np.asarray(base,float),1e-4,1-1e-4)
    z=np.log(p/(1-p))[None,:]+fitted['beta'][:,None]+fitted['delta'][:,target,None]
    for at,i in enumerate(query_items):
        if int(i) in fitted['lookup']:z[:,at]+=fitted['item'][:,fitted['lookup'][int(i)]]
    return np.clip(expit(np.clip(z,-30,30)),.01,.99)
